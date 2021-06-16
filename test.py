from __future__ import division
from __future__ import print_function

import time
import numpy as np
import os
from sklearn.metrics import f1_score
import argparse
import collections
import random
import sys

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from torch.utils.data import DataLoader

from data import (persuasiveDataset, persuasive_collate_fn, treeDataset, tree_collate_fn)
from model.model import ArgTree
from util.parameter import (add_default_args, add_default_args, add_encoder_args, add_parsing_args,
        add_graph_args, add_optim_args, add_test_args, add_embed_args, add_dataset_args)

from util.loss import (hinge)
from optim import RAdam, Ranger

from util.parsing import tree_cal_score, tree_evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    parser = add_encoder_args(parser)
    parser = add_graph_args(parser)
    parser = add_optim_args(parser)
    parser = add_test_args(parser)
    parser = add_embed_args(parser)
    parser = add_dataset_args(parser)
    parser = add_parsing_args(parser)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args

def check_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if(args.cuda):
        torch.cuda.manual_seed(args.seed)

def build_data(args, persuasive_path, tree_path):
    batch_size = args.batchsize

    # need to check
    num_worker=6
    data_path, embed_path, pair_path = persuasive_path
    dataset = persuasiveDataset(data_path=data_path, embed_path=embed_path, pair_path=pair_path, top=args.top, train=False, direct=args.direct)
    persuasive_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, collate_fn=persuasive_collate_fn)
        
    
    if(args.multitask):
        tree = []
        num_worker=0
        data_path, embed_path = tree_path
        dataset = treeDataset(data_path=data_path+'_test', embed_path=embed_path)
        tree_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, collate_fn=tree_collate_fn)
    else:
        tree_dataloader = None

    return persuasive_dataloader, tree_dataloader

def convert(data, device):
    if(isinstance(data, dict)):
        temp = {}
        for key in data:
            try:
                temp[key] = data[key].to(device)
            except:
                pass
        return temp
    elif(isinstance(data, list)):
        return [_.to(device) for _ in data]

def update(outputs, criterion=None, dtype=None):
    stat = {}
    if(dtype=='tree'):
        label = {}
        for key, val in outputs['label'].items():
            label[key] = torch.cat([_.view(-1) for _ in val], dim=-1)
        
        pred = {}
        pred['link_sort'] = []
        for _ in outputs['pred']['link_sort']:
            pred['link_sort'].extend(_)
        
        for key in ['link', 'type', 'link_type', 'adu_len', 'link_mst']:
            pred[key] = torch.cat([_.view(-1) for _ in outputs['pred'][key]], dim=0)

        stat = tree_evaluate(pred, label, pred['adu_len'])
        stat['loss'] = {}
        for key, val in outputs['loss'].items():
            stat['loss'][key] = val/outputs['count']

    elif(dtype=='persuasive'):
        stat = []
        # persuasive
        pred = torch.cat(outputs[0][0], dim=-1).view(-1)
        label = torch.cat(outputs[1][0], dim=-1).view(-1)
        stat.append( (criterion(pred, label), pred.mean(), (pred>0).float().mean()))
        
        # direction
        criterion = nn.BCEWithLogitsLoss()
        for i in range(1, 3):
            pred = torch.cat(outputs[0][i], dim=-1).view(-1)
            label = torch.cat(outputs[1][i], dim=-1).view(-1)
            stat.append( 
                    (criterion(pred, label.float()), ((pred>0).long()==label).float().mean())
                )
    return stat

def update_pred(pred, output, adu_len):
    for key in ['link_mst']:
        pred[key].append(output[key])
    #for key in ['type', 'link_type']:
    pred['type'].append( (output['type'] > 0).long() )
    pred['link_type'].append( (output['link_type'].sigmoid()>0.3).long() )
    
    pred['link_sort'].append(output['link'].sort(-1, descending=True)[1])
    pred['link'].append( output['link'].max(-1)[1] )
    pred['adu_len'].append( adu_len )
    
def test(args, epoch, persuasive_iter, tree_iter, model, criterion, device, alpha=[(0,0), (0,0)]):
    model.eval()
    t = time.time()

    total_preds = [[[], [], []], [[], [], []]]
    with torch.no_grad():
        for i, datas in enumerate(persuasive_iter):
            outputs = []
            for data in datas:
                data = convert(data, device)
                pred = model(**data)
                outputs.append(pred)

            labels = [
                0,
                [datas[0]['author'].cuda(), datas[1]['author'].cuda()],
                [datas[0]['para_author'].cuda(), datas[1]['para_author'].cuda()]
            ]
        
            p = (outputs[1][0]-outputs[0][0]).view(-1)
            total_preds[0][0].append(p)
            total_preds[1][0].append(torch.ones_like(p).to(p.device))

            for j in range(1, 3):
                for k in range(2):
                    index = (labels[j][k]>0).view(-1)
                    total_preds[0][j].append( outputs[k][j].view(-1)[index] )
                    total_preds[1][j].append( (labels[j][k].view(-1)[index]-1) )
            
            

    nt = time.time()
    torch.save(total_preds, args.pred_path)
    stat = update(total_preds, criterion, 'persuasive')
    print(epoch,'time: {:.4f}s'.format(nt - t),
        '\npersuasive: [loss: {:.4f}, diff: {:.4f}, acc: {:.4f}]'.format(stat[0][0], stat[0][1], stat[0][2]),
        '\tdirect: [adu_loss: {:.4f}, adu_acc: {:.4f}, para_loss: {:.4f}, para_acc: {:.4f}]'.format(stat[1][0], stat[1][1], stat[2][0], stat[2][1]),
        flush=True)
    persuasive_stat = stat
    # do tree parsing analysis
    
    model.train()
    return persuasive_stat[0][2]

def main():
    args = parse_args()
    check_seed(args)

    # Model and optimizer
    model = ArgTree(args)
    print(model, flush=True)
    if(args.cuda):
        model = model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('finish build model')
    model.load_state_dict(torch.load(args.model_path))
    

    print('finish load model')
    
    print('*'*10)
    print(args)
    print('*'*10)
    
    # Load data
    persuasive_path = (args.data_path, args.elmo_path, args.pair_path)
    tree_path = (None, None)
    if(args.multitask):
        tree_path = (args.parsing_path+'/CMV', args.parsing_path+'/CMVELMo.hdf5')

    persuasive, tree = build_data(args, persuasive_path, tree_path)

    print('finish build data')
    # Train model
    t_total = time.time()
    print(args.criterion) 
    if(args.criterion =="bce" ):
        criterion = nn.BCEWithLogitsLoss()
    elif(args.criterion == 'hinge'):
        criterion = hinge
    else:
        raise ValueError('no this loss function')
    
    print('test:')
    test(args, 'End', persuasive, tree, model, criterion, device)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing

if(__name__ == '__main__'):
    main()
