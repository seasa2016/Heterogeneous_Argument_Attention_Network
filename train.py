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
        add_graph_args, add_optim_args, add_trainer_args, add_embed_args, add_dataset_args)

from util.loss import (hinge)
from optim import RAdam, Ranger

from util.parsing import tree_cal_score, tree_evaluate, persuasive_cal_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    parser = add_encoder_args(parser)
    parser = add_graph_args(parser)
    parser = add_optim_args(parser)
    parser = add_trainer_args(parser)
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)

def build_data(args, persuasive_path, tree_path):
    batch_size = args.batchsize

    # need to check
    num_worker=6
    persuasive = []
    data_path, embed_path, pair_path = persuasive_path
    if(args.total):
        print('use full data to train')
        temp = [('', '_all', True, batch_size), ('_dev', '_all', False, batch_size*8), ('_test', '_test', False, batch_size*8)]
    else:
        temp = [('_train', '_all', True, batch_size), ('_dev', '_all', False, batch_size*8), ('_test', '_test', False, batch_size*8)]

    for dtype, embtype, shuffle, b in temp:
        dataset = persuasiveDataset(data_path=data_path+dtype, embed_path=embed_path+embtype, pair_path=pair_path+dtype, top=args.top, train=shuffle, direct=args.direct)
        #dataset = persuasiveDataset(data_path=data_path+dtype, embed_path=embed_path+embtype, pair_path=pair_path+dtype, train=False)
        dataloader = DataLoader(dataset, batch_size=b, shuffle=shuffle, num_workers=num_worker, collate_fn=persuasive_collate_fn)
        persuasive.append(dataloader)
    
    if(args.multitask):
        tree = []
        data_path, embed_path = tree_path
        for dtype, shuffle in [('_train', True), ('_dev', False), ('_test', False)]:
            dataset = treeDataset(data_path=data_path+dtype, embed_path=embed_path)
            dataloader = DataLoader(dataset, batch_size=batch_size*4, shuffle=shuffle, num_workers=num_worker, collate_fn=tree_collate_fn)
            tree.append(dataloader)
    else:
        tree = [None, None, None]

    return persuasive, tree

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
    

def train(args, persuasive_data_iter, tree_data_iter, model, criterion, device, multitask=False):
    # initial for training 
    model.train()
    # build up optimizer
    if(args.optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer == 'AdamW'):
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer == 'Ranger'):
        optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer == 'Radam'):
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=args.lr*1000, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    grad_clip = args.grad_clip
    save_path = args.save_path
    accumulate = args.accumulate
    print_every = 100*accumulate
    eval_every = 25*accumulate
    #print_every, eval_every = 2, 2

    total_epoch = args.epoch*len(persuasive_data_iter[0])
    print('total training step:', total_epoch)
    persuasive_datas = iter(persuasive_data_iter[0])
    if((tree_data_iter[0] is not None) and multitask):
        tree_datas = iter(tree_data_iter[0])

    multi_alpha = (args.ac_type_alpha, args.link_type_alpha)
    direct_alpha = (args.adu_alpha, args.para_alpha)
    alpha = (direct_alpha, multi_alpha)
    tree_count = args.tree_count
    best_acc = [0, 0]


    # start training
    model.zero_grad()
    t = time.time()
    persuasive = [[[], [], []], [[], [], []]]
    tree_preds = {'label':collections.defaultdict(list), 'pred':collections.defaultdict(list), 'loss':collections.defaultdict(float), 'count':0}
    for count in range(1, total_epoch+1):
        try:
            datas = next(persuasive_datas)
        except:
            persuasive_datas = iter(persuasive_data_iter[0])
            datas = next(persuasive_datas)
        
        outputs = []
        for data in datas:
            data = convert(data, device)
            pred = model(**data)
            outputs.append(pred)
        labels = {
            'adu_direct':[datas[0]['author'], datas[1]['author']],
            'para_direct':[datas[0]['para_author'], datas[1]['para_author']],
        }
        # simply compare two value
        loss, outputs, labels = persuasive_cal_score(outputs, labels, criterion, direct_alpha)
        
        for i, (p, l) in enumerate(zip(outputs, labels)):
            persuasive[0][i].append(p)
            persuasive[1][i].append(l)

        loss.backward()

        if(multitask and (count%tree_count==0)):
            try:
                data, label = next(tree_datas)
            except:
                tree_datas = iter(tree_data_iter[0])
                data, label = next(tree_datas)
            data = convert(data, device)
            output = model(**data, multitask=True)
            output = {'type':output[0], 'link':output[1], 'link_type':output[2]}
            label = convert(label, device)
            loss, loss_stat, output = tree_cal_score(output, label, None, multi_alpha)
            loss.backward()

            for key, val in label.items():
                tree_preds['label'][key].append( val.detach().cpu() )
            for key, val in loss_stat.items():
                tree_preds['loss'][key] += val
            update_pred(tree_preds['pred'], output, data['adu_length'])
            tree_preds['count'] += 1

        if(count%accumulate==0):
            #utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        if(count % eval_every==0):
            stat = update(persuasive, criterion, dtype='persuasive')

            nt = time.time()
            print('now:{}, time: {:.4f}s'.format(count, nt - t),
                '\npersuasive: [loss: {:.4f}, diff: {:.4f}, acc: {:.4f}]'.format(stat[0][0], stat[0][1], stat[0][2]),
                '\tdirect: [adu_loss: {:.4f}, adu_acc: {:.4f}, para_loss: {:.4f}, para_acc: {:.4f}]'.format(stat[1][0], stat[1][1], stat[2][0], stat[2][1]),
                flush=True)

            if(multitask):
                stat = update(tree_preds, dtype='tree')
                print('acc: [link_mst: {:.4f}, link: {:.4f}, type: {:.4f}, link_type: {:.4f}]'. format(
                    stat['acc']['link_mst'], stat['acc']['link'], stat['acc']['type'], stat['acc']['link_type']
                ))
                print('f1: type: [premise: {:.4f}, claim: {:.4f}], link_type: [support{:.4f}, attack: {:.4f}]'. format(
                    stat['type']['premise'], stat['type']['claim'], stat['link_type']['support'], stat['link_type']['attack']
                ))
                print('mrr: {:.4f}'.format(stat['mrr_link']), flush=True)
            t = nt
            
            persuasive = [[[], [], []], [[], [], []]]
            tree_preds = {'label':collections.defaultdict(list), 'pred':collections.defaultdict(list), 'loss':collections.defaultdict(float), 'count':0}
            scheduler.step()
       
        if(count % print_every == 0):
            dev_acc = test('dev {}'.format(count), persuasive_data_iter[1], tree_data_iter[1], model, criterion, device, alpha)
            test_acc = test('test {}'.format(count), persuasive_data_iter[2], tree_data_iter[2], model, criterion, device, alpha)
            if(dev_acc>best_acc[0]):
                best_acc = [dev_acc, test_acc]
            torch.save(model.state_dict(), save_path+'/check_{}.pt'.format(count))   
    print('all finish with acc:', best_acc)


def test(epoch, persuasive_iter, tree_iter, model, criterion, device, alpha=[(0,0), (0,0)]):
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
            """
            if(i==2):
                break
            """

    nt = time.time()
    stat = update(total_preds, criterion, 'persuasive')
    print(epoch,'time: {:.4f}s'.format(nt - t),
        '\npersuasive: [loss: {:.4f}, diff: {:.4f}, acc: {:.4f}]'.format(stat[0][0], stat[0][1], stat[0][2]),
        '\tdirect: [adu_loss: {:.4f}, adu_acc: {:.4f}, para_loss: {:.4f}, para_acc: {:.4f}]'.format(stat[1][0], stat[1][1], stat[2][0], stat[2][1]),
        flush=True)
    persuasive_stat = stat
    # do tree parsing analysis
    if(tree_iter is not None):
        tree_preds = {'label':collections.defaultdict(list), 'pred':collections.defaultdict(list), 'loss':collections.defaultdict(float), 'count':0}
        for i, (data, label) in enumerate(tree_iter):
            data = convert(data, device)
            output = model(**data, multitask=True)
            output = {'type':output[0], 'link':output[1], 'link_type':output[2]}
            
            label = convert(label, device)
            loss, loss_stat, output = tree_cal_score(output, label, None, alpha[1])

            for key, val in label.items():
                tree_preds['label'][key].append( val.detach().cpu() )
            for key, val in loss_stat.items():
                tree_preds['loss'][key] += val
            update_pred(tree_preds['pred'], output, data['adu_length'])
            tree_preds['count'] += 1
            
        stat = update(tree_preds, dtype='tree')
        print('acc: [link_mst: {:.4f}, link: {:.4f}, type: {:.4f}, link_type: {:.4f}]'. format(
            stat['acc']['link_mst'], stat['acc']['link'], stat['acc']['type'], stat['acc']['link_type']
        ))
        print('f1: type: [premise: {:.4f}, claim: {:.4f}], link_type: [support{:.4f}, attack: {:.4f}]'. format(
            stat['type']['premise'], stat['type']['claim'], stat['link_type']['support'], stat['link_type']['attack']
        ))
        print('mrr: {:.4f}'.format(stat['mrr_link']))
    
    model.train()
    return persuasive_stat[0][2]

def main():
    args = parse_args()
    if(not os.path.isdir(args.save_path)):
        os.makedirs(args.save_path)
    else:
        print('file exist', file=sys.stderr)
        #raise ValueError('file exist')
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
    print('*'*10)
    print(args)
    print('*'*10)
    multi_alpha = (args.ac_type_alpha, args.link_type_alpha)
    direct_alpha = (args.adu_alpha, args.para_alpha)
    alpha = (direct_alpha, multi_alpha)
    
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
    print('accumulate: ', args.accumulate)
    if(args.criterion =="bce" ):
        criterion = nn.BCEWithLogitsLoss()
    elif(args.criterion == 'hinge'):
        criterion = hinge
    else:
        raise ValueError('no this loss function')
    


    train(args, persuasive, tree, model, criterion, device, args.multitask)   
    torch.save(model.state_dict(), args.save_path+'/check_last.pt')  
    print("Optimization Finished!")
    print('dev:')
    test('End', persuasive[1], tree[1], model, criterion, device, alpha=alpha)
    print('test:')
    test('End', persuasive[2], tree[2], model, criterion, device, alpha=alpha)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing

if(__name__ == '__main__'):
    main()
