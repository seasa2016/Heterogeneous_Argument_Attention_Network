import math
from util.MST import decode_mst
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

def fscore_binary(ys, ts, adu_len, max_n_spans):
    def convert(index, l):
        #print(index)
        eye = torch.eye(l, dtype=torch.long)

        root_row = (index == max_n_spans)
        index[root_row] = 0
        #print(index)

        data = eye[index]
        data[root_row] = torch.zeros(l, dtype=torch.long)
        for i in range(l):
            data[i, i] = -1
        #print(data)
        data = data[data > -1]
        #print(data)
        return data

    t_all = []
    y_all = []

    start = 0
    for l in adu_len:
        t_flat = convert(ts[start:start+l].clone(), l).view(-1).tolist()
        t_all.extend(t_flat)

        y_flat = convert(ys[start:start+l].clone(), l).view(-1).tolist()
        y_all.extend(y_flat)

        start += l

    f1_link = f1_score(t_all, y_all, pos_label=1)
    f1_no_link = f1_score(t_all, y_all, pos_label=0)
    return f1_link, f1_no_link

def persuasive_cal_score(output, label, criterion, alpha=(0, 0)):
    """
    label:
        'adu_direct':[datas[0]['author'], datas[1]['author']],
        'para_direct':[datas[0]['para_author'], datas[1]['para_author']],
    """
    adu_alpha, para_alpha = alpha

    loss, stat = 0, {}
    device = output[0][0].device
    # persuasive part
    persuasive_preds = (output[1][0]-output[0][0]).view(-1)
    persuasive_labels = torch.ones_like(persuasive_preds).to(device)
    loss += criterion(persuasive_preds, persuasive_labels)

    
    # direct part
    criterion = nn.BCEWithLogitsLoss()

    # adu direct part
    adu_direct = []
    adu_labels = []
    adu_preds = []
    for i in range(2):
        index = (label['adu_direct'][i]>0).view(-1)
        adu_labels.append(label['adu_direct'][i].view(-1)[index])
        adu_preds.append(output[i][1].view(-1)[index])
    
    adu_preds = torch.cat(adu_preds, dim=-1).to(device)
    adu_labels = (torch.cat(adu_labels, dim=-1)-1).long().to(device)
    loss += adu_alpha*criterion(adu_preds, adu_labels.float())

    # para direct part
    adu_direct = []
    para_labels = []
    para_preds = []
    for i in range(2):
        index = (label['para_direct'][i]>0).view(-1)
        para_labels.append(label['para_direct'][i].view(-1)[index])
        para_preds.append(output[i][2].view(-1)[index])

    para_preds = torch.cat(para_preds, dim=-1).to(device)
    para_labels = (torch.cat(para_labels, dim=-1)-1).long().to(device)
    loss += para_alpha*criterion(para_preds, para_labels.float())

    outputs = [persuasive_preds.detach().cpu(), adu_preds.detach().cpu(), para_preds.detach().cpu()]
    labels = [persuasive_labels.detach().cpu(), adu_labels.detach().cpu(), para_labels.detach().cpu()]
    # 'persuasive', 'adu_direct', 'para_direct'
    return loss, outputs, labels


def tree_cal_score(output, label, adu_len=None, alpha=(0, 0)):
    ac_type_alpha, link_type_alpha = alpha
    
    device = output['link'].device
    if(label['link'] is None):
        mask = torch.tril(torch.ones(13, 13),-1)[adu_len].view(-1)
    else:
        mask = (label['link']>=0).view(-1)
    index = torch.arange(0, mask.shape[0], dtype=torch.long).to(device).masked_select(mask)

    #y_link_mst = decode_mst(output['link'].softmax(-1).detach(), adu_len)
    y_link_mst = output['link'].max(-1)[1]
    y_link_mst = y_link_mst.view(-1)[index]

    if((label['link'] is None) and (label['type'] is None) and (label['link_type'] is None)):
        return {
                'link_mst':y_link_mst.detach().cpu(), 'link':output['link'].detach().cpu(), 'type':output['type'].detach().cpu(), 'link_type':output['link_type'].detach().cpu()}
    #print('here?')
    stat = {}   
    temp_link = label['link'].clone().view(-1)

    loss = 0
    # link target score
    criterion = nn.CrossEntropyLoss()
    
    batch_size, n_spans, link_len = output['link'].shape
    mask = (label['link']>=0).view(-1)
    index = torch.arange(0, mask.shape[0], dtype=torch.long).to(device).masked_select(mask)
    output['link'] = output['link'].view(-1, link_len)[index]
    label['link'] = label['link'].view(-1)[index]
    loss_link = criterion(output['link'], label['link'])

    stat['link'] = loss_link.detach().cpu().item()
    loss += (1 - ac_type_alpha - link_type_alpha)*loss_link

    # type loss
    criterion = nn.BCEWithLogitsLoss()
    
    batch_size, n_spans = output['type'].shape
    mask = (label['type']>=0).view(-1)
    index = torch.arange(0, mask.shape[0], dtype=torch.long).to(device).masked_select(mask)
    output['type'] = output['type'].view(-1)[index]
    label['type'] = label['type'].view(-1)[index]

    loss_type = criterion(output['type'], label['type'].float())
    stat['type'] = loss_type.detach().cpu().item()

    loss += ac_type_alpha*loss_type

    # link type loss
    # output['link_type'] contains all pair's relation score, we should extract out what we need
    criterion = nn.BCEWithLogitsLoss()
    batch_size, n_spans, _ = output['link_type'].shape

    second_index = (label['link_type'].view(-1)[index]<2)
    if(second_index.sum().cpu().item() == 0):
        stat['link_type'] = 0
        loss_link_type = 0
    else:
        output['link_type'] = output['link_type'].view(-1, _)[index]

        ex_index = ((torch.arange(0, batch_size, dtype=torch.long).to(device)*_).unsqueeze(-1) + temp_link).view(-1)[index]

        output['link_type'] = output['link_type'].view(-1)[ex_index][second_index]
        label['link_type'] = label['link_type'].view(-1)[index][second_index]
        #print(output['link_type'].view(-1)[ex_index][second_index].shape)
        #print(label['link_type'].view(-1)[index][second_index].shape)
        #output['link_type'] = output['type']
        #label['link_type'] = label['type']

        loss_link_type = criterion(output['link_type'], label['link_type'].float())
        stat['link_type'] = loss_link_type.detach().cpu().item()

        loss += link_type_alpha*loss_link_type
    
    return loss, stat, {
        'link_mst':y_link_mst.detach().cpu(), 'link':output['link'].detach().cpu(), 'type':output['type'].detach().cpu(), 'link_type':output['link_type'].detach().cpu()
    }

def tree_evaluate(pred, label, adu_len):
    stat = {'acc':{}, 'macro_f':{}, 'link':{}, 'link_mst':{}, 'type':{}, 'link_type':{}}

    macro_f_scores = []
    ###########################
    # link prediction results #
    ###########################
    if('link' in pred):
        pred_link = pred['link']
        ts_link = label['link']
        stat['acc']['link'] = (pred_link == ts_link).float().mean().item()
        """
        f_link = fscore_binary(pred_link, ts_link, adu_len, self.max_n_spans)
        stat['link']['link'], stat['link']['nolink'] = f_link
        stat['macro_f']['link'] = sum(f_link)/2

        macro_f_scores.append( stat['macro_f']['link'] )
        """
    if('link_mst' in pred):
        pred_link_mst = pred['link_mst']
        stat['acc']['link_mst'] = (pred_link_mst == ts_link).float().mean().item()
        """
        f_link = fscore_binary(pred_link_mst, ts_link, adu_len, self.max_n_spans)
        stat['link_mst']['link'], stat['link_mst']['nolink'] = f_link
        stat['macro_f']['link_mst'] = sum(f_link)/2
        
        macro_f_scores.append( stat['macro_f']['link_mst'] )
        """
    pred_link_sorts = pred['link_sort']
    stat['mrr_link'] = 0
    for t, pred_link_sort in zip(ts_link, pred_link_sorts):
        for index, t_gold in enumerate(pred_link_sort):
            if(t_gold == t):
                stat['mrr_link'] += 1/(index+1)
    stat['mrr_link'] /= len(pred_link_sorts)


    ##############################
    # ac_type prediction results #
    ##############################
    if('type' in pred):
        pred_type = pred['type'].long()
        ts_type = label['type'].long()
    
        stat['acc']['type'] = ((pred_type == ts_type).float().mean().item())
        
        f_type = [f1_score(ts_type==i, pred_type==i, average='binary') for i in range(2)]
        stat['type']['premise'], stat['type']['claim'] = f_type
        stat['macro_f']['type'] = sum(f_type)/len(f_type)

        #macro_f_scores.append( stat['macro_f']['type'] )
        
    ################################
    # link type prediction results #
    ################################
    if('link_type' in pred):
        pred_link_type  = pred['link_type'].long()
        ts_link_type = label['link_type'].long()

        stat['acc']['link_type'] = (pred_link_type == ts_link_type).float().mean()
        f_link_type = [f1_score(ts_link_type==i, pred_link_type==i, average='binary') for i in range(2)]
        
        stat['link_type']['support'], stat['link_type']['attack'] = f_link_type
        stat['macro_f']['link_type'] = sum(f_link_type)/len(f_link_type)

        #macro_f_scores.append( stat['macro_f']['link_type'] )
        
    #stat['macro_f']['total'] = sum(macro_f_scores)/len(macro_f_scores)

    return stat
