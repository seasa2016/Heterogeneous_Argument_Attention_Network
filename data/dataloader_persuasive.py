import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import collections
import copy
import h5py
import numpy as np
import json
import random
import os

class persuasiveDataset(Dataset):
    def __init__(self, data_path, embed_path, pair_path, max_adu=128, top=3, train=False, direct='left'):
        self.top = top
        self.train = train

        self.elmo_path = [embed_path+'_pos.hdf5', embed_path+'_neg.hdf5']

        self.data = {}
        save_data_path = '{}_{}_{}.pt'.format(data_path, top, direct)
        print('use ', save_data_path)
        if(os.path.isfile(save_data_path)):
            self.data = torch.load(save_data_path)
        else:
            with open(data_path) as f:
                for line in f:
                    temp = json.loads(line)
                    index, side, post_index = [int(_) for _ in temp['uid'].split('_')]
                    
                    if(index not in self.data):
                        self.data[index] = [{}, {}]


                    #for key in ['link_score', 'link_type_score', 'type_score']:
                    #    temp[key] = torch.tensor(temp[key], dtype=torch.float)
                    for key in ['pre_mask', 'last_mask', 'shell_span', 'span', 'para_span', 'ac_position_info', 'author', 'para_author', 'adu_label']:
                        temp[key] = torch.tensor(temp[key], dtype=torch.long)

                    link_graph = []
                    for map_type in range(3):
                        graph = collections.defaultdict(float)
                        if(direct=='left' or direct=='both'):
                            for a, link_rank in enumerate(temp['link_rank'][map_type]):
                                for b, val in link_rank[:self.top]:
                                    graph[(a, b)] = max(val, graph[(a, b)])
                                graph[(a, a)] = 1
                        if(direct=='right' or direct=='both'):
                            for a, link_rank in enumerate(temp['link_rank'][map_type]):
                                for b, val in link_rank[:self.top]:
                                    graph[(b, a)] = max(val, graph[(b, a)])
                                graph[(a, a)] = 1
                                
                        link_graph.append(graph)
                    del temp['link_rank']
                    temp['graph'] = link_graph

                    temp['elmo_path'] = self.elmo_path
                    temp['adu_length'] = len(temp['span'])
                    temp['para_length'] = len(temp['elmo_index'])

                    temp['file_index'] = side

                    self.data[index][side][post_index] = temp
                torch.save(self.data, save_data_path)

        with open(pair_path) as f:
            self.pair = json.loads(f.readline())
        if(train):
            mapping = {'cat':{}, 'list':{}}
            temp = []
            for index, pos, neg in self.pair:
                if(index not in mapping['cat']):
                    mapping['cat'][index] = len(temp)
                    mapping['list'][ len(temp) ] = index
                    temp.append([])
                    
                index = mapping['cat'][index]
                temp[index].append((pos, neg))
            for _ in temp:
                random.shuffle(_)
                
            self.pair = temp
            self.count = [0] * len(self.pair)
            self.mapping = mapping
    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        """ 
        'topic_index', 'elmo_index', 
        'shell_span', 'span', 'para_span'
        'mask', 'adu_label', 'ac_position_info', 'uid', 
        'sum_mask', 
        'link_rank', 'reply_rank'
        """
        if(self.train):
            index = self.mapping['list'][idx]
            t = self.count[idx]
            self.count[idx] = ((self.count[idx]+1) % len(self.pair[idx]))
            neg_post_index, pos_post_index = self.pair[idx][t]
        else:
            index, neg_post_index, pos_post_index = self.pair[idx]

        data_pair = []

        for side, post_index in enumerate([neg_post_index, pos_post_index]):
            sample = {}
            
            #for key in ['uid', 'elmo_index', 'topic_index', 'graph', 'sum_mask', 'shell_span', 'span', 'ac_position_info']:
            #   sample[key] = self.data[index][side][post_index][key]

            data_pair.append(copy.deepcopy(self.data[index][side][post_index]))
            data_pair[-1]['elmo_path'] = self.elmo_path
        return data_pair

def persuasive_collate_fn(src):
    """
        'elmo_path', 'topic_index', 'elmo_index', 'adu_length',
        'shell_span', 'span', 'ac_position_info', 'uid',
        'mask', 'adu_label', 'ac_position_info', 'uid', 
        'sum_mask', 'link_rank'
    """
    def padding(data, dtype, val=0):
        # first find max in every dimension
        size = len(data[0].shape)

        temp_len = np.array( [ _.shape for _ in data] )
        max_len = [len(data)] + temp_len.max(axis=0).tolist()

        temp = torch.zeros(max_len, dtype=dtype)+val
        if(size == 4):
            for i in range(len(data)):
                temp[i, :temp_len[i][0], :temp_len[i][1], :temp_len[i][2], :temp_len[i][3]] = data[i]
        elif(size == 3):
            for i in range(len(data)):
                temp[i, :temp_len[i][0], :temp_len[i][1], :temp_len[i][2]] = data[i]
        elif(size == 2):
            for i in range(len(data)):
                temp[i, :temp_len[i][0], :temp_len[i][1]] = data[i]
        elif(size==1):
            for i in range(len(data)):
                temp[i, :temp_len[i][0]] = data[i]
        else:
            raise ValueError('no this size {size}')
        return temp

    # convert
    outputs = []

    for side in range(2):
        data = dict()
        for key in src[0][side]:
            data[key] = [ _[side][key] for _ in src]
        
        span_copy = copy.deepcopy([ _.tolist() for _ in data['span']])

        output = dict()

        # preprocess elmo and topic embedding
        elmo_file = h5py.File( data['elmo_path'][0][side], 'r')
        #elmo_file = [ h5py.File( _, 'r') for _ in data['elmo_path'][0]]
            
        topic_emb = [
            torch.tensor(elmo_file.get(str(_))[()], dtype=torch.float)
            for index, _ in zip(data['file_index'], data['topic_index'])
        ]
        output['topic_length'] = torch.tensor( [ _.shape[1] for _ in topic_emb])
        output['topic_emb'] = padding(topic_emb, dtype=torch.float)

        elmo_emb = []
        elmo_length = []
        adu_length = []

        acc = 0
        for i, (index, post) in enumerate(zip(data['file_index'], data['elmo_index'])):
            for _ in post:
                elmo_emb.append(
                    torch.tensor(elmo_file.get(str(_))[()], dtype=torch.float)
                )
                elmo_length.append(
                    elmo_emb[-1].shape[1]
                )
            data['shell_span'][i][:, 0] += acc
            data['span'][i][:, 0] += acc
            data['para_span'][i][:, 0] += acc

            acc += len(post)
        elmo_file.close()

        output['elmo_emb'] = padding(elmo_emb, dtype=torch.float)
        output['elmo_length'] = torch.tensor(elmo_length, dtype=torch.long)
        output['adu_length'] = torch.tensor(data['adu_length'], dtype=torch.long).view(-1)
        output['para_length'] = torch.tensor(data['para_length'], dtype=torch.long).view(-1)

        for key in ['shell_span', 'span', 'para_span', 'ac_position_info', 'author', 'para_author', 'adu_label']:
            output[key] = padding(data[key], dtype=torch.long)
        
        length_max = output['para_length'].max()
        output['para_mask'] = torch.stack([
            torch.cat([torch.ones(l), torch.zeros(length_max-l)], dim=-1) for l in output['para_length']
            ])
        length_max = output['adu_length'].max()
        output['adu_mask'] = torch.stack([
            torch.cat([torch.ones(l), torch.zeros(length_max-l)], dim=-1) for l in output['adu_length']
            ])

        # uid
        output['uid'] = data['uid']

        # build up link graph
        # adu graph
        batch_size, max_adu_len = len(output['uid']), output['adu_length'].max().item()+1
        max_reply_len = max([len(elmo_index) for elmo_index in data['elmo_index']])+max_adu_len

        # link between adu
        adj = []
        for map_type in range(3):
            temp_graph = torch.stack([torch.eye(max_reply_len, dtype=torch.float) for _ in range(batch_size)])

            for i, link_graph in enumerate(data['graph']):
                for pos, val in link_graph[map_type].items():
                    temp_graph[i, pos[0], pos[1]] = val

            t_inv = 1/(temp_graph.sum(-1))
            t_inv[  torch.isinf(t_inv) ] = 0
            t_diag = torch.stack([ torch.diag(_) for _ in t_inv])

            adj.append(torch.matmul(t_diag, temp_graph))
        output['adu_graph'] = torch.stack(adj, dim=0)
        
        
        # link between adu and paragraph
        reply_graph = torch.stack([torch.eye(max_reply_len, dtype=torch.float) for _ in range(batch_size)])
        for i, (span, l) in enumerate(zip(span_copy, data['adu_length'])):
            for j, (index, _, _) in enumerate(span[:l]):
                #reply_graph[i, j, index+max_adu_len] = 1
                reply_graph[i, index+max_adu_len, j] = 1

        for i, reply_ranks in enumerate(data['reply_rank']):
            #print(len(data['elmo_index'][i]))
            for j, reply_rank in enumerate(reply_ranks):
                for k in reply_rank:
                    reply_graph[i, j+max_adu_len, k+max_adu_len] = 1
                    reply_graph[i, k+max_adu_len, j+max_adu_len] = 1
                
        t_inv = 1/(reply_graph.sum(-1))
        t_inv[  torch.isinf(t_inv) ] = 0
        t_diag = torch.stack([ torch.diag(_) for _ in t_inv])
        reply_graph = torch.matmul(t_diag, reply_graph)
        output['reply_graph'] = reply_graph

        # 'sum_mask'
        """
        max_elmo_len = max([ len(elmo_index) for elmo_index in data['elmo_index']])
        
        for key in ['pre_mask', 'last_mask']:
            mask = torch.zeros(batch_size, max_elmo_len, dtype=torch.float)
            for i, (start, end) in enumerate(data[key]):
                mask[i, start:end] = 1
            output[key] = mask
        """

        outputs.append(output)

    return outputs


if(__name__ == '__main__'):
    data_path = './../../preprocess/cmv_raw_origin_full_final/tree/train_3/data_test'
    embed_path = './../../preprocess/cmv_raw_origin_full_final/tree/cmv_elmo_test'
    pair_path = './../../preprocess/cmv_raw_origin_full_final/tree/train_3/graph_pair_test'

    random.seed(39)
    np.random.seed(39)
    torch.manual_seed(39)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(39)

    dataloader = persuasiveDataset(data_path=data_path, embed_path=embed_path, pair_path=pair_path)
    print('----------')
    for key in dataloader[0][0]:
        try:
            print(key, dataloader[0][0][key].shape)
        except:
            print(dataloader[0][0][key])

    print('----------')
    batch_size = 2
    train_dataloader = DataLoader(dataloader, batch_size=batch_size,shuffle=True, num_workers=2,collate_fn=persuasive_collate_fn)
    for i, datas in enumerate(train_dataloader):
        #if(i%1000==0):
        print(i)
        if(i==0):
            #torch.save(data, './test_data')
            for key in datas[0]:
                try:
                    print(key, datas[0][key].shape, datas[0][key].float().mean())
                except:
                    pass
                    #print(datas[0][key])
            """
            print()
            print((datas[0]['adu_graph']>0).int().tolist())
            print()
            print((datas[0]['reply_graph']>0).int().sum(-1).tolist())
            print()
            """
            
        pass
