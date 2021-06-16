import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import copy
import h5py
import numpy as np
import json

class treeDataset(Dataset):
    def __init__(self, data_path, embed_path, max_adu=128):
        self.data = []
        with open(data_path) as f:
            for line in f:
                temp = json.loads(line)
                if( len(temp['span'])>max_adu):
                    continue
                if(len(temp['span'])==0 or len(temp['shell_span'])==0):
                    print(temp['uid'])
                    continue
                    print(temp['span'])
                    print(temp['shell_span'])
                    temp['span'] = [[0, 0, 0]]
                    temp['shell_span'] = [[0, 0, 0]]
                    temp['ac_position_info'] = [[0, 0]]
                
                for key in ['shell_span', 'span', 'ac_position_info', 'adu_label', 'rel_label']:
                    if(key in temp):
                        try:
                            temp[key] = torch.tensor(temp[key], dtype=torch.long)
                        except:
                            print(key, temp[key])
                            None+1
                temp['author'] = torch.tensor(temp['author'], dtype=torch.long)
                temp['adu_length'] = len(temp['span'])
                if('mask' in temp):
                    temp['mask'] = [torch.tensor(_, dtype=torch.long) for _ in temp['mask']]

                self.data.append( temp)
        self.elmo_path = embed_path


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        train:
            'adu_label', 'ref_label', 'mask', 'rel_label'
        both:
            'topic_index', 'elmo_index', 'adu_length', 'shell_span', 'span', 'adu_label', 'ac_position_info', 'uid'
        """
        temp = copy.deepcopy(self.data[idx])
        temp['elmo_path'] = self.elmo_path
        return temp

def tree_collate_fn(src):
    """
        'elmo_path', 'topic_index', 'elmo_index', 'adu_length',
        'shell_span', 'span', 'ac_position_info', 'uid'
    """
    def padding(data, dtype, val=0.0):
        # first find max in every dimension
        size = len(data[0].shape)
        try:
            temp_len = torch.tensor( [ _.shape for _ in data] )
        except:
            print(data)
            None+1
        max_len = [len(data)] + temp_len.max(dim=0)[0].tolist()

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
    data = dict()
    for key in src[0]:
        data[key] = [ _[key] for _ in src]

    output = dict()
    # preprocess elmo and topic embedding
    with h5py.File(data['elmo_path'][0], 'r') as elmo_file:
        topic_emb = [
            torch.tensor(elmo_file.get(str(_))[()], dtype=torch.float)
            for _ in data['topic_index']
        ]
        output['topic_length'] = torch.tensor( [ _.shape[1] for _ in topic_emb])
        output['topic_emb'] = padding(topic_emb, dtype=torch.float)

        elmo_emb = []
        elmo_length = []
        para_length = []

        acc = 0
        data['para_span'] = []
        for i, post in enumerate(data['elmo_index']):
            data['para_span'].append([])
            for j, _ in enumerate(post):
                elmo_emb.append(
                    torch.tensor(elmo_file.get(str(_))[()], dtype=torch.float)
                )
                elmo_length.append(
                    elmo_emb[-1].shape[1]
                )
                data['para_span'][-1].append([j+acc, 1, elmo_emb[-1].shape[1]-2])
            data['para_span'][-1] = torch.tensor(data['para_span'][-1], dtype=torch.long)
            data['shell_span'][i][:, 0] += acc
            data['span'][i][:, 0] += acc
            para_length.append(len(post))

            acc += len(post)

        output['elmo_emb'] = padding(elmo_emb, dtype=torch.float)
        output['elmo_length'] = torch.tensor(elmo_length, dtype=torch.long)
        output['para_length'] = torch.tensor(para_length, dtype=torch.long)

    output['adu_length'] = torch.tensor(data['adu_length'], dtype=torch.long)

    for key in ['shell_span', 'span', 'ac_position_info', 'para_span', 'author']:
        output[key] = padding(data[key], dtype=torch.long)

    # uid
    output['uid'] = data['uid']
    label = {}
    if(('rel_label' in data) and ('ref_label' in data)):
        label['type'] = padding(data['adu_label'], dtype=torch.long, val=-1)
        label['link_type'] = padding(data['rel_label'], dtype=torch.long, val=-1)

        max_adu_length = output['adu_length'].max().item()
        
        rel_label = []
        for _ in data['ref_label']:
            rel_label.append(                  
                torch.tensor( [ max_adu_length if(l=='title') else l for l in _ ], dtype=torch.long)
            )
        label['link'] = padding(rel_label, dtype=torch.long, val=-1)

        mask = [padding(_, dtype=torch.long) for _ in data['mask']]
        output['mask'] = padding(mask, dtype=torch.long)

        batch_size, adu_dim = label['link_type'].shape
        assert(adu_dim==max_adu_length), 'dimension should be same'
        for i in range(batch_size):
            for j in range(adu_dim):
                index = label['link'][i,j].item()
                if(index == -1 or index == max_adu_length):
                    continue
                if(output['mask'][i, j, index].item() == 0 or j==index):
                    label['link_type'][i, j] = -1
                    label['link'][i, j] = -1

    return output, label


if(__name__ == '__main__'):
    data_path = './../../preprocess/span/CMV_train'
    embed_path = './../../preprocess/span/CMVELMo.hdf5'

    dataloader = treeDataset(data_path=data_path, embed_path=embed_path)
    print('----------')
    for key in dataloader[0]:
        try:
            print(key, dataloader[0][key].shape)
        except:
            print(key, dataloader[0][key])

    print('----------')
    batch_size = 64
    train_dataloader = DataLoader(dataloader, batch_size=batch_size,shuffle=False, num_workers=4,collate_fn=tree_collate_fn)
    for i, (data, label) in enumerate(train_dataloader):
        #if(i%1000==0):
        print(i)

        if(i==0):
            torch.save(data, './test_data')
            for key in data:
                try:
                    print(key, data[key].shape)
                except:
                    print(data[key])

                # print(data[key])
        pass
