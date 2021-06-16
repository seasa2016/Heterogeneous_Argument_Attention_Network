"""
python preprocess_persuasive.py "base folder" "path to pred" "p or not"
"""

import json
import numpy as np
import torch
import h5py
import collections
import copy
import nltk
import random
import sys
import os
#############################
best_threshold = [0.7, 0.2]

#############################

datas = {}
for side, key in enumerate(['pos', 'neg']):
    with open(sys.argv[1]+"/tree/pre_{}.jsons".format(key)) as f:
        
        for line in f:
            temp = json.loads(line)
            index = int(temp['uid'].split('_')[0])
            post_index = int(temp['uid'].split('_')[-1])
            
            if(index not in datas):
                datas[index] = [{}, {}]
                
            datas[index][side][post_index] = temp

clean_datas = {}
for side, key in enumerate(['pos', 'neg']):
    with open(sys.argv[1]+"/tree/clean_post_{}.jsons".format(key)) as f:
        for line in f:
            temp = json.loads(line)
            index = int(temp['uid'].split('_')[0])
            post_index = int(temp['uid'].split('_')[-1])
            
            if(index not in clean_datas):
                clean_datas[index] = [{}, {}]
                
            clean_datas[index][side][post_index] = temp

cmv_texts = []
for i, key in enumerate(['pos', 'neg']):
    with open(sys.argv[1]+'/tree/cmv_elmo_{}.txt'.format(key)) as f:
        cmv_texts.append( [_ for _ in f] )

# need to postprocess data, if we haven't make it.
clean_pred_datas = {}
for side, key in enumerate(['pos', 'neg']):
    with open(sys.argv[2]+'/pred_{}'.format(key)) as f:
        pred_datas = [json.loads(line) for line in f]
        
    for i, pred_post in enumerate(pred_datas):
        # find the exact length
        index = int(pred_post['uid'].split('_')[0])
        post_index = int(pred_post['uid'].split('_')[-1])
        l = len(datas[index][side][post_index]['span'])

        # convert to tensor
        pred_temp = {}
        for key in ['link_score', 'link_type_score']:
            pred_temp[key] = torch.tensor(pred_post[key], dtype=torch.float)
            assert((pred_temp[key].shape[0]+1) == pred_temp[key].shape[1])

            adu_v = pred_temp[key][:l, :l]
            title_v = pred_temp[key][:l, -1].unsqueeze(1)
            pred_temp[key] = torch.cat([adu_v, title_v], dim=1)

        pred_temp['type_score'] = torch.tensor( pred_post['type_score'], dtype=torch.float)
        pred_temp['type_score'] = pred_temp['type_score'][:l]


        if(index not in clean_pred_datas):
            clean_pred_datas[index] = [{}, {}]

        clean_pred_datas[index][side][post_index] = {
            'uid':pred_post['uid'],
            'link_score':pred_temp['link_score'],
            'link_type_score':pred_temp['link_type_score'],
            'type_score':pred_temp['type_score']
        }
del pred_datas

# remove no enough pair, since we have limit on the length of adu
# thus there might be data not predable
clean_pred_datas_list = list(clean_pred_datas.keys())
for index in clean_pred_datas_list:
    if(len(clean_pred_datas[index][1])==0 or len(clean_pred_datas[index][0])==0):
        del clean_pred_datas[index]

# we need to convert data to list and save it as cache file

if(not os.path.isdir(sys.argv[1]+'/tree/clean/')):
    os.makedirs(sys.argv[1]+'/tree/clean/')
if(not os.path.isdir(sys.argv[1]+'/tree/train/')):
    os.makedirs(sys.argv[1]+'/tree/train/')
if(not os.path.isdir(sys.argv[1]+'/tree_bert/clean/')):
    os.makedirs(sys.argv[1]+'/tree_bert/clean/')
if(not os.path.isdir(sys.argv[1]+'/tree_bert/train/')):
    os.makedirs(sys.argv[1]+'/tree_bert/train/')

    
with open(sys.argv[1]+'/tree/clean/clean_pair', 'w') as f:
    for index in clean_pred_datas:
        for side in range(2):
            for post_index, val in clean_pred_datas[index][side].items():
                
                f.write(json.dumps({
                    'uid':'{}_{}_{}'.format(index, side, post_index),
                    'link_score':val['link_score'].tolist(),
                    'link_type_score':val['link_type_score'].tolist(),
                    'type_score':val['type_score'].tolist(),
                }))
                f.write('\n')

# If we had preprocessed, load from cache.
clean_pred_datas = {}
with open(sys.argv[1]+'/tree/clean/clean_pair') as f:
    
    for line in f:
        val = json.loads(line)
        index, side, post_index = [int(_)  for _ in val['uid'].split('_')]
        
        if(index not in clean_pred_datas):
            clean_pred_datas[index] = [{}, {}]
        clean_pred_datas[index][side][post_index] = val

c = [0, 0, 0, 0]
clean_mask = {}

# deal with negitive path
side = 0
for index in clean_pred_datas:
    clean_mask[index] = [{}, {}]
    
    for post_index in clean_pred_datas[index][side]:
        
        temp = clean_datas[index][side][post_index]

        ori_author = temp['content'][0]['author']
        for re_index in range(len(temp['content'])-1, 0 , -1):
            if(temp['content'][re_index]['author'] != ori_author):
                break
        else:
            if(len(temp['content'])>2):
                c[0] += 1
                continue
                print(side, index)

        if(re_index<=0):
            c[1] += 1
            continue

        sum_mask = []
        for temp_index in range(re_index):
            sum_mask.extend([0]* len(temp['content'][temp_index]['type']))
        sum_mask.extend([1]* len(temp['content'][re_index]['type']))

        clean_mask[index][side][post_index] = copy.deepcopy( datas[index][side][post_index] )
        clean_mask[index][side][post_index]['re_index'] = re_index
        for key in ['link_score', 'link_type_score', 'type_score']:
            if( not isinstance(clean_pred_datas[index][side][post_index][key], list)):
                clean_mask[index][side][post_index][key] = copy.deepcopy( clean_pred_datas[index][side][post_index][key].tolist())
            else:
                clean_mask[index][side][post_index][key] = copy.deepcopy( clean_pred_datas[index][side][post_index][key])
        clean_mask[index][side][post_index]['sum_mask'] = sum_mask
        assert(len(clean_mask[index][side][post_index]['sum_mask']) <= (len(clean_mask[index][side][post_index]['span'])))
print(side, c)

# deal with positive side, typically use the post before the origin author
c, side= 0, 1
clean_pred_datas_index = list(clean_pred_datas.keys())
for index in clean_pred_datas_index:
    if(len(clean_mask[index][0])==0):
        del clean_mask[index]
        continue
        
    for post_index in clean_pred_datas[index][side]:
        temp = clean_datas[index][side][post_index]
        
        re_index = len(temp['content'])-1
        
        sum_mask = []
        for temp_index in range(re_index):
            sum_mask.extend([0]* len(temp['content'][temp_index]['type']))
        sum_mask.extend([1]* len(temp['content'][re_index]['type']))
        
        if(len(temp['content'][re_index]['type'])==0):
            c += 1

        clean_mask[index][side][post_index] = copy.deepcopy( datas[index][side][post_index] )
        clean_mask[index][side][post_index]['re_index'] = re_index
        for key in ['link_score', 'link_type_score', 'type_score']:
            if( not isinstance(clean_pred_datas[index][side][post_index][key], list)):
                clean_mask[index][side][post_index][key] = copy.deepcopy( clean_pred_datas[index][side][post_index][key].tolist())
            else:
                clean_mask[index][side][post_index][key] = copy.deepcopy( clean_pred_datas[index][side][post_index][key])
        clean_mask[index][side][post_index]['sum_mask'] = sum_mask
        assert(len(clean_mask[index][side][post_index]['sum_mask']) <= (len(clean_mask[index][side][post_index]['span'])))
print(side, c)

# truncate the out of range reply
for index, pair in clean_mask.items():
    for side in range(2):
        for post_index, path in pair[side].items():
            
            adu_length, para_length = 0, 0
            for i, _ in enumerate(clean_datas[index][side][post_index]['content']):
                adu_length += len(_['type'])
                para_length += len(_['bio'])
                if(i==path['re_index']):
                    break
            if(len(path['shell_span'])!=adu_length):
                print("{}, {}, {}".format(index, side, post_index))
            
            # here we need to truncate out of range data
            
            # adu_length
            for key in ['shell_span', 'span', 'adu_label', 'ac_position_info', 'type_score']:
                path[key] = path[key][:adu_length]
                
            for key in ['mask'][:adu_length]:
                for i in range(len(path[key])):
                    path[key][i] = path[key][i][:adu_length]
            
            for key in ['link_score', 'link_type_score']:
                score = torch.tensor(path[key], dtype=torch.float)
                adu_v = score[:adu_length, :adu_length]
                topic_v = score[:adu_length, -1].unsqueeze(1)
                score = torch.cat([adu_v, topic_v], dim=1)
                
                path[key] = score.tolist()
                
            # para_length
            for key in ['elmo_index']:
                path[key] = path[key][:para_length]
                #None+1


# remove data with delete sent inside,
check = set()
for index, pair in clean_mask.items():
    for side in range(2):
        for post_index, path in pair[side].items():
            for i, _ in enumerate(clean_datas[index][side][post_index]['content']):
                if(_['context'] == ['deleted']):
                    check.add( (index, side, post_index))
print('delete', len(check))
for index, side, post_index in check:
    del clean_mask[index][side][post_index]
    
clean_mask_list = list(clean_mask.keys())
for index in clean_mask_list:
    if(len(clean_mask[index][1])==0 or len(clean_mask[index][0])==0):
        del clean_mask[index]


# we need to convert data to list and save it as cache file
with open(sys.argv[1]+'/tree/clean/trunc_pair', 'w') as f:
    for index in clean_mask:
        for side in range(2):
            for post_index, val in clean_mask[index][side].items():
                val['uid'] = '{}_{}_{}'.format(index, side, post_index)
                
                f.write(json.dumps(val))
                f.write('\n')


# here we need to decide what will graph look like
# there might be several connected type we could discuss.

# we need to convert data to list and save it as cache file
clean_mask = {}
with open(sys.argv[1]+'/tree/clean/trunc_pair') as f:
    for line in f:
        temp = json.loads(line)
        index, side, post_index = [int(_) for _ in temp['uid'].split('_')]
        
        if(index not in clean_mask):
            clean_mask[index] = [{}, {}]
        clean_mask[index][side][post_index] = temp
    
for index, pair in clean_mask.items():
    for side in range(2):
        for post_index, path in pair[side].items():
            
            size = len(path['span'])
            link_rank = [[], [], []]
            if(size):
                mask = torch.zeros(size, size+1)
                for i, m in enumerate(path['mask'][:size]):
                    mask[i, :len(m)] = torch.tensor(m)
                    mask[i, i] = 0
                mask[:,-1] = 1
                mask = mask.byte()

                # normal part
                link_scores = torch.tensor(path['link_score'], dtype=torch.float)
                link_scores = torch.where(mask, link_scores, torch.full_like(link_scores, -1e18))
                link_scores = link_scores.softmax(-1)

                link_type_scores = torch.tensor(path['link_type_score'], dtype=torch.float)
                link_type_scores = torch.where(mask, link_type_scores, torch.full_like(link_type_scores, -1e18))
                link_type_scores = (link_type_scores.sigmoid()>best_threshold[1]).long()

                # deal with adu part
                for i, (link_score, link_type_score) in enumerate(zip(link_scores, link_type_scores)):
                    for _ in link_rank:
                        _.append([])
                    # need to maintain  undirected graph
                    for j, (link, link_type) in enumerate(zip(link_score, link_type_score)):
                        link_rank[link_type][i].append((j, link.item()))
                        link_rank[2][i].append((j, link.item()))
                    else:
                        link_rank[link_type][i][-1] = (-1, link.item())
                        link_rank[2][i][-1] = (-1, link.item())
            
                for map_type in range(len(link_rank)):
                    for node_id in range(len(link_rank[map_type])):
                        link_rank[ map_type ][node_id] = sorted(link_rank[ map_type ][node_id], key= lambda x:x[1], reverse=True)            
            
            path['link_rank'] = link_rank

                    
            # additional part, here we additional build several reply node
            adu_length = size
            
            ll = [ len(_['bio']) for _ in clean_datas[index][side][post_index]['content'][:path['re_index']+1]]
            reply_link = [ [] for _ in range(sum(ll))]
            prev, now = 0, 0
            for i in range(len(ll)):
                now += ll[i]
                for j in range(prev, now):
                    for k in range(now, now+ll[i+1]):
                        reply_link[j].append(k)

                prev = now
                if(i==path['re_index']-1):
                    break
            #print(len(reply_link), reply_link)
            a, b, c = 0, 0, 0
            for i, l in enumerate(ll):
                a=b
                b=c
                c+=l
                for j in range(b, c):
                    for k in range(j+1, c):
                        reply_link[j].append(k)

                if(i==path['re_index']):
                    break
                    
            path['reply_rank'] = reply_link
            path['pre_mask'] = (a, b)
            path['last_mask'] = (b, c)
            
            path['para_span'] = [ (i, 1, len(cmv_texts[side][elmo_index].split())-2)  for i, elmo_index in enumerate(path['elmo_index']) ]
            #None+1

# we need to convert data to list and save it as cache file
with open(sys.argv[1]+'/tree/train/graph_pair_total', 'w') as f:
    for index in clean_mask:
        for side in range(2):
            for post_index, val in clean_mask[index][side].items():
                temp = {}
                temp['uid'] = '{}_{}_{}'.format(index, side, post_index)
                for key in ['topic_index', 'elmo_index', 'shell_span', 'span', 'mask', 'author', 'para_author', 'adu_label', 'ac_position_info', 're_index', 'pre_mask', 'last_mask', 'link_rank', 'reply_rank', 'para_span']:
                    temp[key] = val[key]
                f.write(json.dumps(temp))
                f.write('\n')

# start to make training data pair.
# we need to convert data to list and save it as cache file


clean_mask = {}
with open(sys.argv[1]+'/tree/train/graph_pair_total') as f:
    for line in f:
        temp = json.loads(line)
        index, side, post_index = [int(_) for _ in temp['uid'].split('_')]
        
        if(index not in clean_mask):
            clean_mask[index] = [{}, {}]
        clean_mask[index][side][post_index] = temp
        
def extractpair(ori_data, index_list):
    defualt_word = set(['<ac>', '</ac>', '<para>', '</para>', '<link>', '<cite>', '<reply>', '<user>', '<edit>'])
    data = {}
    for index in index_list:
        data[index] = ori_data[index]
    
    pairs = []
    for index, pair in data.items():
        c = [set(), set()]
        for neg_post_index, neg_path in pair[0].items():
            neg_set = set()
            for _ in clean_datas[index][0][neg_post_index]['content']:
                for sent in _['context']:
                    for word in sent.split():
                        neg_set.add(word)

            for pos_post_index, pos_path in pair[1].items():
                pos_set = set()
                for _ in clean_datas[index][1][pos_post_index]['content']:
                    for sent in _['context']:
                        for word in sent.split():
                            pos_set.add(word)
                jaccard = nltk.jaccard_distance(pos_set-defualt_word, neg_set-defualt_word)
                if(jaccard<=0.5):
                    pairs.append((index, neg_post_index, pos_post_index))
                    c[0].add(neg_post_index)
                    c[1].add(pos_post_index)
        
        for side in range(2):
            post_index_list = list(pair[side].keys())
            
            for post_index in post_index_list:
                if(post_index not in c[side]):
                    del pair[side][post_index]
        
    return data, pairs

def write_out(data, pair, dtype):
    # we need to convert data to list and save it as cache file
    with open(sys.argv[1]+'/tree/train/data{}'.format(dtype), 'w') as f:
        for index in data:
            for side in range(2):
                for post_index, val in data[index][side].items():
                    temp = {}
                    temp['uid'] = '{}_{}_{}'.format(index, side, post_index)
                    for key in ['topic_index', 'elmo_index', 'shell_span', 'span', 'mask', 'author', 'para_author', 'adu_label', 'ac_position_info', 're_index', 'pre_mask', 'last_mask', 'link_rank', 'reply_rank', 'para_span']:
                        temp[key] = val[key]
                    f.write(json.dumps(temp))
                    f.write('\n')
        
    with open(sys.argv[1]+'/tree/train/graph_pair{}'.format(dtype), 'w') as f:
        f.write(json.dumps(pair))
        f.write('\n')
    print(dtype, len(pair))

random.seed(1728)
clean_mask_list = list(clean_mask.keys())
random.shuffle(clean_mask_list)
l = len(clean_mask_list)
if(sys.argv[3]=='3'):
    data_list_train, data_list_dev, data_list_test = {}, {}, {}
    data_train, pair_train = extractpair(clean_mask, clean_mask_list[:l*7//10])
    write_out(data_train, pair_train, '_train')
    data_dev, pair_dev = extractpair(clean_mask, clean_mask_list[l*7//10:l*8//10])
    write_out(data_dev, pair_dev, '_dev')
    data_test, pair_test = extractpair(clean_mask, clean_mask_list[l*8//10:])
    write_out(data_test, pair_test, '_test')
elif(sys.argv[3]=='2'):
    data, pair = extractpair(clean_mask, clean_mask_list[:l*8//10])
    write_out(data, pair, '_train')
    data, pair = extractpair(clean_mask, clean_mask_list[l*8//10:])
    write_out(data, pair, '_dev')
elif(sys.argv[3]=='1'):
    data, pair = extractpair(clean_mask, clean_mask_list)
    write_out(data, pair, '')

# bert version
def write_out_bert(data, pair, dtype):
    # we need to convert data to list and save it as cache file
    with open(sys.argv[1]+'/tree_bert/train/data{}'.format(dtype), 'w') as f:
        for index in data:
            for side in range(2):
                for post_index, val in data[index][side].items():
                    temp = {}
                    temp['uid'] = '{}_{}_{}'.format(index, side, post_index)
                    for key in ['topic', 'text', 'shell_span', 'span', 'mask', 'author', 'para_author', 'adu_label', 'ac_position_info', 're_index', 'pre_mask', 'last_mask', 'link_rank', 'reply_rank', 'para_span']:
                        temp[key] = val[key]
                    f.write(json.dumps(temp))
                    f.write('\n')
                    
    with open(sys.argv[1]+'/tree_bert/train/graph_pair{}'.format(dtype), 'w') as f:
        f.write(json.dumps(pair))
        f.write('\n')
    print(dtype, len(pair))

clean_bert_data = {}
for side, key in enumerate(['pos', 'neg']):
    with open(sys.argv[1]+'/tree_bert/pre_{}.jsons'.format(key)) as f:
        for line in f:
            temp = json.loads(line)
            index, post_index = [int(_) for _ in temp['uid'].split('_')]
            
            if(index not in clean_bert_data):
                clean_bert_data[index] = [{}, {}]
            clean_bert_data[index][side][post_index] = temp

for index in clean_mask:
    for side in range(2):
        for post_index in clean_mask[index][side]:
            key = None
            for key in ['topic_index', 'elmo_index']:
                del clean_mask[index][side][post_index][key]
                
            for key in ['topic', 'text', 'shell_span', 'span', 'mask', 'author', 'para_author', 'adu_label', 'ac_position_info', 'uid']:
                clean_mask[index][side][post_index][key] = clean_bert_data[index][side][post_index][key]

random.seed(1728)
clean_mask_list = list(clean_mask.keys())
random.shuffle(clean_mask_list)
l = len(clean_mask_list)
if(sys.argv[3]=='3'):
    data_list_train, data_list_dev, data_list_test = {}, {}, {}
    data_train, pair_train = extractpair(clean_mask, clean_mask_list[:l*7//10])
    write_out_bert(data_train, pair_train, '_train')
    data_dev, pair_dev = extractpair(clean_mask, clean_mask_list[l*7//10:l*8//10])
    write_out_bert(data_dev, pair_dev, '_dev')
    data_test, pair_test = extractpair(clean_mask, clean_mask_list[l*8//10:])
    write_out_bert(data_test, pair_test, '_test')
elif(sys.argv[3]=='2'):

    data, pair = extractpair(clean_mask, clean_mask_list[:l*8//10])
    write_out_bert(data, pair, '_train')
    data, pair = extractpair(clean_mask, clean_mask_list[l*8//10:])
    write_out_bert(data, pair, '_dev')
elif(sys.argv[3]=='1'):
    data, pair = extractpair(clean_mask, clean_mask_list)
    write_out_bert(data, pair, '')




















