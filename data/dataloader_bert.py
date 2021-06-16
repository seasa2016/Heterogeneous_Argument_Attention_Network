import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import collections
import copy
import h5py
import numpy as np
import json
import os

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        print(max_length, total_length)
        raise ValueError('the length should larger than all max {0} now {1}'.format(max_length, total_length))
        tokens_b.pop()

def check_span(edu_index, tokenizer, text, spans, shell_spans):
    if(len(spans)==0):
        return tokenizer.tokenize(text), [], []

    add = 0
    temp_text = []
    span = [
        [0, 2*len(spans), [[edu_index]], spans],
        [0, 2*len(shell_spans), [[edu_index]], shell_spans]
    ]

    for index, word in enumerate(text.split()):
        temp = tokenizer.tokenize(word)
        for i in range(2):
            if( span[i][0] < span[i][1] ):

                while(span[i][3][span[i][0]>>1][1+(span[i][0]&1)] == index):
                    span[i][2][-1].append(index+add)

                    if(span[i][0]&1):
                        span[i][2][-1][-1]+=(len(temp)-1)
                        span[i][2].append([edu_index])

                    span[i][0] += 1
                    if( span[i][0] >= span[i][1] ):
                        break

        temp_text.extend(temp)
        add+= (len(temp)-1)
    span[0][2].pop()
    span[1][2].pop()

    return temp_text, span[0][2], span[1][2]

def preprocess(examples, tokenizer, max_length=512,
    pad_on_left=False, pad_token=0, pad_token_segment_id=0,
    mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    def padding(input_ids, attention_mask, token_type_ids, max_length, pad_token, mask_padding_with_zero, pad_token_segment_id, pad_on_left):
        """
            prepare input feature
        """
        padding_length = max_length - len(input_ids)
        if(pad_on_left):
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
        
        return input_ids, attention_mask, token_type_ids

    
    ex_index = 0
    for index in examples:
        for side in range(2):
            for post_index, example in examples[index][side].items():
                if ex_index % 100 == 0:
                    print("Writing example %d" % (ex_index))
                #print(example['span'])
                #print(example['shell_span'])
                #print()
    
                # 'shell_span', 'span', 'mask', 'adu_label', 'ac_position_info', 're_index', 'sum_mask', 'link_rank', 'reply_rank', 'para_span', 'topic', 'text'
                # context part
                bert_data = {'input_ids':[], 'attention_mask':[], 'token_type_ids':[], 'span':[], 'shell_span':[], 'para_span':[], 'sent_length':[], 'topic_length':None, 'para_length':None}
                tokens_b = tokenizer.tokenize(example['topic'].lower())
                bert_data['topic_length'] = len(tokens_b)
                bert_data['para_length'] = len(example['text'])
                
                start = 0
                span, shell_span = example['span'], example['shell_span']
                for i, text in enumerate(example['text']):
                    for end in range(start, len(span)):
                        if(i != span[end][0]):
                            break
                    else:
                        end+=1
                    #print(span[start:end])
                    #print(shell_span[start:end])
                    tokens_a, temp_span, temp_shell_span = check_span(i, tokenizer, text, span[start:end], shell_span[start:end])
                    #print(temp_span)
                    #print(temp_shell_span)
                    #print()
                    start = end
                    bert_data['sent_length'].append(len(tokens_a))

                    try:
                        truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
                    except:
                        print('*'*20)
                        print(example)
                        print(len(tokens_a), tokens_a)
                        print(len(tokens_b), tokens_b)
                        print(max_length)
                        print()
                        print()

                        None+1
                    para_len = len(tokens_a)
                    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                    token_type_ids = [0] * len(tokens)

                    tokens += tokens_b + ["[SEP]"]
                    token_type_ids += [1] * (len(tokens_b) + 1)

                    input_ids = tokenizer.convert_tokens_to_ids(tokens)

                    # The mask has 1 for real tokens and 0 for padding tokens. Only real
                    # tokens are attended to.
                    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    input_ids, attention_mask, token_type_ids = padding(input_ids, attention_mask, token_type_ids, max_length,
                                                                    pad_token, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
                    
                    bert_data['input_ids'].append(input_ids)
                    bert_data['attention_mask'].append(attention_mask)
                    bert_data['token_type_ids'].append(token_type_ids)

                    bert_data['span'].extend(temp_span)
                    bert_data['shell_span'].extend(temp_shell_span)
                    bert_data['para_span'].append((i, 0, para_len-1))

                ################
                ## title part ##
                ################
                inputs = tokenizer.encode_plus(example['topic'], example['topic'], add_special_tokens=True, max_length=max_length)
                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                
                input_ids, attention_mask, token_type_ids = padding(input_ids, attention_mask, token_type_ids, max_length,
                                                                    pad_token, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
                
                bert_data['topic_input_ids'] = input_ids
                bert_data['topic_attention_mask'] = attention_mask
                bert_data['topic_token_type_ids'] = token_type_ids

                if(ex_index < 5):
                    print("*** Example ***")
                    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                        print("{}: {}".format(key, " ".join([str(x) for x in bert_data[key][0]])))

                    for key in ['topic_input_ids', 'topic_attention_mask', 'topic_token_type_ids', 'span', 'shell_span', 'para_span']:
                        print("{}: {}".format(key, " ".join([str(x) for x in bert_data[key]])))
                    print('*'*10)
                    """
                    print()
                    print(bert_data['span'])
                    print('*'*10)
                    None+1
                    """
                for key, val in bert_data.items():
                    example[key] = torch.tensor(val, dtype=torch.long)
                ex_index += 1

class itemDataset(Dataset):
    def __init__(self, data_path, pair_path, tokenizer):
        self.top = 3
        if(os.path.isfile(data_path+'.pt')):
            self.data = torch.load(data_path+'.pt')
        else:
            self.data = {}
            with open(data_path) as f:
                for line in f:
                    temp = json.loads(line)
                    index, side, post_index = [int(_) for _ in temp['uid'].split('_')]
                    
                    if(index not in self.data):
                        self.data[index] = [{}, {}]


                    #for key in ['link_score', 'link_type_score', 'type_score']:
                    #    temp[key] = torch.tensor(temp[key], dtype=torch.float)
                    for key in ['sum_mask', 'para_span', 'ac_position_info']:
                        temp[key] = torch.tensor(temp[key], dtype=torch.long)

                    link_graph = []
                    for map_type in range(3):
                        graph = collections.defaultdict(float)
                        for a, link_rank in enumerate(temp['link_rank'][map_type]):
                            for b, val in link_rank[:self.top]:
                                graph[(a, b)] = max(val, graph[(a, b)])
                                graph[(b, a)] = max(val, graph[(b, a)])
                            graph[(a, a)] = 1

                        link_graph.append(graph)
                    temp['graph'] = link_graph
                    temp['adu_length'] = torch.tensor(len(temp['span']), dtype=torch.long)

                    self.data[index][side][post_index] = temp
            
            preprocess(self.data, tokenizer)
            torch.save(self.data, data_path+'.pt')

        with open(pair_path) as f:
            self.pair = json.loads(f.readline())

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        """ 
        'span', 'shell_span', 'para_span'
        'mask', 'adu_label', 'ac_position_info', 'uid', 
        'sum_mask', 
        'link_rank', 'reply_rank',
        'input_ids', 'attention_mask', 'token_type_ids'
        'topic_input_ids', 'topic_attention_mask', 'topic_token_type_ids'
        """
        index, neg_post_index, pos_post_index = self.pair[idx]

        data_pair = [copy.deepcopy(self.data[index][0][neg_post_index]), 
                        copy.deepcopy(self.data[index][1][pos_post_index])]

        return data_pair

def collate_fn(src):
    """ 
        'span', 'shell_span', 'para_span'
        'mask', 'adu_label', 'ac_position_info', 'uid', 
        'sum_mask', 
        'link_rank', 'reply_rank',
        'input_ids', 'attention_mask', 'token_type_ids',
        'topic_input_ids', 'topic_attention_mask', 'topic_token_type_ids',
        'adu_length', 'para_length', 'topic_length', 'sent_length'
    """
    def padding(data, dtype):
        # first find max in every dimension
        size = len(data[0].shape)

        temp_len = np.array( [ _.shape for _ in data] )
        max_len = [len(data)] + temp_len.max(axis=0).tolist()

        temp = torch.zeros(max_len, dtype=dtype)
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

        for key in ['topic_input_ids', 'topic_attention_mask', 'topic_token_type_ids']:
            output[key] = padding(data[key], dtype=torch.long)

        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            output[key] = torch.cat(data[key], dim=0)

        acc = 0
        for i, l in enumerate(data['para_length']):
            data['shell_span'][i][:, 0] += acc
            data['span'][i][:, 0] += acc
            data['para_span'][i][:, 0] += acc
            acc += l
        
        for key in ['sent_length', 'adu_length', 'para_length', 'topic_length']:
            output[key] = torch.cat([ _.view(-1) for _ in data[key]], dim=0).view(-1)


        for key in ['shell_span', 'span', 'para_span', 'ac_position_info']:
            output[key] = padding(data[key], dtype=torch.long)
    
        """
        # sum_mask should be longer than span 1
        batchsize, max_mask = output['sum_mask'].shape
        max_adu_len = output['adu_length'].max().item()+1
        output['sum_mask'] = torch.cat([output['sum_mask'], torch.zeros(batchsize, max_adu_len-max_mask, dtype=torch.long)], dim=-1)
        """

        # uid
        output['uid'] = data['uid']

        # build up link graph
        # adu graph
        batch_size, max_adu_len = len(output['uid']), output['adu_length'].max().item()+1
        adj = []
        for map_type in range(3):
            temp_graph = torch.zeros(batch_size, max_adu_len, max_adu_len, dtype=torch.float)

            for i, link_graph in enumerate(data['graph']):
                for pos, val in link_graph[map_type].items():
                    temp_graph[i, pos[0], pos[1]] = val


            t_inv = 1/(temp_graph.sum(-1))
            t_inv[  torch.isinf(t_inv) ] = 0
            t_diag = torch.stack([ torch.diag(_) for _ in t_inv])

            adj.append(torch.matmul(t_diag, temp_graph))
        output['adu_graph'] = torch.stack(adj, dim=0)
        
        # para graph
        max_reply_len = max(data['para_length'])+max_adu_len
        reply_graph = torch.zeros(batch_size, max_reply_len, max_reply_len, dtype=torch.float)
        for i, (span, l) in enumerate(zip(span_copy, data['adu_length'])):
            for j, (index, _, _) in enumerate(span[:l]):
                reply_graph[i, j, index+max_adu_len] = 1
                reply_graph[i, index+max_adu_len, j] = 1

        for i, reply_ranks in enumerate(data['reply_rank']):
            #print(len(data['elmo_index'][i]))
            for j, reply_rank in enumerate(reply_ranks):
                #print(reply_rank)
                for k in reply_rank:
                    reply_graph[i, j+max_adu_len, k+max_adu_len] = 1
                    reply_graph[i, k+max_adu_len, j+max_adu_len] = 1

        t_inv = 1/(reply_graph.sum(-1))
        t_inv[  torch.isinf(t_inv) ] = 0
        t_diag = torch.stack([ torch.diag(_) for _ in t_inv])
        reply_graph = torch.matmul(t_diag, reply_graph)
        output['reply_graph'] = reply_graph

        # 'sum_mask'
        max_elmo_len = max(data['para_length'])
        sum_mask = torch.zeros(batch_size, max_elmo_len, dtype=torch.float)
        for i, (start, end) in enumerate(data['sum_mask']):
            sum_mask[i, start:end] = 1
        output['sum_mask'] = sum_mask

        #output['label'] = torch.tensor(data['file_index'], dtype=torch.long)

        outputs.append(output)

    return outputs


if(__name__ == '__main__'):
    data_path = './../../preprocess/cmv_raw_v2/tree_bert/train/data_dev'
    pair_path = './../../preprocess/cmv_raw_v2/tree_bert/train/graph_pair_dev'


    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    dataloader = itemDataset(data_path=data_path, pair_path=pair_path, tokenizer=tokenizer)
    print('----------')
    for key in dataloader[0][0]:
        try:
            print(key, dataloader[0][0][key].shape)
        except:
            pass
            #print(dataloader[0][0][key])

    print('----------')
    batch_size = 64
    train_dataloader = DataLoader(dataloader, batch_size=batch_size,shuffle=False, num_workers=4,collate_fn=collate_fn)
    for i, data in enumerate(train_dataloader):
        #if(i%1000==0):
        print(i)
        if(i==0):
            #torch.save(data, './test_data')
            for key in data[0]:
                try:
                    print(key, data[0][key].shape)
                except:
                    pass
                    #print(data[0][key])

                # print(data[key])
        pass
