"""
python preprocess_tree.py "predict file" "mapping file" "base folder"

"""

import torch
import json
import collections
from pytorch_pretrained_bert.tokenization import BertTokenizer
import string
import copy
import sys
import os


tokenizer = BertTokenizer.from_pretrained('/nfs/nas-7.1/kyhuang/lab/arg_ranking/arg_parsing_bert/saved_models/gaku_essay_55/', do_lower_case=True)




    
def checkandvote(sent, recovers, bios, dtypes):
    mapping_bio = ['B', 'I', 'O', 'E']
    mapping_type = ['P', 'C']
    
    
    recover_sent, recover_bio = [], []
    adu_type = []
    count = [0, 0]
    
    last = torch.rand(2)
    for index, (word, r, bio, dtype) in enumerate(zip(sent[1:], recovers, bios, dtypes)):
        if(word[:2]=='##'):
            if(bio==0):
                last = torch.zeros_like(dtype)
                
                recover_bio[-1] = 'B'
                count[0] += 1
                
            if( bio == 3 ):
                if( recover_bio[-1] == 'B' ):
                    recover_bio[-1] = 'O'
                    count[0] -= 1
                else:
                    adu_type.append(
                        mapping_type[last.argmax(-1).item()]
                    )
                    recover_bio[-1] = mapping_bio[bio]
                    count[1] += 1
                
            recover_sent[-1] += word.split('#')[-1]
        else:
            if( bio == 0 ):
                last = torch.zeros_like(dtype)
                count[0] += 1
                
            elif( bio == 3 ):
                adu_type.append(
                    mapping_type[last.argmax(-1).item()]
                )
                count[1] += 1

            recover_sent.append(word)
            recover_bio.append(mapping_bio[bio])

        if(bio<=1):
            last += dtype
    
    if(count[1] != count[0]):
        adu_type.append(
            mapping_type[last.argmax(-1).item()]
        )
        count[1] +=1
        recover_bio[-1] = 'O'
        if(recover_bio[-2] == 'B'):
            recover_bio[-2] = 'O'
        else:
            recover_bio[-2] = 'E'
        
    assert count[0]==(len(adu_type)),"{}\n{}\n{}".format(bios, recover_bio, adu_type)
    assert count[1]==(len(adu_type)),"{}\n{}\n{}".format(bios, recover_bio, adu_type)
    return recover_sent, recover_bio, adu_type
        
# postprocess     
def remove_fe_punctation(sent, bios, dtypes):
    #print(sent)
    #print(bios)
    punc = '!#$%&\*+,-./:;?@^_`|~'
    
    new_type = []
    type_index = 0
    
    for index in range(len(sent)-1):
        if(bios[index] == 'B'):
            if(sent[index] in punc):
                bios[index] = 'O'
                if(bios[index+1] == 'E'):
                    bios[index+1] = 'O'
                    type_index += 1
                else:
                    bios[index+1] = 'B'
        elif(bios[index] == 'E'):
            new_type.append( dtypes[type_index])
            type_index += 1
    else:
        if(bios[len(sent)-1] == 'E'):
            new_type.append( dtypes[type_index])
    # return sent, bios, new_type
    
    new_new_type = []
    type_index = len(new_type)-1
    for index in range(len(sent)-1, 0, -1):
        if(bios[index] == 'E'):
            if(sent[index] in punc):
                bios[index] = 'O'
                if(bios[index-1] == 'B'):
                    bios[index-1] = 'O'
                    type_index -= 1
                else:
                    bios[index-1] = 'E'
        elif(bios[index] == 'B'):
            new_new_type.append( new_type[type_index])
            type_index -= 1
    else:
        if(bios[0] == 'B'):
            new_new_type.append( new_type[type_index])
                
    return ' '.join(sent), bios, new_new_type

def remove_last_conj(sent, bios, dtypes):
    #print(sent)
    #print(bios)
    #print(dtypes)
    conj = ['and', 'or']
    
    new_type = []
    type_index = len(dtypes)-1
    
    for index in range(len(sent)-1, 0, -1):
        if(bios[index] == 'E'):
            if(sent[index] in conj):
                bios[index] = 'O'
                if(bios[index-1] == 'B'):
                    bios[index-1] = 'O'
                    type_index -= 1
                else:
                    bios[index-1] = 'E'
        elif(bios[index] == 'B'):
            new_type.append( dtypes[type_index])
            type_index -= 1
    else:
        if(bios[0] == 'B'):
            #print(type_index)
            new_type.append( dtypes[type_index])
    
    return ' '.join(sent), bios, new_type[::-1]

def removebio(sent, bios, dtype, remove_set):
    new_sent, new_bio, new_type = [], [], []
    buffer = []
    type_index = 0
            
    for index, (word, bio) in enumerate( zip(sent, bios)):
        new_sent.append( word )

        if( bio=='O'):
            new_bio.append( bio )
        else:
            buffer.append(word)

            if( bio == 'E'):
                l = len(buffer)
                new_type.append( dtype[type_index] )
                type_index += 1
                
                for remove_len in sorted(sort_front_conj, reverse=True):
                    if(len(buffer)<remove_len):
                        continue
                        
                    remove_sents = sort_front_conj[remove_len]
                    check_sent = ' '.join(buffer[:remove_len])
                

                    if(check_sent in remove_sents):
                        # convert these data to others
                        new_bio.extend( ['O']*(remove_len) )
                        if((l-remove_len)>1):
                            l = l-remove_len
                            new_bio.append('B')
                            if(l>=2):
                                new_bio.extend( ['I']*(l-2) )
                            new_bio.append('E')
                        elif( (l-remove_len)==1 ):
                            new_bio.append('O')
                            new_type.pop()
                        elif(l==remove_len):
                            new_type.pop()
                        else:
                            raise ValueError(' should be shorter')
                        break
                else:    
                    new_bio.append('B')
                    if(l>=2):
                        new_bio.extend( ['I']*(l-2) )
                    new_bio.append('E')

                buffer = []
    assert(len(new_sent) == len(new_bio)),"{}_{}\n{}\n{}\n".format(len(new_sent), len(new_bio), new_sent, new_bio)
    return ' '.join(new_sent), new_bio, new_type


def concat(total_type):
    dtype = []
    for _ in total_type:
        dtype.extend(_)
    return dtype


def find_shell(index, last, context, adu_index):
    if(index<=0):
        return (adu_index, 0, 0)
    for i in range(index, last, -1):
        if(context[i] in [".", "!", "?", "</ac>", "<para>"]):
            if(i>=index):
                return (adu_index, index, index)
            else:
                return (adu_index, i+1, index)
    
    return (adu_index, index, index)

def prepare(data, elmo_preprocess):
    mask = []
    shell_span, span, elmo_index = [], [], []
    adu_label = []
    author = []
    para_author = []

    topic_elmo_index = len(elmo_preprocess)
    elmo_preprocess.append('<topic> '+data['topic']+' </topic>')
    elmo_preprocess[-1] = ' '.join(elmo_preprocess[-1].split())
    
    ac_position_info = []
    adu_index = 0
    for post_pos, post in enumerate(data['content']):
        for bio, context in zip(post['bio'], post['context']):
            elmo_index.append(len(elmo_preprocess))

            elmo_preprocess.append(['<para>'])
            bios, sent = bio.split(), context.split()

            last = -1
            for bio, word in zip(bios, sent):
                if(bio == 'B'):
                    span.append([adu_index, len(elmo_preprocess[-1]), 0])
                    elmo_preprocess[-1].append('<ac>')
                    
                    shell_span.append(find_shell(len(elmo_preprocess[-1])-1, last, elmo_preprocess[-1], adu_index))
                elmo_preprocess[-1].append(word)
                
                if(bio == 'E'):
                    span[-1][-1] = len(elmo_preprocess[-1])
                    last = span[-1][-1]
                    
                    elmo_preprocess[-1].append('</ac>')
                    
            elmo_preprocess[-1].append('</para>')
            elmo_preprocess[-1] = ' '.join(elmo_preprocess[-1])
            # update parameter
            adu_index += 1
                
        # build mask
        dtype = post['type']
        prev_len, now_len = len(adu_label), len(dtype)
        for adu_pos, _ in enumerate(dtype):
            if(_ == 'P'):
                # constrain search space to this reply
                mask.append([0]*prev_len + [1]*now_len)

            elif(_ == 'C'):
                # constrain search space to all
                mask.append([1]*(prev_len + now_len))

            ac_position_info.append([adu_pos, post_pos])

        adu_label.extend( dtype )

    if(('[deleted]' in data['pos_author'][0]) and data['content'][-1]['author'] == '[deleted]'):
        for index, post in enumerate(data['content']):
            author.extend([1+index&1]*len(post['type']))
            para_author.extend([1+index&1]*len(post['bio']))
    else:
        for post in data['content']:
            if(post['author'] in data['pos_author'][0]):
                author.extend([1]*len(post['type']))
                para_author.extend([1]*len(post['bio']))
            elif(post['author'] in data['pos_author'][1]):
                author.extend([2]*len(post['type']))
                para_author.extend([2]*len(post['bio']))
            else:
                author.extend([0]*len(post['type']))
                para_author.extend([0]*len(post['bio']))

    adu_map = {'P':0, 'C':1}
    adu_label = [ adu_map[_] for _ in adu_label]
        
    return {
        'topic_index':topic_elmo_index, 
        'elmo_index':elmo_index,
        'shell_span':shell_span, 
        'span':span, 
        'mask':mask, 
        'author':author,
        'para_author':para_author,
        'adu_label':adu_label,
        'ac_position_info':ac_position_info
    }

 
def find_shell_bert(index, last, context, adu_index):
    if(index<=0):
        return (adu_index, 0, 0)
    for i in range(index, last, -1):
        if(context[i] in [".", "!", "?"]):
            if(i>=index):
                return (adu_index, index, index)
            else:
                return (adu_index, i+1, index)
    
    return (adu_index, index, index)

def prepare_bert(data):
    mask = []
    shell_span, span, text = [], [], []
    adu_label = []
    author = []
    para_author = []
    topic = data['topic']
    
    ac_position_info = []
    adu_index = 0
    for post_pos, post in enumerate(data['content']):
        for bio, context in zip(post['bio'], post['context']):
            text.append( context.split() )

            bios, sent = bio.split(), context.split()

            last = -1
            for index, (bio, word) in enumerate(zip(bios, sent)):
                if(bio == 'B'):
                    span.append([adu_index, index, 0])
                    shell_span.append(find_shell_bert(index-1, last, text[-1], adu_index))
                
                if(bio == 'E'):
                    span[-1][-1] = index
                    last = span[-1][-1]
            
            text[-1] = ' '.join(text[-1])
            adu_index += 1
        
        # build mask
        dtype = post['type']
        prev_len, now_len = len(adu_label), len(dtype)
        for adu_pos, _ in enumerate(dtype):
            if(_ == 'P'):
                # constrain search space to this reply
                mask.append([0]*prev_len + [1]*now_len)

            elif(_ == 'C'):
                # constrain search space to all
                mask.append([1]*(prev_len + now_len))

            ac_position_info.append([adu_pos, post_pos])

        adu_label.extend( dtype )

    if(('[deleted]' in data['pos_author'][0]) and data['content'][-1]['author'] == '[deleted]'):
        for index, post in enumerate(data['content']):
            author.extend([1+index&1]*len(post['type']))
            para_author.extend([1+index&1]*len(post['bio']))
    else:
        for post in data['content']:
            if(post['author'] in data['pos_author'][0]):
                author.extend([1]*len(post['type']))
                para_author.extend([1]*len(post['bio']))
            elif(post['author'] in data['pos_author'][1]):
                author.extend([2]*len(post['type']))
                para_author.extend([2]*len(post['bio']))
            else:
                author.extend([0]*len(post['type']))
                para_author.extend([0]*len(post['bio']))
    adu_map = {'P':0, 'C':1}
    adu_label = [ adu_map[_] for _ in adu_label] 
        
    return {
        'topic':topic, 
        'text':text,
        'shell_span':shell_span, 
        'span':span, 
        'mask':mask, 
        'author':author,
        'para_author':para_author,
        'adu_label':adu_label,
        'ac_position_info':ac_position_info
    }


if(__name__ == '__main__'):
    pred = {}
    index_mapping = {}
    for side, key in enumerate(['pos', 'neg', 'op']):
        pred[key] = torch.load(sys.argv[1]+'_{}'.format(key))
        with open(sys.argv[2]+'_{}'.format(key)) as f:
            index_mapping[key] = [line for line in f]

    # recover data from prediction
    recover = {}
    for key in ['pos', 'neg', 'op']:
        recover[key] = {'sent':[], 'bio':[], 'index':[],'type':[]}
        for index in range(len(pred[key]['id'])):
            temp = checkandvote(
                            tokenizer.convert_ids_to_tokens(pred[key]['id'][index].tolist()), 
                            pred[key]['recover'][index].tolist(), pred[key]['bio'][index],
                            torch.tensor(pred[key]['type'][index]).softmax(-1)
                        )
            
            recover[key]['sent'].append(' '.join(temp[0]))
            recover[key]['bio'].append(temp[1])
            recover[key]['type'].append(temp[2])
            recover[key]['index'].append(pred[key]['index'][index].item())

    # remove prefix
    front_conj = json.load(open('qq.json'))
    sort_front_conj = collections.defaultdict(list)
    for _ in front_conj:
        if(_[0] in string.punctuation):
            continue
        l = len(_.strip().split())
        sort_front_conj[l].append(_)

    for side in recover:
        for index in range(len(recover[side]['sent'])):
            temp = remove_fe_punctation(recover[side]['sent'][index].split().copy(), recover[side]['bio'][index].copy(), recover[side]['type'][index].copy())
            temp = remove_last_conj(temp[0].split().copy(), temp[1].copy(), temp[2].copy())
            temp = remove_fe_punctation(temp[0].split().copy(), temp[1].copy(), temp[2].copy())
            temp = removebio(temp[0].split().copy(), temp[1].copy(), temp[2].copy(), sort_front_conj)
            temp = remove_fe_punctation(temp[0].split().copy(), temp[1].copy(), temp[2].copy())
            
            recover[side]['sent'][index] = temp[0] 
            recover[side]['bio'][index]  = temp[1]
            recover[side]['type'][index] = temp[2]

            c = [0, 0]
            for word, bio in zip(recover[side]['sent'][index].split(), recover[side]['bio'][index]):
                if(bio == 'B'): c[0] += 1
                if(bio == 'E'): c[1] += 1
            assert(c[0] == c[1])
            assert(c[0] == len(recover[side]['type'][index]))

    for side in recover:
        for index in range(len(recover[side]['sent'])):
            if( ('< link >' in recover[side]['sent'][index]) or 
                ('< cite >' in recover[side]['sent'][index]) or 
                ('< user >' in recover[side]['sent'][index]) or
                ('< edit >' in recover[side]['sent'][index]) or 
                ('< reply >' in recover[side]['sent'][index])):
                sent = recover[side]['sent'][index].split()
                bios = recover[side]['bio'][index]
                dtype = recover[side]['type'][index]

                temp_sent, temp_bio, temp_type = [], [], []
                word_index, type_index = 0, 0
                while(word_index<len(sent)):
                    temp_sent.append(sent[word_index])
                    temp_bio.append(bios[word_index])
                    
                    if(bios[word_index]=='B'):
                        temp_type.append(dtype[type_index])
                        type_index += 1

                    for key in ['link', 'cite', 'reply', 'user', 'edit']:
                        if(sent[word_index:word_index+3]==['<', key, '>']):
                            temp_sent.pop()
                            temp_bio.pop()
                            temp_sent.append('<{}>'.format(key))

                            if(bios[word_index:word_index+3] == ['I', 'I', 'I']):
                                temp_bio.append('I')
                            elif(bios[word_index:word_index+3] == ['I', 'I', 'E']):
                                temp_bio.append('E')
                            elif(bios[word_index:word_index+3] == ['I', 'E', 'O']):
                                temp_bio.append('E')

                            elif(bios[word_index:word_index+3] == ['E', 'O', 'O']):
                                temp_bio.append('E')

                            elif(bios[word_index:word_index+3] == ['B', 'I', 'I']):
                                temp_bio.append('B')
                            elif(bios[word_index:word_index+3] == ['B', 'I', 'E']):
                                temp_bio.append('O')
                                temp_type.pop()

                            elif(bios[word_index:word_index+3] == ['O', 'O', 'O']):
                                temp_bio.append('O')
                            elif(bios[word_index:word_index+3] == ['O', 'B', 'I']):
                                temp_bio.append('B')
                                temp_type.append(dtype[type_index])
                                type_index += 1

                            word_index += 2
                            break
                    word_index += 1
                    
                recover[side]['sent'][index] = ' '.join(temp_sent)
                recover[side]['bio'][index] = temp_bio
                recover[side]['type'][index] = temp_type


    pairs = {}
    for side, key in enumerate(['pos', 'neg']):
        with open(sys.argv[3]+'/parsing/clean_{}'.format(key)) as f:
            f.readline()

            for line in f:
                clean_post = json.loads(line)
                index, post_index, reply_index = [int(_) for _ in clean_post['uid'].split('_')]
                
                if(index not in pairs):
                    pairs[index] = {'content':[[], []]}
                
                while(len(pairs[index]['content'][side])<=post_index):
                    pairs[index]['content'][side].append([])
                    
                while(len(pairs[index]['content'][side][post_index])<=reply_index):
                    pairs[index]['content'][side][post_index].append(None)
                
                pairs[index]['content'][side][post_index][reply_index] = {
                    'bio':[None for _ in clean_post['context']],
                    'context':clean_post['context'],
                    'author':clean_post['author'],
                    'type':[None for _ in clean_post['context']]
                }

    with open(sys.argv[3]+'/parsing/clean_op') as f:
        f.readline()
        for line in f:
            clean_post = json.loads(line)    

            index = int(clean_post['uid'])
            pairs[index]['op_info'] = {
                    'bio':[None for _ in clean_post['context']],
                    'context':clean_post['context'],
                    'author':clean_post['author'],
                    'type':[None for _ in clean_post['context']],
                    'topic':clean_post['topic'],
                    'pos_author':clean_post['pos_author'],
                }

    print(len(pairs))
    for side, key in enumerate(['pos', 'neg']):
        for index, sent, bio, dtype in zip(recover[key]['index'], recover[key]['sent'], recover[key]['bio'], recover[key]['type']):
            index, post_index, reply_index, para_index = [ int(_) for _ in index_mapping[key][index].split('_')]

            # bert tokenizer is different from nltk, thus need to use the new one
            pairs[index]['content'][side][post_index][reply_index]['context'][para_index] = sent
            pairs[index]['content'][side][post_index][reply_index]['bio'][para_index] = ' '.join(bio)
            pairs[index]['content'][side][post_index][reply_index]['type'][para_index] = dtype

    key = 'op'
    for index, sent, bio, dtype in zip(recover[key]['index'], recover[key]['sent'], recover[key]['bio'], recover[key]['type']):
        index, para_index = [ int(_) for _ in index_mapping[key][index].split('_')]

        # bert tokenizer is different from nltk, thus need to use the new one
        pairs[index]['op_info']['context'][para_index] = sent
        pairs[index]['op_info']['bio'][para_index] = ' '.join(bio)
        pairs[index]['op_info']['type'][para_index] = dtype

    count = 0
    pair_index = list(pairs.keys())
    for index in pair_index:
        try:
            pairs[index]['op_info']['type'] = concat(pairs[index]['op_info']['type'])
        except:
            count += 1
            del pairs[index]
            continue
        
        for side in range(2):
            temp_post = []
            pos_authors = []
            for post, pos_author in zip(pairs[index]['content'][side], pairs[index]['op_info']['pos_author'][side]):
                temp_reply = []
                
                for reply in post:
                    try:
                        temp_reply.append({
                            'bio':reply['bio'],
                            'context':reply['context'],
                            'author':reply['author'],
                            'type':concat(reply['type'])
                        })
                    except TypeError:
                        count += 1
                        break
                else:
                    pos_authors.append(pos_author)
                    temp_post.append(temp_reply)
            
            pairs[index]['op_info']['pos_author'][side] = pos_authors
            pairs[index]['content'][side] = temp_post
            if(len(pairs[index]['content'][side]) == 0):
                del pairs[index]
                break
    count = 0
    for index in pairs:
        count += (len(pairs[index]['content'][0])*len(pairs[index]['content'][1]))
    print('original pair:', count)

    posts = [[], []]
    for index in pairs:
        for side in range(2):
            op_info = {}
            for key in ['bio', 'context', 'author', 'type']:
                op_info[key] = pairs[index]['op_info'][key]
                
            for post_index, post in enumerate(pairs[index]['content'][side]):
                temp = [copy.deepcopy(op_info)]
                temp.extend(post)
                
                posts[side].append({
                    'content':temp,
                    'topic':pairs[index]['op_info']['topic'],
                    'uid':'{}_{}'.format(index, post_index),
                    'pos_author': pairs[index]['op_info']['pos_author'][side][post_index]
                })
                
    # save post
    if(not os.path.isdir(sys.argv[3]+'/tree/')):
        os.makedirs(sys.argv[3]+'/tree/')
    if(not os.path.isdir(sys.argv[3]+'/tree_bert/')):
        os.makedirs(sys.argv[3]+'/tree_bert/')

    for i, key in enumerate(['pos', 'neg']):
        with open(sys.argv[3]+'/tree/clean_post_{}.jsons'.format(key), 'w') as f:
            for _ in posts[i]:
                f.write(json.dumps(_))
                f.write('\n')

    posts = []
    for i, key in enumerate(['pos', 'neg']):
        with open(sys.argv[3]+'/tree/clean_post_{}.jsons'.format(key)) as f:
            posts.append([json.loads(line) for line in f])
    prepare_data = [[], []]
    elmo_preprocess = [[], []]

    for side in range(2):
        for _ in posts[side]:
            temp = prepare(_, elmo_preprocess[side])
            temp['uid'] = _['uid']
            
            prepare_data[side].append(temp)
            
            for _ in temp['elmo_index']:
                assert(_<len(elmo_preprocess[side])),'{} {}'.format(side, i)
            assert(temp['topic_index']<len(elmo_preprocess[side])),'{} {}'.format(side, i)

    for side, key in enumerate(['pos', 'neg']):
        with open(sys.argv[3]+'/tree/cmv_elmo_{}.txt'.format(key), 'w') as f:
            for line in elmo_preprocess[side]:
                f.write(line)
                f.write('\n')

    for side, key in enumerate(['pos', 'neg']):
        with open(sys.argv[3]+'/tree/pre_{}.jsons'.format(key), 'w') as f:
            for _ in prepare_data[side]:
                f.write(json.dumps(_))
                f.write('\n')

    posts = []
    for i, key in enumerate(['pos', 'neg']):
        with open(sys.argv[3]+'/tree/clean_post_{}.jsons'.format(key)) as f:
            posts.append([json.loads(line) for line in f])
    
    prepare_data = [[], []]

    for side in range(2):
        for _ in posts[side]:
            temp = prepare_bert(_)
            temp['uid'] = _['uid']
            
            prepare_data[side].append(temp)

    for i, key in enumerate(['pos', 'neg']):
        with open(sys.argv[3]+'/tree_bert/pre_{}.jsons'.format(key), 'w') as f:
            for _ in prepare_data[i]:
                f.write(json.dumps(_))
                f.write('\n')
