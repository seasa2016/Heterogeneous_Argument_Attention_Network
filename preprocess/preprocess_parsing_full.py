"""
python preprocess_parsing.py "input file" "output folder"

"""

import glob
import codecs
import json
import os
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
import copy
import re
import multiprocessing 
import math
import time
import copy
import collections
import sys
import nltk



def parsing(in_data):
    #we need to remove useless data here
    total = []  #use this to mantain all child
    temp = {}
    
    for key in ['title', 'num_comments', 'name', 'author', 'url']:
        temp[key] = in_data[key]
    temp['body'] = in_data['selftext']
    
    temp['child'] = []
    
    total.append(temp)
    count = 0
    
    for data in in_data['comments']:
        child_temp = {}
        parent_id = data['parent_id']
        
        for key in ['author', 'name', 'body']:
            if(key in data):
                child_temp[key] = data[key]
        child_temp['child'] = []
        
        if('author' in data and data['author']=='DeltaBot'):
            count += 1
            
        for past in total: 
            if(past['name'] == parent_id):
                past['child'].append(child_temp)
                break
        total.append(child_temp)
    
    temp['count'] = count
    
    return temp

# find for all pair, if there are multiple choice, 
# choose the middle one
def dfs(data, past):
    temp = {}
    for key in ['author', 'name', 'body']:
        if(key in data):
            temp[key] = data[key]
            
    if('body' in data):
        past.append(temp)
        if( len(data['child']) == 0 ):
            yield (0, copy.deepcopy(past))
        else:
            
            for _ in data['child']:
                if(('author' in _) and (_['author'] == 'DeltaBot') and ('Confirmed:' in _['body'])):
                    past.append({
                        'author':_['author'],
                        'name':_['name'],
                        'body':_['body']
                    })
                    
                    yield (1, copy.deepcopy(past))
                    past.pop()
                    break
            else:
                for _ in data['child']:
                    yield from dfs(_, past)
            
        past.pop()

def find_pair(data):
    count = [[], []]
    op_info = {}
    for key in ['title', 'num_comments', 'author', 'name', 'url', 'body']:
        op_info[key] = data[key]
    
    for c, child in enumerate(data['child']):
        temp = [[], []]
        for index, flow in dfs(child, []):
            temp[index].append(flow)

        if(len(temp[1])>0):
            count[1].extend(temp[1])
        else:
            count[0].extend(temp[0])
            
    # here we check for positive and negitive sample for the correctness
    op_author = data['author']
    for side in range(2):
        temp = []
        for flow in count[side]:
            for _ in flow:
                if('[deleted]' == _['body']):
                    break
            else:
                temp.append(flow)
        count[side] = temp
        
    temp = []
    for path in count[0]:
        for _ in path:
            if(_['author'] == op_author):
                temp.append(path)
                break
                
    count[0] = temp
    # truncate the post to the mini_len

    # negative
    c = []
    for _ in count[0]:
        author = _[0]['author']
        for index in range(len(_)-1, -1, -1):
            if(_[index]['author'] != op_author):
                break
        
        if(_[index]['author'] != op_author):
            _ = _[:index+1]
            c.append([author, _])
            
    # deal with author
    pos_author = [[], []]
    temp = []
    check_list = set()
    for key, val in c:
        au = (_['author'] for _ in val)
        
        if(au in check_list):
            continue
        check_list.add(au)
        temp.append(val)
        pos_author[0].append([(op_author,), (key,)])
    count[0] = temp
    
    # positive
    c = {}
    for _ in count[1]:
        last = (_[-2]['author'], (_[0]['author'], _[-3]['author']))
        
        _ = _[:-2]
        if(len(_)):
            author = _[0]['author']
            if( author not in c):
                c[author] = (_, last)
            elif( len(c[author][0]) < len(_) ):
                c[author] = (_, last)
    count[1] = []
    for key, val in c.items():
        pos_author[1].append([(op_author, val[1][0]), val[1][1]])
        count[1].append(val[0])
    
    return count, op_info, pos_author

mispell_dict = {'didn\'t':'did not', 'doesn\'t':'does not', 'isn\'t':'is not', 'shouldn\'t':'should not', 'wasn\'t': 'was not',
                'hasn\'t': 'has not','won\'t':'wont','theatre': 'theater', 'cancelled': 'canceled', 'organisation': 'organization',
                'labour': 'labor', 'favourite': 'favorite', 'travelling': 'traveling', 'washingtons': 'washington', 'marylands': 'maryland',
                'chinas': 'china', 'russias': 'russia', '‘the': 'the', 'irans': 'iran', 'dulles': 'dulle' ,'commuincation':'communication',
                'parantage':'parentage','gorvernment':'government', '&#8710;':'∆'}
punct_dict = {'--':'', '’':'\'', '‘':'\'', '“':'"', '”':'"', '``':'"', "''":'"',"]":'',"[":'','_____':'',') )':')','( (':'(', '//':''}

def rem_url(text):
    def check(text, sym_pair):
        if('http' in text):
            temp = ''
            while(True):
                lindex = text.find(sym_pair[0]+'http')
                if(lindex==-1):
                    temp+=text
                    break
                temp += text[:lindex]
                rindex = text.find(sym_pair[-1], lindex)
                if(rindex==-1):
                    temp+=text[lindex:]
                    break
                temp += '<link>'
                text = text[rindex+1:]
            return temp
        return text
    
    text = check(text, '()')
    text = check(text, '[]')
    
    if('http' in text):
        text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "<link>", text)
        
    return text

def rem_sym(text):
    index = len(text)-1
    while((index>=1) and (text[index] == text[index-1]) and (not (text[index].isalpha() or text[index]=='~' or text[index].isdecimal()))):
        index -=1
    text = text[:index+1]
    index = 0
    while((index<len(text)) and (not (text[index].isalpha() or text[index].isdecimal() or text[index]=='~'))):
        if(text[index]=='>' or text[index]=='∆'):
            break
        index +=1
    text = text[index:]
    return text

def rem_italic(text):
    sent, buffer = [], []
    flag = False
    try:
        for word in text.split(' '):
            if(word[0]=='_'):flag = True

            if(flag):buffer.append(word)
            else:sent.append(word)

            if(word[-1]=='_'):
                flag = False
                sent.append(' '.join(buffer)[1:-1])
                buffer.clear()
        else:
            sent.extend(buffer)
    except:
        print([text])
        None+1
        
    return ' '.join(sent)

def rem_delete(text):
    sent = []
    buffer = []
    flag = False
    for word in text.split(' '):
        if(word[:2]=='~~'):
            word = word[2:]
            flag = True
        if(word[-2:]=='~~'):
            buffer.clear()
            flag = False
            continue
            
        if(flag):buffer.append(word)
        else:sent.append(word)
    else:
        sent.extend(buffer)
        
    return ' '.join(sent)

def clean(text):
    #build up inverted file
    text = text.lower().strip()
    if(text==''):
        return ''
    
    text = rem_url(text)
        
    for punct_c, punct_r  in punct_dict.items():
        text = text.replace(punct_c, punct_r)
        
    for key,data in mispell_dict.items():
        text = text.replace(key,data)
    
    """
    text = rem_sym(text)
    if(text==''):
        return ''
    """
    
    for _ in ['***', '**', '*', '^']:
        text = text.replace(_, '') 
    
    text = ' '.join(text.split())
    sent = []
    for word in word_tokenize(text):
        if(len(word)>1):
            if(word[0]=='.' and word[1]!='.'):
                sent.append('.')
                word = word[1:]
        if('user' in word and word[0]=='/'):
            word = '<user>'
        elif('/u/' in word and word[0]=='/'):
            word = '<user>'
        elif('/r/' in word and word[0]=='/'):
            word = '<reply>'
        sent.append(word)
        
    sent = ' '.join(sent)
    if(len(sent)): sent = rem_delete(sent)
    sent = ' '.join(sent.split())
    if(len(sent)): sent = rem_italic(sent)
    sent = ' '.join(sent.split())
    
    for key in ['link', 'edit', 'cite']:
        sent = sent.replace('< {} >'.format(key), '<{}>'.format(key))
    
    for punct_c, punct_r  in punct_dict.items():
        sent = sent.replace(punct_c, punct_r)
    sent = ' '.join(sent.split()).strip()
    
    return sent

# for here, we need to partition post into paragraph
def job(q, base, total_text):
    def clean_post(text):
        data = []
        for para in text.split('\n\n'):
            
            num = 1
            temp_para = []
            for sent in para.split('\n'):
                sent = ' '.join(sent.split())

                if(len(sent) and sent[0]=='*'):
                    # check if dot at fromt
                    sent = '{}. {}.'.format(num, sent)
                    num += 1
                else:
                    num = 1
                    # check if copy from front
                    if(len(sent) and sent[:4].lower()=='edit'):
                        sent = '<edit>'
                    elif(sent.startswith("&gt;") or sent.startswith(">")):
                        sent = '<cite>'
                
                temp_para.append(sent)
            para = clean(' '.join(temp_para))
            if(len(para)):
                data.append(para)
            
        return data
        
    
    for index, text in enumerate(total_text):
        index += base
        
        arr = {
            'op_info':copy.deepcopy(text[1]),
            'text':[[], []],
            'pos_author':copy.deepcopy(text[2])
        }
        #print(arr['op_info']['body'])
        arr['op_info']['body'] = clean_post(arr['op_info']['body'])

        for side in range(2):
            for path_index, path in enumerate(text[0][side]):
                arr['text'][side].append([])
                for re_index, _ in enumerate(path):
                    arr['text'][side][-1].append( {
                        'body':clean_post( _['body'] ),
                        'author': _['author']
                    } )                   
        q.put(arr)
        
        
def write_out(f,  q):
    while(not q.empty()):
        arr = q.get()
        
        arr['op_info']['title'] = arr['op_info']['title'].lower()
        for _ in ['cmv:', 'cmv :', 'cmw:', 'cmv', '/r/changemyview']:
            arr['op_info']['title'] = arr['op_info']['title'].replace(_, '')
        arr['op_info']['title'] = ' '.join(arr['op_info']['title'].split())

        f.write(json.dumps(arr))
        f.write('\n')
    f.flush()

def checkandwait(f, process, delay_time=5, end=False):
    while((len(process) >= num_cpu) or end):
        print(pair_index, len(process))
        if(len(process)==0): break
            
        temp_process = []
        for _ in process:
            _.join(timeout = delay_time)

        count = 0
        for _ in process:
            if(_.exitcode == None):
                temp_process.append(_)
            elif(_.exitcode == 0):
                #process[i].close()
                pass
            else:
                print('error')
                
        process = temp_process
        write_out(f, q)
    else:
        write_out(f, q)
    return process

def clean_sent(sent):
    for a in [',', '.', '?']:
        for b in [',', '.', '?']:
            sent = sent.replace('{} {}'.format(a,b), ' {} '.format(b))
    sent = sent.replace("ca n't", "can't")
    sent = sent.replace(" n't", "n't")
    return sent.lower()
                    

if(__name__ == '__main__'):
    # check for all data, and search for delta data
    # make output folder 
    if(not os.path.isdir(sys.argv[2])):
        os.makedirs(sys.argv[2]+'/parsing/')

    datas = []
    with open(sys.argv[1]) as f:
        for i,line in enumerate(f):
            datas.append( parsing(json.loads(line)))
            
    pairs = []
    for index, _ in enumerate(datas):
        if(_['count']>=1):        
            count, op_info, pos_author = find_pair(_)
            
            if( (len(count[1])==0) or  (len(count[0])==0)):
                continue
            pairs.append((count, op_info, pos_author))

    # use multi-process for cleaning
    num_cpu = 15
    q = multiprocessing.Queue()
    f = open(sys.argv[2]+'/parsing/bert_pre', 'w')
    base = 256
    process = []
    for pair_index in range(math.ceil(len(pairs)/base)):
        process.append(multiprocessing.Process(
                        target = job, 
                        args = (q, pair_index*base, pairs[pair_index*base:(pair_index+1)*base])
                        ))
        process[-1].start()
        process = checkandwait(f, process)
            
    else:
        process = checkandwait(f, process, 5, True)    
        f.close()
    for _ in process:
        _.terminate()

    clean_pairs = []
    with open(sys.argv[2]+'/parsing/bert_pre') as f:
        for line in f:
            clean_pairs.append(json.loads(line))
    print(len(clean_pairs))
    print(clean_pairs[0].keys())

    for index, pair in enumerate(clean_pairs):
        for side in range(2):
            for post in pair['text'][side]:
                for reply in post:
                    if(reply['body']==[]):
                        continue
                    temp_para = [clean_sent(reply['body'][0])]
                    pre = reply['body'][0]
                    
                    for sent in reply['body'][1:]:
                        clean_s = clean_sent(sent)
                        if(pre == '<link>' or pre == '<cite>'):
                            temp_para[-1] += ' '
                            temp_para[-1] += clean_s
                        #elif(sent[0].isdigit()):
                        #    temp_para[-1] += ' '
                        #    temp_para[-1] += sent
                        else:
                            temp_para.append(clean_s)
                        pre = sent
                    reply['body'] = temp_para
    count = 0
    for index, pair in enumerate(clean_pairs):
        count += len(pair['text'][0])*len(pair['text'][1])
    print('origin', count)
    
    jaccard_pairs = []
    count = 0
    for pair in clean_pairs:
        c = [set(), set()]
        author_set = set()
        for side in range(2):
            for path in pair['text'][side]:
                for _ in path:
                    author_set.add(_['author'])
        if(len(author_set)<10):
            continue
        op_set = set()
        for sent in pair['op_info']['body']:
            for word in word_tokenize(sent):
                op_set.add(word)
            
        for neg_index, neg_path in enumerate(pair['text'][0]):
            neg_set = set()
            
            for _ in neg_path:
                for sent in _['body']:
                    for word in sent.split():
                        neg_set.add(word)
            else:
                for pos_index, pos_path in enumerate(pair['text'][1]):
                    pos_set = set()

                    for _ in pos_path:
                        for sent in _['body']:
                            for word in sent.split():
                                pos_set.add(word)
                    else:
                        jaccard = nltk.jaccard_distance(pos_set|op_set, neg_set|op_set)
                        if(jaccard<=0.5):
                            count += 1
                            c[0].add(neg_index)
                            c[1].add(pos_index)
        if( len(c[0])!=0 and len(c[1])!=0 ):
            jaccard_pairs.append(copy.deepcopy(pair))
            for side in range(2):
                jaccard_pairs[-1]['text'][side] = [pair['text'][side][index] for index in c[side]]
                jaccard_pairs[-1]['pos_author'][side] = [pair['pos_author'][side][index] for index in c[side]]
            
            
    print('after', count)

    #output
    f = [ open(sys.argv[2]+'/parsing/clean_{0}'.format(key), 'w') for key in ['pos', 'neg'] ]
    for _ in f:
        _.write('\n')
    for index, pair in enumerate(jaccard_pairs):    
        for side in range(2):
            for post_index, post in enumerate(pair['text'][side]):
                for reply_index, reply in enumerate(post):
                    f[side].write(json.dumps({
                        'topic':pair['op_info']['title'],
                        'context':reply['body'],
                        'author':reply['author'],
                        'uid':'{}_{}_{}'.format(index, post_index, reply_index)
                    }))
                    f[side].write('\n')
    for _ in f:
        _.close()

    with open(sys.argv[2]+'/parsing/clean_op', 'w') as f:
        f.write('\n')
        for index, pair in enumerate(jaccard_pairs):    
            f.write(json.dumps({
                'topic':pair['op_info']['title'],
                'context':pair['op_info']['body'],
                'author':pair['op_info']['author'],
                'url':pair['op_info']['url'],
                'pos_author':pair['pos_author'],
                'uid':'{}'.format(index)
            }))
            f.write('\n')
