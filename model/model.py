"""
Here we wnat to build up several module,
1. paragraph contextual encoder
2. extractor
3. adu encoder(?)
4. aggregater
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
from model.context_encoder import Elmo_encoder
from model.adu_encoder import (span_encoder, para_encoder)
from model.extractor import build_extractor
from model.graph import HeterogeneousGraph
from model.module import noiselayer, positionLayer, final_extractor
from model.decode import TransDecoder

class ArgTree(nn.Module):
    def __init__(self, args):
        super(ArgTree, self).__init__()
        self.noise = noiselayer(0.0, args.dropout)
        self.dropout2d = nn.Dropout2d(args.dropout_lstm)

        # encoder part
        self.context_encoder = Elmo_encoder(args)
        # we use diffextractor here
        self.extractor = build_extractor(args)
        
        # position
        self.positionlayer = positionLayer(args)
        self.adu_label = args.adu_label
        if(self.adu_label):
            self.adu_label_emb = nn.Embedding(2, 32)

        self.adu_encoder = span_encoder(args)
        self.para_encoder = para_encoder(args)

        # graph part 
        args.nfeat = 2*args.hDim
        self.graph_decoder = HeterogeneousGraph(args)
        
        if(self.graph_decoder.concat):
            args.para_in = args.nfeat + (args.graph_layers-1)*args.nhid*args.nheads + args.nhid
        else:
            args.para_in = args.nhid
        
        
        self.para_decoder = args.para_decoder
        if(self.para_decoder):
            self.para_decoder = para_encoder(args)
            args.nhid = args.hDim*2 # hdim, bidirectional
        else:
            args.nhid = args.hDim # hdim
        

        args.final_extract_type = ['max', 'mean', 'attention']
        self.final_extractor = final_extractor(args, args.final_extract_type)


        self.final = args.final
        if(self.final=='pair'):
            args.nhid *= 2
        #self.fc = nn.Linear(2*args.nhid, 1)
        self.fc = nn.Linear(args.nhid+args.nhid, 1)
        self.direct_fc = nn.Linear(args.para_in, 1)
        self.topic_fc = nn.Sequential(nn.Linear(args.para_in, args.nhid), nn.Tanh())

        # pair part
        self.multitask = args.multitask
        if(self.multitask):
            self.parsing_decoder = TransDecoder(args, args.max_n_spans_para)

    def forward(self, span, shell_span, ac_position_info,
            elmo_emb, topic_emb, author, 
            elmo_length, adu_length, topic_length, 
            para_author=None, para_span=None, para_length=None, adu_label=None,
            adu_graph=None, reply_graph=None, 
            mask=None, para_mask=None, adu_mask=None, multitask=False):

        device = span.device
        # 1. context encoder
        elmo_emb, topic_emb = self.noise(elmo_emb), self.noise(topic_emb)
        topic_reps, word_reps = self.context_encoder(elmo_emb, topic_emb, elmo_length, topic_length)

        # get the relative position embedding and author embedding
        position_info, relative_position_info = self.positionlayer(ac_position_info)
        
        # 2. extract adu from word encoder
        topic_reps, para_reps, span_reps, adu_reps = self.extractor(topic_reps, word_reps, topic_length, para_span, span, shell_span)
        # dropout
                
        #adu_reps = self.dropout2d(adu_reps)
        #span_reps = self.dropout2d(span_reps)

        topic_reps = self.noise(topic_reps)
        span_reps = self.noise(span_reps)
        adu_reps = self.noise(adu_reps)
        
        
        if(self.adu_label):
            adu_label_reps = self.adu_label_emb(adu_label)
        else:
            adu_label_reps = None

        # 3. encode adu, para information
        span_adu_reps = self.adu_encoder(span_reps, adu_reps, position_info, adu_length, adu_label_reps=adu_label_reps)
        
        if(multitask):
            return  self.parsing_decoder(span_adu_reps, topic_reps, adu_length, relative_position_info)
        else:
            total_reps = torch.cat([span_adu_reps, topic_reps.unsqueeze(1)], dim=1)
            # get the paragraph author embedding
    
            # only for persuasive need para feat
            para_reps = self.noise(para_reps)
            #para_reps = self.dropout2d(para_reps)
            para_reps = self.para_encoder(para_reps, para_length)

            # 4. aggregater using graph method
            #para_reps, total_reps = self.noise(para_reps), self.noise(total_reps)
            adu_reps, topic_reps, para_reps = self.graph_decoder(total_reps, para_reps, adu_graph, reply_graph)

            adu_direct = self.direct_fc(adu_reps)
            para_direct = self.direct_fc(para_reps)
            # 5. final aggrgate using single direction lstm
            if(self.para_decoder):
                #adu_out = self.para_decoder(adu_reps, adu_length)
                out = self.para_decoder(para_reps, para_length)
            else:
                out = para_reps

            """
            for para_author, we have three label, 
            other: 0, pos: 1, neg: 2
            """
            if(self.final=='pair'):
                pos_out = self.final_extractor(out, (((para_author==1)|(para_author==0)) & para_mask.byte()) )
                neg_out = self.final_extractor(out, (((para_author==2)|(para_author==0)) & para_mask.byte()) )
                out = torch.cat([pos_out, neg_out], dim=-1)
            else:
                out = self.final_extractor(out, para_mask.byte())    
                #adu_out = self.final_extractor(adu_out, adu_mask.byte())
                #out = torch.cat([out, adu_out], dim=-1)
            
            out = torch.cat([out, self.topic_fc(topic_reps.squeeze(1))], dim=-1)
            out = self.noise(out)
            pair_score = self.fc(out)

            return pair_score, adu_direct, para_direct

