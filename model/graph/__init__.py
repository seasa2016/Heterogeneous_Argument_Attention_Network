import torch.nn as nn
import torch
from model.graph.heterogeneous import HeterogeneousLayer
import model.module as module

class HeterogeneousGraph(nn.Module):
    def __init__(self, args):
        super(HeterogeneousGraph, self).__init__()

        self.nhid = args.nhid
        self.layers_len = args.graph_layers
        self.layers = nn.ModuleList()
        self.layers.append(HeterogeneousLayer(args.nfeat, args.nhid, args.nheads,args.ngraph, 
                        dropout=args.dropout, alpha=args.graph_alpha, concat=True, dtype=args.graph))
        for _ in range(args.graph_layers-1):
            self.layers.append(HeterogeneousLayer(args.nhid, args.nhid, args.nheads, args.ngraph,
                        dropout=args.dropout, alpha=args.graph_alpha, concat=True, dtype=args.graph))
        self.dropout = nn.Dropout(args.dropout)
        self.concat = args.concat

        if(args.rnn == 'LSTM'):
            self.rnn = module.lstm(args.nhid, args.nhid, bidirectional=False)
        elif(args.rnn == 'GRU'):
            self.rnn = module.gru(args.nhid, args.nhid, bidirectional=False)

    def forward(self, adu_x, para_x, adu_adj, reply_adj):
        batchsize, adu_len, _ = adu_x.shape
        batchsize, para_len, _ = para_x.shape

        temp = [[adu_x], [para_x]]
        for i, layer in enumerate(self.layers):
            adu_x, para_x = self.dropout(adu_x), self.dropout(para_x)
            adu_x, para_x = layer(adu_x, para_x, adu_adj, reply_adj)
            temp[0].append(adu_x)
            temp[1].append(para_x)
            #print(adu_x.shape, para_x.shape, flush=True)
        
        outs = []
        for _ in temp:
            length = len(_)
            _ = torch.stack(_, 2).view(-1, length, self.nhid)
            reps, hidden = self.rnn(_, torch.zeros(_.shape[0])+length)
            outs.append(reps[:,-1])

        adu_x, para_x = outs
        adu_x = adu_x.view(batchsize, adu_len, -1)
        para_x = para_x.view(batchsize, para_len, -1)
            
        return adu_x[:,:-1], adu_x[:,-1].unsqueeze(1), para_x
