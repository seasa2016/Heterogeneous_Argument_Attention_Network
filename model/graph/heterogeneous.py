import torch
import torch.nn as nn
from model.graph.aggregate import SemanticAttention, SemanticMax
from model.graph.gcn import GraphConvolutionLayer
from model.graph.gat import GraphAttentionLayer

class HeterogeneousLayer(nn.Module):
    def __init__(self, feat_in, feat_out, nheads, ngraphs,
                        dropout=0.1, alpha=0.01, concat=False, dtype='GCN'):
        super(HeterogeneousLayer, self).__init__()
        self.ngraphs = ngraphs
        if(dtype == 'GCN'):
            self.layers = nn.ModuleList([
                GraphConvolutionLayer(feat_in, feat_out) for _ in range(ngraphs)
            ])
        elif(dtype == 'GAT'):
            self.layers = nn.ModuleList([
                GraphAttentionLayer(feat_in, feat_out, nheads, 
                dropout=dropout, alpha=alpha, concat=concat) for _ in range(ngraphs)
            ])
        else:
            raise ValueError('no this implement')
        self.aggregate = SemanticAttention(feat_out if(concat or (dtype=='GCN')) else feat_out//nheads, 16)
        #self.aggregate = SemanticMax(feat_out if(concat or (dtype=='GCN')) else feat_out//nheads, 16)
        self.graph_select = [0,1,2,3]
        temp = {0:'neg', 1:'pos', 2:'all', 3:'hier'}
        print('graph select:', end=' ')
        for _ in self.graph_select:
            print(temp[_], end=' ')
        print()
        self.dropout = nn.Dropout(dropout)
    def forward(self, adu_x, para_x, adu_adjs, para_adj):
        """
        batch_size, max_adu_len, _ = adu_x.shape
        
        outs = []
        for (layer, adu_adj) in zip(self.layers[:-1], adu_adjs):
            outs.append(layer(adu_x, adu_adj))
            
        select_outs = [outs[_] for _ in self.graph_select]
        adu_output = self.aggregate(torch.stack(select_outs, dim=-2))

        return adu_output, para_x
        """
        
        batch_size, max_adu_len, _ = adu_x.shape
        para_support = torch.cat([adu_x, para_x], dim=1)

        outs = []
        for (layer, adu_adj) in zip(self.layers[:-1], adu_adjs):
            outs.append(layer(para_support, adu_adj))
        
        outs.append( self.layers[-1](para_support, para_adj) )
        
        select_outs = [outs[_] for _ in self.graph_select]
        para_output = self.aggregate(torch.stack(select_outs, dim=-2))

        return para_output[:, :max_adu_len], para_output[:, max_adu_len:]
        """
        batch_size, max_adu_len, _ = adu_x.shape
        para_support = torch.cat([adu_x, para_x], dim=1)
        para_output = self.layers[-1](para_support, para_adj)
        
        return para_output[:, :max_adu_len], para_output[:, max_adu_len:]
        """