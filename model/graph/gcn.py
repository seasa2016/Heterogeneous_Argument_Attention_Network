import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = None
        if(bias):
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.activate = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)

        self.linear.weight.data.uniform_(-stdv, stdv)

        if(self.bias is not None):
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        batchsize, max_adu_len, _ = x.shape
        # adu part
        support = self.linear(x)
        output = torch.matmul(adj, support)
        
        if(self.bias is not None):
            output += self.bias
        return self.activate(output)
