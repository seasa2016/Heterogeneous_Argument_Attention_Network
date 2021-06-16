import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, nhead, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.concat = concat

        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(in_features, out_features)
        self.fcs = nn.ModuleList(
            [nn.Linear(2*out_features//self.nhead, 1, bias=False) for _ in range(self.nhead)]
            )
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        for _ in [self.W]:
            nn.init.xavier_uniform_(_.weight.data, gain=1.414)
        for _ in self.fcs:
            nn.init.xavier_uniform_(_.weight.data, gain=1.414)

    def forward(self, x, adj):
        out = self.W(x)
        batchsize, sent_dim, feat_dim = out.shape
        outs = out.split(feat_dim//self.nhead, -1)

        h_primes = []
        for i, (out, fc) in enumerate(zip(outs, self.fcs)):
            a_input = torch.cat([   
                out.repeat(1, sent_dim, 1).view(batchsize, sent_dim*sent_dim, -1),
                out.repeat(1, 1, sent_dim).view(batchsize, sent_dim*sent_dim, -1)
                ], dim=-1).view(batchsize, sent_dim, sent_dim, -1)
            
            e = self.leakyrelu( fc(a_input).squeeze(-1))

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = attention.softmax(dim=-1)
            attention = self.dropout(attention)
            h_prime = torch.matmul(attention, out)
            h_primes.append(h_prime)

        if(self.concat):
            h_primes = torch.cat(h_primes, dim=-1)
            return F.elu(h_primes)
        else:
            """
            for i in range(self.nhead):
                h_primes[i] = F.elu(h_primes[i])
            """
            return F.elu(sum(h_primes)/self.nhead)

