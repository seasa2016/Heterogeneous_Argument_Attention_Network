import torch
import torch.nn as nn

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, z):
        batchsize, length, n_graph, _ = z.shape
        device = z.device

        z = z.view(batchsize*length, n_graph, -1)

        w = self.project(z).squeeze(-1)
        mask = torch.ones(batchsize*length, n_graph).to(device)
        if(self.training):
            mask = self.dropout(mask)

        w = torch.where((mask>0).byte(), w, torch.zeros(batchsize*length, n_graph, dtype=torch.float).to(device)-1e18)

        beta = torch.softmax(w, dim=-1).unsqueeze(-2)

        out = torch.bmm(beta, z).view(batchsize, length, _)
        return out

class SemanticMax(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticMax, self).__init__()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, z):
        z = self.dropout(z)
        return z.max(-2)[0]
