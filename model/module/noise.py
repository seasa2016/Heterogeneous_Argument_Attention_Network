import torch
import torch.nn as nn

class noiselayer(nn.Module):
    def __init__(self, variance, dropout_rate):
        super(noiselayer, self).__init__()
        self.variance = variance
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if(self.training==False or self.variance==0.0):
            pass
        else:
            x = x + torch.randn_like(x)*self.variance
        return self.dropout(x)
        
    def __repr__(self):
        return 'variance:{}, dopout:{}'.format(self.variance, self.dropout_rate)
