import torch
import torch.nn as nn

def hinge(x, label=None):
    criterion = nn.ReLU()
    out = criterion(-x+2)

    norm = (out>0).float().sum()

    return out.sum()/norm