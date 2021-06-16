import torch
import torch.nn as nn

class final_extractor(nn.Module):
    def __init__(self, args, dtype=['mean', 'max']):
        super(final_extractor, self).__init__()

        self.module = []
        self.dtype = dtype
        
        mag = 0
        if('max' in self.dtype):
            self.module.append(self.max)
            mag += 1
        if('mean' in self.dtype):
            self.module.append(self.mean)
            mag += 1
        if('attention' in self.dtype):
            self.dtype.append(str(args.nhid))
            self.module.append(self.attention)
            self.fc_weight = nn.Linear( args.nhid, 1)
            mag += 1

        args.nhid = args.nhid*mag

    def forward(self, feats, mask):
        lengths = mask.long().sum(-1)

        temp_feat = feats[mask>0]
        temp = []
        start, end = 0, 0
        for l in lengths:
            end += l.item()
            temp.append(
                torch.cat([ m(temp_feat[start:end]) for m in self.module], dim=-1)
                )
            
            start = end
        
        return torch.stack(temp)

    def mean(self, feats):
        return feats.mean(0)

    def max(self, feats):
        return feats.max(0)[0]

    def attention(self, feats):
        weight = self.fc_weight(feats).squeeze(-1).softmax(-1)
            
        return torch.matmul(weight, feats)

    def __repr__(self):
        return ', '.join(self.dtype)

if(__name__ == '__main__'):
    class T:
        pass
    t = T()
    t.nhid = 4
    feats = torch.rand(2, 4, 2)
    print(feats)
    lengths = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 1]], dtype=torch.long)

    model = final_extractor(t, dtype=['mean', 'max', 'attention'])
    print(model(feats, lengths))
