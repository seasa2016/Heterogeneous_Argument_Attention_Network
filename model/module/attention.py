import torch
import torch.nn as nn

class single_attention(nn.Module):
    def __init__(self, args):
        super(single_attention, self).__init__()
        self.fc_weight = nn.Linear( args.atten_in, 1)

    def span_extract(self, x, x_spans):
        weight = self.fc_weight(x).squeeze(-1)
        
        span_reps = []
        for row, spans in enumerate(x_spans):
            span_reps.append([])
            for elmo_index, start, end in spans:
                temp_y = x[elmo_index, start:end+1]
                temp_weight = weight[elmo_index, start:end+1].softmax(-1)
                
                span_reps[-1].append(torch.matmul(temp_weight.unsqueeze(-2), temp_y))

            span_reps[-1] = torch.cat(span_reps[-1], dim=-2)

        return torch.stack(span_reps)

    def length_extract(self, x, x_lens):
        weight = self.fc_weight(x).squeeze(-1)
        
        span_reps = []
        for elmo_index, l in enumerate(x_lens):
            temp_y = x[elmo_index, :l]
            temp_weight = weight[elmo_index, :l].softmax(-1)
                
            span_reps.append(torch.matmul(temp_weight.unsqueeze(-2), temp_y))

        return torch.cat(span_reps, dim=-2)
