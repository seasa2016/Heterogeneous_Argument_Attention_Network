"""
Here implement several extractor for edu representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import single_attention

def build_extractor(args):
    if(args.extractor=='diff'):
        return diff_extractor(args)
    elif(args.extractor=='pool'):
        return max_extractor(args)
    elif(args.extractor=='attention'):
        return attention_extractor(args)
    else:
        raise NotImplementedError('no this extractor')


class diff_extractor(nn.Module):
    def __init__(self, args):
        super(diff_extractor, self).__init__()
        args.span_rep_size = args.hDim*4
        args.para_in = args.hDim*4

    def forward(self, topic_reps, word_reps, topic_lens, para_spans, x_spans, shell_spans):
        topic_reps = end_extract(topic_reps, topic_lens)
        if(para_spans is not None):
            para_reps = diff_extract(word_reps, para_spans)
        else:
            para_reps = None
        adu_reps = diff_extract(word_reps, x_spans)
        span_reps = diff_extract(word_reps, shell_spans)

        return topic_reps, para_reps, span_reps, adu_reps

class max_extractor(nn.Module):
    def __init__(self, args):
        super(max_extractor, self).__init__()
        args.span_rep_size = args.hDim*2
        args.para_in = args.hDim*2

    def forward(self, topic_reps, word_reps, topic_lens, para_spans, x_spans, shell_spans):
        topic_reps = topic_reps.max(1)[0]
        para_reps = maxpool_extract(word_reps, para_spans)
        adu_reps = maxpool_extract(word_reps, x_spans)
        span_reps = maxpool_extract(word_reps, shell_spans) 

        return topic_reps, para_reps, span_reps, adu_reps

class attention_extractor(nn.Module):
    def __init__(self, args):
        super(attention_extractor, self).__init__()
        args.atten_in = args.hDim*2
        self.extractor = single_attention(args)

        args.span_rep_size = args.hDim*2
        args.para_in = args.hDim*2

    def forward(self, topic_reps, word_reps, topic_lens, para_spans, x_spans, shell_spans):
        topic_reps = self.extractor.length_extract(topic_reps, topic_lens)
        para_reps = self.extractor.span_extract(word_reps, para_spans)
        span_reps = self.extractor.span_extract(word_reps, x_spans)
        adu_reps = self.extractor.span_extract(word_reps, shell_spans)

        return topic_reps, para_reps, span_reps, adu_reps


def diff_extract(ys_l, x_spans=None):
    hDim = ys_l.shape[-1]>>1

    span_reps = []
    for row, spans in enumerate(x_spans):
        span_reps.append([])
        for elmo_index, start, end in spans:
            # print(elmo_index, start, end)
            start_hidden_states_forward = ys_l[elmo_index, start-1, :hDim]
            end_hidden_states_forward = ys_l[elmo_index, end, :hDim]

            start_hidden_states_backward = ys_l[elmo_index, end+1, hDim:]
            end_hidden_states_backward = ys_l[elmo_index, start, hDim:]

            span_forward = end_hidden_states_forward - start_hidden_states_forward
            span_backward = end_hidden_states_backward - start_hidden_states_backward

            span_reps[-1].append(torch.cat(
            [span_forward, span_backward, start_hidden_states_forward, start_hidden_states_backward],
            dim=-1))

        span_reps[-1] = torch.stack(span_reps[-1])

    return torch.stack(span_reps)

def end_extract(ys_l, x_len=None, single=False):
    if(single):
        span_reps = [
            ys_l[row, l-1] for row, l in enumerate(x_len)
            ]

    else:
        hDim = ys_l.shape[-1]>>1
        span_reps = [
            torch.cat([ys_l[row, l-1, :hDim], ys_l[row, 0, hDim:]], dim=-1) for row, l in enumerate(x_len)
            ]

    return torch.stack(span_reps)

def maxpool_extract(ys_l, x_spans=None):
    span_reps = []
    for row, spans in enumerate(x_spans):

        span_reps.append(
            torch.stack([
                    ys_l[elmo_index, start:end+1].max(0)[0] for elmo_index, start, end in spans
            ]))

    return torch.stack(span_reps)
