import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def pack(seq, seq_length, batch_first):
    sorted_seq_lengths, indices = torch.sort(seq_length, descending=True)
    _, desorted_indices = torch.sort(indices, descending=False)

    if(batch_first):
        seq = seq[indices]
    else:
        seq = seq[:, indices]

    packed_inputs = nn.utils.rnn.pack_padded_sequence(seq,
                                                    sorted_seq_lengths.cpu().numpy(),
                                                    batch_first=batch_first)
    return packed_inputs, sorted_seq_lengths, desorted_indices

def unpack(res, state, desorted_indices, batch_first):
    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=batch_first)

    device = padded_res.device
    trunc_batch, trunc_sent, hidden_dim = padded_res.shape

    if(isinstance(state, list) or isinstance(state, tuple)):
        state = [ _[:,desorted_indices] for _ in state]
    else:
        state = state[:,desorted_indices]

    if(batch_first):
        desorted_res = padded_res[desorted_indices]
    else:
        desorted_res = padded_res[:, desorted_indices]

    return desorted_res, state

class lstm(nn.Module):
    """
    self fefine module for dealing with pack and unpack
    """
    def __init__(self, input_dim, output_dim, num_layers=1, batch_first=True, dropout=0, bidirectional=True):
        super(lstm, self).__init__()

        self.embedding_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers

        self.batch_first = batch_first
        self.dropout =dropout
        self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=bidirectional, batch_first=self.batch_first, dropout=self.dropout)

    def init(self):
        # orthogonal initial
        for para in self.encoder.parameters():
            if(len(para.shape)>1):
                init.orthogonal_(para)

    def forward(self, word_embeds, word_seq_lengths):
        #################
        # sentence part #
        #################
        word_seq_lengths = word_seq_lengths.view(-1)

        # we truncate the zero length data
        packed_inputs, _, desorted_indices = pack(word_embeds, word_seq_lengths, self.batch_first)
        res, state = self.encoder(packed_inputs)
        result, hidden = unpack(res, state, desorted_indices, self.batch_first)

        return result, hidden

class gru(nn.Module):
    """
    self fefine module for dealing with pack and unpack
    """
    def __init__(self, input_dim, output_dim, num_layers=1, batch_first=True, dropout=0, bidirectional=True):
        super(gru, self).__init__()

        self.embedding_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers

        self.batch_first = batch_first
        self.dropout =dropout
        self.encoder = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=bidirectional, batch_first=self.batch_first, dropout=self.dropout)

    def init(self):
        # orthogonal initial
        for para in self.encoder.parameters():
            if(len(para.shape)>1):
                init.orthogonal_(para)

    def forward(self, word_embeds, word_seq_lengths):
        #################
        # sentence part #
        #################
        word_seq_lengths = word_seq_lengths.view(-1)
        # we truncate the zero length data
        packed_inputs, _, desorted_indices = pack(word_embeds, word_seq_lengths, self.batch_first)
        res, state = self.encoder(packed_inputs)
        result, hidden = unpack(res, state, desorted_indices, self.batch_first)

        return result, hidden

if(__name__ == '__main__'):
    
    x = torch.rand(2, 3, 2)
    length = torch.tensor([2, 3], dtype=torch.long)
    
    model = lstm(2, 3, 1, bidirectional=False)
    a, b = model(x, length)
    print(a)
    print(b)
    print()
    model = gru(2, 3, 1, bidirectional=False)
    a, b = model(x, length)
    print(a)
    print(b)
    print()
