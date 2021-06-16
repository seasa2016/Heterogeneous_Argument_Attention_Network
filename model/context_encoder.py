import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model import module

class Elmo_encoder(nn.Module):
    def __init__(self, args):
        super(Elmo_encoder, self).__init__()
        ##########################
        # set default attributes #
        ##########################
        self.eDim = args.eDim
        self.hDim = args.hDim
        self.dropout_lstm = args.dropout_lstm
        self.dropout_word = args.dropout_word
        self.dropout_embedding = args.dropout_embedding
        
        self.args = args
        self.word_dropout = nn.Dropout2d(self.dropout_word)
        ################
        # elmo setting #
        ################
        self.eDim = 1024
        args.eDim = 1024
        
        self.elmo_task_gamma = nn.Parameter(torch.ones(1))
        self.elmo_task_s = nn.Parameter(torch.ones(3))
        self.elmo_dropout = nn.Dropout(self.dropout_embedding)

        ##########
        # Default #
        ##########
        if(args.rnn == 'LSTM'):
            self.Bilstm = module.lstm(input_dim=self.eDim, output_dim=self.hDim,
                        num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
        elif(args.rnn == 'GRU'):
            self.Bilstm = module.gru(input_dim=self.eDim, output_dim=self.hDim,
                        num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)

        #self.Topiclstm = module.lstm(input_dim=self.eDim, output_dim=self.hDim,
        #            num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)

        self.init_para()
    def init_para(self):
        def linear_init(model):
            for para in model.parameters():
                init.uniform_(para, -0.05, 0.05)

        #for _ in [self.Bilstm, self.Topiclstm]:
        for _ in [self.Bilstm]:
            _.init()

    def load_elmo(self, elmo_embed):
        if self.args.elmo_layers == "weighted":
            elmo_embeddings = self.elmo_task_s.softmax(-1).view(1, -1, 1, 1) * elmo_embed
            elmo_embeddings = elmo_embeddings.sum(dim=1)
        elif self.args.elmo_layers == "avg":
            elmo_embeddings = elmo_embed.sum(dim=1)/3
        else:
            elmo_embeddings = elmo_embed[:, int(self.args.elmo_layers)-1]

        if self.args.elmo_task_gamma:
            elmo_embeddings = self.elmo_task_gamma * elmo_embeddings

        elmo_embeddings = self.elmo_dropout(elmo_embeddings)

        return elmo_embeddings

    def forward(self, elmo_embeddings, topic_embeddings,
            xs_lens, topic_lens):
        ###################
        # load embeddings #
        ###################
        xs_embed = self.load_elmo(elmo_embeddings)
        topic_embed = self.load_elmo(topic_embeddings)
        xs_embed = self.word_dropout(xs_embed)
        topic_embed = self.word_dropout(topic_embed)
        ###########
        # encoder #
        ###########
        #topic_reps, _ = self.Topiclstm(topic_embed, topic_lens)
        topic_reps, _ = self.Bilstm(topic_embed, topic_lens)
        word_reps, _ = self.Bilstm(xs_embed, xs_lens)
        
        return topic_reps, word_reps
