import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model import module
#import module

class para_encoder(nn.Module):
    def __init__(self, args, bidirectional=True):
        super(para_encoder, self).__init__()
        self.args = args
        if(args.rnn == 'LSTM'):
            self.encoder = module.lstm(args.para_in, args.hDim, bidirectional=bidirectional)
        elif(args.rnn == 'GRU'):
            self.encoder = module.gru(args.para_in, args.hDim, bidirectional=bidirectional)
        self.init()

    def init(self):
        def linear_init(model):
            for para in model.parameters():
                init.uniform_(para, -0.05, 0.05)

        # lstm init
        for _ in [self.encoder]:
            _.init()

    def forward(self, x_reps, x_lens):
        x_reps, _ = self.encoder(x_reps, x_lens)

        return x_reps

if(__name__ == '__main__'):
    class temp:
        pass
    temp.hDim = 256
    model = para_encoder(temp)

    x = torch.rand(8, 36, 1024)
    x_len = torch.tensor([4, 6, 36, 2, 1, 2, 3, 4],dtype = torch.long)
    model(x, x_len)



class span_encoder(nn.Module):
    def __init__(self, args):
        super(span_encoder, self).__init__()

        ##########################
        # set default attributes #
        ##########################
        self.hDim = args.hDim
        self.dropout_lstm = args.dropout_lstm
        self.max_n_spans = args.max_n_spans_para
        self.position_info_size = args.position_info_size
        self.args = args

        ###############
        # Select LSTM #
        ###############
        self.lstm_ac = args.lstm_ac
        self.lstm_shell = args.lstm_shell
        self.lstm_ac_shell = args.lstm_ac_shell
        self.adu_label = args.adu_label
        
        label_size = 32
        self.span_rep_size = args.span_rep_size
        # output of AC layer
        if(self.lstm_ac):
            self.ac_rep_size = self.hDim*2
            if(self.adu_label):
                self.span_rep_size += label_size

            if(args.rnn == 'LSTM'):
                self.AcBilstm = module.lstm(input_dim=self.span_rep_size, output_dim=self.hDim,
                        num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
            elif(args.rnn == 'GRU'):
                self.AcBilstm = module.gru(input_dim=self.span_rep_size, output_dim=self.hDim,
                        num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
        else:
            self.ac_rep_size =  self.span_rep_size

        # output of AM layer
        if(self.lstm_shell):
            self.shell_rep_size = self.hDim*2
            if(self.adu_label):
                self.span_rep_size += label_size
            if(args.rnn == 'LSTM'):
                self.ShellBilstm = module.lstm(input_dim=self.span_rep_size, output_dim=self.hDim,
                    num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
            elif(args.rnn == 'GRU'):
                self.ShellBilstm = module.gru(input_dim=self.span_rep_size, output_dim=self.hDim,
                    num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
        else:
            self.shell_rep_size = self.span_rep_size

        # the size of ADU representation
        n_ac_shell_latm_layers = 1
        self.ac_shell_rep_size_in = self.ac_rep_size + self.shell_rep_size + self.position_info_size*2
        if(self.adu_label):
            self.ac_shell_rep_size_in += label_size
        if(args.rnn == 'LSTM'):
            self.AcShellBilstm = module.lstm(input_dim=self.ac_shell_rep_size_in, output_dim=self.hDim,
                num_layers=n_ac_shell_latm_layers, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
        elif(args.rnn == 'GRU'):
            self.AcShellBilstm = module.gru(input_dim=self.ac_shell_rep_size_in, output_dim=self.hDim,
                num_layers=n_ac_shell_latm_layers, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
        args.ac_shell_rep_size_out = self.hDim*2

        self.init_para()
    def init_para(self):
        def linear_init(model):
            for para in model.parameters():
                init.uniform_(para, -0.05, 0.05)

        # lstm init
        if(self.lstm_ac):
            self.AcBilstm.init()
        if(self.lstm_shell):
            self.ShellBilstm.init()
        for _ in [self.AcShellBilstm]:
            _.init()
        
    def forward(self, shell_reps, ac_reps, position_info, adu_lens, adu_label_reps=None):
        device = shell_reps.device

        # prepare adu representation
        if(self.lstm_ac):
            if(self.adu_label):
                ac_reps = torch.cat([ac_reps, adu_label_reps], dim=-1)

            ac_reps, _ = self.AcBilstm(ac_reps, adu_lens)
        if(self.lstm_shell):
            if(self.adu_label):
                shell_reps = torch.cat([shell_reps, adu_label_reps], dim=-1)
            shell_reps, _ = self.ShellBilstm(shell_reps, adu_lens)

        ac_shell_reps = torch.cat([ac_reps, shell_reps, position_info.float()], dim=-1)
        if(self.adu_label):
            ac_shell_reps = torch.cat([ac_shell_reps, adu_label_reps], dim=-1)

        assert(ac_shell_reps.shape[-1] == self.ac_shell_rep_size_in), '{} {}'.format(ac_shell_reps.shape[-1], self.ac_shell_rep_size_in)

        final_reps, _ = self.AcShellBilstm(ac_shell_reps, adu_lens)

        return final_reps
