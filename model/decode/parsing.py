import torch
import torch.nn as nn
import torch.nn.functional as F
from model import module

class TransDecoder(nn.Module):
    def __init__(self, args, max_n_spans):
        super(TransDecoder, self).__init__()
        self.hDim = args.hDim
        self.dropout = args.dropout
        self.dropout_lstm = args.dropout_lstm
        
        self.relative_adu_info_size = args.relative_adu_info_size
        self.relative_post_info_size = args.relative_post_info_size
        self.relative_position_info_size = self.relative_adu_info_size + self.relative_post_info_size

        self.lstm_type = args.lstm_type

        self.dropout = nn.Dropout(self.dropout)

        # output of ADU layer
        if(self.lstm_type):
            self.ac_shell_rep_size_out = args.ac_shell_rep_size_out
            self.LastBilstm = module.lstm(input_dim=self.ac_shell_rep_size_out, output_dim=self.hDim,
                                num_layers=1, batch_first=True, dropout=self.dropout_lstm)
            self.reps_for_type_classification = 2*self.hDim
        else:
            self.reps_for_type_classification = args.ac_shell_rep_size_out

        self.AcTypeLayer = nn.Linear(in_features=self.reps_for_type_classification, out_features=1)

        # the size of ADU representations for link identification
        # share the parameter
        self.type_rep_size = 2*self.hDim if(self.lstm_type) else args.ac_shell_rep_size_out
        self.span_pair_size = self.type_rep_size*3 + self.relative_position_info_size
        self.LinkLayer = nn.Sequential(
                nn.Linear(in_features=self.span_pair_size, out_features=64),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=2)
                )

    def init_para(self):
        def linear_init(model):
            for para in model.parameters():
                init.uniform_(para, -0.05, 0.05)

        # linear init
        for _ in [self.AcTypeLayer, self.LinkLayer]:
            linear_init(_)


    def mask_link_scores(self, pair_scores, adu_len, mask=None, mask_type="minus_inf"):
        device = pair_scores.device

        batchsize, n_spans, _ = pair_scores.shape
        #(max_n_spans, max_n_spans+1)

        if(mask is None):
            mask = torch.stack([
                    torch.eye(n_spans, dtype=torch.long).to(device) for _ in range(batchsize)
                    ])
            mask = 1-mask
        mask = torch.cat([mask, torch.ones(batchsize, n_spans, 1, dtype=torch.long).to(device)], dim=-1)
        
        # mask
        for i, _ in enumerate(adu_len):
            mask[i, :, _:-1] = 0
        if(mask_type == "minus_inf"):
            padding = torch.full((batchsize, n_spans, n_spans+1), -1e16,
                            dtype=torch.float, device=device)
        else:
            padding = torch.zeros(batchsize, n_spans, n_spans+1,
                                 dtype=torch.float, device=device)

        #(batchsize, max_n_spans, max_n_spans+1, 1)
        masked_pair_scores = torch.where(mask.byte(), pair_scores, padding)

        return masked_pair_scores

    def calc_pair_score(self, span_reps_pad, topic_reps, relative_position_info):
        ###########################
        # for link identification #
        ###########################
        #(batchsize, max_n_spans, span_representation)
        batchsize, max_n_spans, _ = span_reps_pad.shape
        device = span_reps_pad.device

        #(batchsize, max_n_spans, max_n_spans, span_representation)
        span_reps_matrix = span_reps_pad.unsqueeze(1).expand(-1, max_n_spans, -1, -1)
        span_reps_matrix_t = span_reps_matrix.transpose(2, 1)

        #(batchsize, max_n_spans, max_n_spans, pair_representation)
        pair_reps = torch.cat(
            [span_reps_matrix,
             span_reps_matrix_t,
             span_reps_matrix*span_reps_matrix_t,
             relative_position_info.float()],
            dim=-1)

        ##########################
        #### add topic object ####
        ##########################

        #(batchsize, max_n_spans, span_rep_size)
        root_matrix = topic_reps.unsqueeze(1).expand_as(span_reps_pad)

        #(batchsize, max_n_spans, pair_rep_size)
        pair_reps_with_root = torch.cat([span_reps_pad,
                                        root_matrix,
                                        span_reps_pad*root_matrix,
                                        torch.zeros(batchsize,
                                        max_n_spans,
                                        self.relative_position_info_size, dtype=torch.float).to(device)
                                        ],
                                        dim=-1)

        #(batchsize, max_n_spans, max_n_spans+1, pair_rep_size)
        pair_reps = torch.cat([pair_reps,
                                pair_reps_with_root.view(
                                                batchsize,
                                                max_n_spans,
                                                1,
                                                self.span_pair_size)],
                            dim=2)

        relation_scores, pair_scores = self.LinkLayer(pair_reps).split(1, dim=-1)

        return pair_scores.squeeze(-1), relation_scores.squeeze(-1)

    def forward(self, span_reps, topic_reps, adu_len, relative_position_info, mask=None):
        device = span_reps.device

        if(self.lstm_type):
            span_reps, _ = self.LastBilstm(span_reps, adu_len)
            span_reps = self.dropout(span_reps)
        ac_types = self.AcTypeLayer(span_reps).squeeze(-1)

        ############################
        # span pair representation #
        ############################
        pair_scores, link_types = self.calc_pair_score(span_reps, topic_reps, relative_position_info)

        # mask non pair
        masked_pair_scores = self.mask_link_scores(pair_scores, adu_len, mask, mask_type="minus_inf")

        return ac_types, masked_pair_scores, link_types
