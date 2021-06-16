import torch
import torch.nn as nn
import torch.nn.functional as f

class positionLayer(nn.Module):
    def __init__(self, args):
        super(positionLayer, self).__init__()

        self.position_info_max = 16
        self.relative_post_info_max = 4
        self.relative_adu_info_max = 16

        self.position_info_size = 16
        self.relative_post_info_size = 16
        self.relative_adu_info_size = 16

        args.position_info_max = self.position_info_max
        args.relative_post_info_max = self.relative_post_info_max
        args.relative_adu_info_max = self.relative_adu_info_max
        args.position_info_size = self.position_info_size
        args.relative_post_info_size = self.relative_post_info_size
        args.relative_adu_info_size = self.relative_adu_info_size
        
        self.pos_post_emb = nn.Embedding(self.position_info_max, self.position_info_size)
        self.pos_para_emb = nn.Embedding(self.position_info_max, self.position_info_size)

        self.dist_post_emb = nn.Embedding(self.relative_post_info_max, self.relative_post_info_size)
        self.dist_para_emb = nn.Embedding(self.relative_adu_info_max, self.relative_adu_info_size)

    def forward(self, x_position_info):
        position_info = self.get_position_info(x_position_info)
        relative_position_info = self.get_relative_position_info(x_position_info)

        return position_info, relative_position_info
        
    def clamp(self, inds, dim):
        inds = inds.long().abs()
        inds = inds.clamp(0, dim-1)
        return inds

    def get_position_info(self, x_position_info):
        # the number of ACs in a batch
        batch_size, max_n_spans, _ = x_position_info.shape

        #(batchsize, 3, max_n_spans)
        pos_info = self.clamp(x_position_info, self.position_info_max)
        pos_emb = torch.cat([
            self.pos_post_emb(pos_info[:,:,0]), self.pos_para_emb(pos_info[:,:,1])
        ], dim=-1)

        #(batchsize, max_n_spans, self.position_info_size*2)
        pos_emb = pos_emb.view(batch_size, max_n_spans, self.position_info_size*2)
        return pos_emb

    def get_relative_position_info(self, x_position_info):
        #(batchsize, max_n_spans, 3)
        batch_size, max_n_spans, _ = x_position_info.shape

        span_position_info_matrix = x_position_info.unsqueeze(1).expand(-1, max_n_spans, -1, -1)
        span_position_info_matrix_t = span_position_info_matrix.transpose(2, 1)

        # relative position information
        span_relative_position_info_matrix = span_position_info_matrix - span_position_info_matrix_t
        
        
        relative_adu_info =  span_relative_position_info_matrix[:, :, :, 0]
        relative_post_info = span_relative_position_info_matrix[:, :, :, 1]

        relative_adu_info = self.clamp(relative_adu_info, self.relative_adu_info_max)
        relative_post_info = self.clamp(relative_post_info, self.relative_post_info_max)
        
        relative_adu_info = self.dist_para_emb(relative_adu_info)
        relative_post_info = self.dist_post_emb(relative_post_info)

        #(batchsize, max_n_spans, max_n_spans, relative_position_info_size)
        relative_adu_info = relative_adu_info.view(
                                        batch_size,
                                        max_n_spans,
                                        max_n_spans,
                                        self.relative_adu_info_size)
        relative_post_info = relative_post_info.view(
                                        batch_size,
                                        max_n_spans,
                                        max_n_spans,
                                        self.relative_post_info_size)

        relative_position_info = torch.cat([relative_adu_info, relative_post_info], dim=-1)
        return relative_position_info
