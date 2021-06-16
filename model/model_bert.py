import torch
import torch.nn as nn
from model.adu_encoder import (span_encoder, para_encoder)
from model.extractor import (maxpool_extractor, diff_extractor, end_extractor)
from model.graph import GCN


from transformers import (
    BertPreTrainedModel, BertModel,
    AlbertPreTrainedModel, AlbertModel,
    DistilBertPreTrainedModel,
    XLMPreTrainedModel,
    XLNetPreTrainedModel
)

class BertForPersuasive(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config, args):
        super(BertForPersuasive, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        args.hDim = 768
        args.span_rep_size = args.hDim
        args.para_in = args.hDim
        self.adu_encoder = span_encoder(args)
        self.para_encoder = para_encoder(args)
        self.dropout_lstm = nn.Dropout(args.dropout_lstm)

        args.nfeat = 2*args.hDim
        self.decoder = GCN(args)
        self.fc = nn.Sequential(
                nn.Linear(args.nhid, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, args.nclass)
                )
        self.init_weights()

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None,
        topic_input_ids=None, topic_attention_mask=None, topic_token_type_ids=None,
        span=None, shell_span=None, para_span=None, ac_position_info=None,
        adu_length=None, para_length=None, topic_length=None, sent_length=None,
        adu_graph=None, reply_graph=None, sum_mask=None
    ):
        device = input_ids.device

        # 1. context encoder
        # paragraph
        outputs = self.bert(input_ids, 
                attention_mask=attention_mask, token_type_ids=token_type_ids)
        content_feat = outputs[0]
        # topic        
        outputs = self.bert(topic_input_ids,
            attention_mask=topic_attention_mask, token_type_ids=topic_token_type_ids)
        topic_feat = outputs[0]
        
        # 2. extract adu from word encoder
        para_reps = content_feat[:, 0]
        topic_reps = topic_feat[:,0]

        content_reps = content_feat[:, 1:]
        span_reps, adu_reps = maxpool_extractor(content_reps, span), maxpool_extractor(content_reps, shell_span)

        span_reps, adu_reps = self.dropout(span_reps), self.dropout(adu_reps)
        para_reps = self.dropout(para_reps)
        topic_reps = self.dropout(topic_reps)

        # 3. encode adu, para information
        para_reps = self.para_encoder(para_reps, para_length)
        span_adu_reps = self.adu_encoder(span_reps, adu_reps, ac_position_info, adu_length)
        total_reps = torch.cat([span_adu_reps, topic_reps.unsqueeze(1)], dim=1)

        total_reps, para_reps = self.dropout_lstm(total_reps), self.dropout_lstm(para_reps)

        # 4. aggregate using fraph information
        out = self.decoder(total_reps, para_reps, adu_graph, reply_graph)

        # 5. take out the last reply
        sum_mask = sum_mask.float()
        out = torch.diag(1/sum_mask.sum(-1)).to(device).unsqueeze(1).matmul(sum_mask).matmul(out).squeeze(1)
        out = self.fc(out)
        return out

class AlbertForPersuasive(AlbertPreTrainedModel):
    def __init__(self, config, args):
        super(AlbertForPersuasive, self).__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        args.hDim = 768
        args.span_rep_size = args.hDim
        args.para_in = args.hDim
        args.hDim = 768//2
        self.adu_encoder = span_encoder(args)
        self.para_encoder = para_encoder(args)
        self.dropout_lstm = nn.Dropout(args.dropout_lstm)

        args.nfeat = 2*args.hDim
        self.decoder = GCN(args)
        self.fc = nn.Sequential(
                nn.Linear(args.nhid, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, args.nclass)
                )
        self.init_weights()

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None,
        topic_input_ids=None, topic_attention_mask=None, topic_token_type_ids=None,
        span=None, shell_span=None, para_span=None, ac_position_info=None,
        adu_length=None, para_length=None, topic_length=None, sent_length=None,
        adu_graph=None, reply_graph=None, sum_mask=None
    ):
        device = input_ids.device

        # 1. context encoder
        # paragraph
        outputs = self.albert(input_ids, 
                attention_mask=attention_mask, token_type_ids=token_type_ids)
        content_feat = outputs[0]
        # topic        
        outputs = self.albert(topic_input_ids,
            attention_mask=topic_attention_mask, token_type_ids=topic_token_type_ids)
        topic_feat = outputs[0]
        
        # 2. extract adu from word encoder
        para_reps = content_feat[:, 0]
        topic_reps = topic_feat[:,0]

        content_reps = content_feat[:, 1:]
        span_reps, adu_reps = maxpool_extractor(content_reps, span), maxpool_extractor(content_reps, shell_span)

        span_reps, adu_reps = self.dropout(span_reps), self.dropout(adu_reps)
        para_reps = self.dropout(para_reps)
        topic_reps = self.dropout(topic_reps)

        # 3. encode adu, para information
        para_reps = self.para_encoder(para_reps, para_length)
        span_adu_reps = self.adu_encoder(span_reps, adu_reps, ac_position_info, adu_length)
        total_reps = torch.cat([span_adu_reps, topic_reps.unsqueeze(1)], dim=1)

        total_reps, para_reps = self.dropout_lstm(total_reps), self.dropout_lstm(para_reps)

        # 4. aggregate using fraph information
        out = self.decoder(total_reps, para_reps, adu_graph, reply_graph)

        # 5. take out the last reply
        sum_mask = sum_mask.float()
        out = torch.diag(1/sum_mask.sum(-1)).to(device).unsqueeze(1).matmul(sum_mask).matmul(out).squeeze(1)
        out = self.fc(out)
        return out
