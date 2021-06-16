import os
from datetime import datetime

def add_test_args(parser):
    parser.add_argument("--batchsize", default=64, type=int, help="batch size for input data")
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--pred-path", required=True, type=str)

    
    return parser


def add_default_args(parser):
    parser.add_argument("--seed", default=39, type=int)
    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument("--dev", default=False, action='store_true')
    parser.add_argument("--test", default=False, action='store_true')
    
    
    parser.add_argument("--criterion", default='hinge', type=str)
    parser.add_argument("--direct", default='left', type=str)
    

    return parser


def add_encoder_args(parser):
    parser.add_argument("--extractor", default='pool', type=str)
    parser.add_argument("--rnn", default='LSTM', type=str)

    ############
    # feature　＃
    ############
    parser.add_argument("--adu-label", action="store_true")
    parser.add_argument("--adu-alpha", default=0.01, type=float)
    parser.add_argument("--para-alpha", default=1, type=float)
    parser.add_argument("--author-feat", action="store_true")
    parser.add_argument("--adu-out", action="store_true")

    ######################
    # Hierarchical LSTMs #
    ######################
    parser.add_argument("--lstm-ac", action="store_true")
    parser.add_argument("--lstm-shell", action="store_true")
    parser.add_argument("--lstm-ac-shell", action="store_true" )

    ##############
    # dimensions #
    ##############
    parser.add_argument("-ed", "--eDim", default=300, type=int)
    parser.add_argument("-hd", "--hDim", default=256, type=int)

    parser.add_argument("--max_n_spans_para", default=128, type=int)
    ###########
    # dropout #
    ###########
    parser.add_argument("-d", "--dropout", default=0.5, type=float)
    parser.add_argument("-dl", "--dropout-lstm", default=0.1, type=float)
    parser.add_argument("-de", "--dropout-embedding", default=0.1, type=float )
    parser.add_argument("-dw", "--dropout-word", default=0.3, type=float )

    return parser

def add_graph_args(parser):
    parser.add_argument("--nhid", default=256, type=int)
    parser.add_argument("--nclass", default=1, type=int)
    parser.add_argument("--nheads", default=1, type=int)
    parser.add_argument("--graph_layers", default=1, type=int)

    parser.add_argument("--graph_alpha", default=0.01, type=float)
    parser.add_argument("--graph", default='GCN', type=str)
    parser.add_argument("--ngraph", default=4, type=int)
    parser.add_argument("--concat", action="store_true" )

    parser.add_argument("--para_decoder", action="store_true" )
    parser.add_argument("--final", default='pair', type=str)

    parser.add_argument("--top", default=3, type=int)

    return parser


def add_optim_args(parser):

    #############
    # optimizer #
    #############
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    return parser



def add_trainer_args(parser):

    #############
    # iteration #
    #############
    parser.add_argument("--epoch", default=32, type=int)
    parser.add_argument("--batchsize", default=4, type=int)
    parser.add_argument("--accumulate", default=4, type=int)
    parser.add_argument("--lr_step", default=10, type=int)
    parser.add_argument("--lr_gamma", default=0.8, type=float)
    parser.add_argument("--grad_clip", default=10, type=float)
    parser.add_argument("--save-path", required=True, type=str)
    parser.add_argument("--total", action="store_true")
    
    return parser


def add_embed_args(parser):
    ########
    # ELMo #
    ########
    parser.add_argument("--use-elmo", type=int, default=0)
    parser.add_argument("--elmo-path", required=True, type=str)

    parser.add_argument(
            "--elmo-layers",
            choices=["1", "2", "3", "avg", "weighted"],
            default="avg"
            )

    parser.add_argument("--elmo-task-gamma", action="store_true")

    return parser


def add_dataset_args(parser):
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--pair-path", required=True, type=str)

    return parser


def add_parsing_args(parser):
    parser.add_argument("--parsing_path", default='/nfs/nas-5.1/kyhuang/preprocess/span/', type=str)
    parser.add_argument("--lstm-type", action="store_true")
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--ac-type-alpha", default=0.25, type=float)
    parser.add_argument("--link-type-alpha", default=0.25, type=float)
    parser.add_argument("--tree-count", default=2, type=int, help='update every (?) time persuasive training')

    return parser
