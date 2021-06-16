# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig, AlbertTokenizer,
    BertConfig, BertTokenizer,
    DistilBertConfig, DistilBertTokenizer,
    RobertaConfig, RobertaTokenizer,
    XLMConfig, XLMRobertaConfig,
    XLMRobertaTokenizer, XLMTokenizer,
    XLNetConfig, XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from data.dataloader_bert import itemDataset, collate_fn

from model.model_bert import (
    BertForPersuasive, AlbertForPersuasive
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from util.bert_parameter import add_bert_args
from util.parameter import (add_encoder_args, add_graph_args, add_dataset_args)
from util.loss import (hinge)
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForPersuasive, BertTokenizer),
    "albert": (AlbertConfig, AlbertForPersuasive, AlbertTokenizer),
    #"xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    #"xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    #"roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    #"distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    #"albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    #"xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_bert_args(parser)
    parser = add_encoder_args(parser)
    parser = add_graph_args(parser)
    parser = add_dataset_args(parser)

    args = parser.parse_args()
    return args


def score(pred, criterion):
    result = {}
    pred = torch.cat(pred, dim=-1).view(-1)
    result['loss'] = criterion(pred, torch.ones_like(pred)).item()
    result['f1'] = f1_score(torch.ones_like(pred), (pred>0).long(), average='macro')
    result['acc'] = (pred>0).float().mean().item()
    result['diff'] = pred.mean().item()
    
    return result, pred

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def build_data(args, data_path, pair_path, tokenizer):
    dataset = {}
    if(args.do_train):
        dataset['train'] = itemDataset(data_path=data_path+'_train', pair_path=pair_path+'_train', tokenizer=tokenizer)
    
    if(args.do_eval):
        dataset['dev'] = itemDataset(data_path=data_path+'_dev', pair_path=pair_path+'_dev', tokenizer=tokenizer)
    
    if(args.do_predict):
        dataset['test'] = itemDataset(data_path=data_path+'_test', pair_path=pair_path+'_test', tokenizer=tokenizer)
 
    return dataset

def convert(data, device):
    temp = {}
    for key, val in data.items():
        try:
            temp[key] = val.to(device)
        except:
            pass
    return temp

def train(args, dataset, model, tokenizer, criterion):
    """ Train the model """
    if(args.local_rank in [-1, 0]):
        tb_writer = SummaryWriter(logdir = args.output_dir+'/runs/')

    train_dataset = dataset['train']
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    if(args.max_steps > 0):
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if( os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    )):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if(args.fp16):
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    # Distributed training (should be after apex fp16 initialization)
    if(args.local_rank != -1):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if(os.path.exists(args.model_name_or_path)):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    temp_pred = []
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batches in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            outputs = []
            for batch in batches:
                inputs = convert(batch, args.device)

                if args.model_type != "distilbert":
                    if(args.model_type not in ["bert", "xlnet"]):
                        # XLM and RoBERTa don"t use segment_ids
                        batch["token_type_ids"] = None
                        batch["topic_input_ids"] = None

                output = model(**inputs)
                outputs.append(output)
            
            pred = (outputs[1]-outputs[0]).view(-1)
            loss = criterion(pred, torch.ones_like(pred))
            
            temp_pred.append(pred.detach().cpu())

            if(args.n_gpu > 1):
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if(args.gradient_accumulation_steps > 1):
                loss = loss / args.gradient_accumulation_steps

            if(args.fp16):
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if((step + 1) % args.gradient_accumulation_steps == 0):
                if(args.fp16):
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if(args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    # Log metrics
                    if(args.local_rank == -1 and args.evaluate_during_training):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, dataset['dev'], criterion, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    result, _ = score(temp_pred, criterion)
                    for key, val in result.items():
                        tb_writer.add_scalar("train_{}".format(key), val, global_step)
                        
                    logging_loss = tr_loss
                    temp_pred = []

                if(args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if(not os.path.exists(output_dir)):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
            if(args.max_steps > 0 and global_step > args.max_steps):
                epoch_iterator.close()
                break
        if(args.max_steps > 0 and global_step > args.max_steps):
            train_iterator.close()
            break

    if(args.local_rank in [-1, 0]):
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_dataset, criterion, mode, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0

    total = {
        'label':{'link':None, 'type':None, 'link_type':None},
        'pred':{
            'link_mst':None, 'link':None, 'type':None, 'link_type':None
        },
        'adu_len':None
    }

    model.eval()
    total_pred = []
    for batches in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            outputs = []
            for batch in batches:
                inputs = convert(batch, args.device)
                ######################################################################################
                if args.model_type != "distilbert":
                    if(args.model_type not in ["bert", "xlnet"]):
                        # XLM and RoBERTa don"t use segment_ids
                        batch["token_type_ids"] = None
                        batch["topic_input_ids"] = None
                        
                output = model(**inputs)
                outputs.append(output)
            pred = (outputs[1]-outputs[0]).view(-1)
            total_pred.append(pred.detach().cpu())
        nb_eval_steps += 1

    result, pred = score(total_pred, criterion)

    logger.info("***** Eval results %s *****", prefix)
    for key, val in result.items():
        logger.info("%s = %s", key, str(val))
    
    return result, pred


def main():
    args = parse_args()

    if( os.path.exists(args.output_dir) and os.listdir(args.output_dir)
        and args.do_train and not args.overwrite_output_dir):
        pass
        #raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir)

    # deal with logger output
    if(args.do_train):
        fh = logging.FileHandler(args.output_dir+'/train.log')
    elif(args.do_predict):
        fh = logging.FileHandler(args.output_dir+'/test.log')
    elif(args.do_eval):
        fh = logging.FileHandler(args.output_dir+'/eval.log')

    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Setup distant debugging if needed
    if(args.server_ip and args.server_port):
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if(args.local_rank == -1 or args.no_cuda):
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if(args.local_rank not in [-1, 0]):
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if(args.model_type=='albert'):
        model.albert = torch.nn.DataParallel(model.albert)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    
    dataset = build_data(args, args.data_path, args.pair_path, tokenizer)
    
    if(args.criterion =="bce" ):
        criterion = nn.BCEWithLogitsLoss()
    elif(args.criterion == 'hinge'):
        criterion = hinge
    else:
        raise ValueError('no this loss function')

    # Training
    if(args.do_train):
        global_step, tr_loss = train(args, dataset, model, tokenizer, criterion)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if(args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0)):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if(args.do_eval and args.local_rank in [-1, 0]):
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, args=args)
            model.to(args.device)
            result, predictions = evaluate(args, model, tokenizer, dataset['dev'], criterion, mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        torch.save(predictions, os.path.join(args.output_dir, "results.pt"))
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if(args.do_predict and args.local_rank in [-1, 0]):
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir, args=args)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, dataset['test'], criterion, mode="test")

        torch.save(predictions, os.path.join(args.output_dir, "results.pt"))

    return results


if __name__ == "__main__":
    main()
