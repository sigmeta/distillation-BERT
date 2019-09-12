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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json
import re
import math
import collections

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForPolyphonyMulti, BertConfig, WEIGHTS_NAME, CONFIG_NAME, OPTIMIZER_NAME,BertForPolyphonyInference
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for polyphony classification."""

    def __init__(self, guid, text, char, label=None, position=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_: string. The untokenized text of the sequence.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            position: int. The position of the polyphony token.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.position = position
        self.char = char


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_ids, label_pos, char):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_ids
        self.label_pos = label_pos
        self.char = char


class DataProcessor(object):
    """Base class for data converters for polyphony classification data sets."""

    def __init__(self, test_set):
        self.test_set = test_set

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.json")))
        with open(os.path.join(data_dir, "train.json"), encoding='utf8') as f:
            train_list = json.loads(f.read())
        return self._create_examples(train_list)

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "test_" + self.test_set + ".json")))
        with open(os.path.join(data_dir, "test_" + self.test_set + ".json"), encoding='utf8') as f:
            test_list = json.loads(f.read())
        return self._create_examples(test_list)

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "info.json")))
        with open(os.path.join(data_dir, "info.json"), encoding='utf8') as f:
            info = json.loads(f.read())
        return info["phones"]

    def _create_examples(self, dcts):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, dct) in enumerate(dcts):
            guid = "%s-%s" % (0, i)
            text = dct['text']
            label = dct['phone']
            position = dct['position']
            char = dct['char']
            examples.append(
                InputExample(guid=guid, text=text, label=label, position=position, char=char))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # Old chinese data does not have '\t'. This is for adaption.
    for i in range(len(label_list)):
        if '\t' not in label_list[i]:
            label_list[i]=label_list[i][0]+'\t'+label_list[i][1:]

    label_map = {label: i for i, label in enumerate(label_list)}
    label_map['_'] = -1
    label_count = [0]*len(label_list)

    # label mask (mask the classes which are not candidates)
    label_word = {label.split('\t')[0]: [] for label in label_list}
    for label in label_list:
        label_word[label.split('\t')[0]].append(label_map[label])
    masks = torch.ones((len(label_list), len(label_list))).byte()
    for i, label in enumerate(label_list):
        masks[i, label_word[label.split('\t')[0]]] = 0
    masks = torch.cat([masks.unsqueeze(0) for _ in range(8)])
    # print(masks.size(),masks)

    # hybrid attention
    attention_mask = torch.ones(12, max_seq_length, max_seq_length, dtype=torch.long)
    # left attention
    attention_mask[:2, :, :] = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.long))
    # right attention
    attention_mask[2:4, :, :] = torch.triu(torch.ones(max_seq_length, max_seq_length, dtype=torch.long))
    # local attention, window size = 3
    attention_mask[4:6, :, :] = torch.triu(
        torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.long), 1), -1)
    attention_mask = torch.cat([attention_mask.unsqueeze(0) for _ in range(8)])

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 100000 == 0:
            print(ex_index)
        tokens_a = example.text

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For the polyphony classification task, the polyphony vector is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        assert len(tokens) == len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # [CLS] + [tokens] + [SEP]
        label_ids = [-1] * max_seq_length

        for i, l in example.label:
            try:
                if '\t' not in l:
                    l=l[0]+'\t'+l[1:]
                assert tokens[i + 1] == l.split('\t')[0]
            except Exception as e:
                print(e)
                print(tokens, i, l)
                continue
            else:
                label_ids[i + 1] = label_map[l]
                label_count[label_map[l]]+=1
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        label_pos = example.position + 1  # First token is [cls]
        assert label_pos < max_seq_length
        # assert tokens[label_pos]==example.label[-1][1][0]

        #polyphony character
        char = example.char



        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label: %s (id = %s)" % (str(example.label), str(label_ids)))
            logger.info("label position: %s" % (str(label_pos)))
            logger.info("character: %s" % (char))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          label_ids=label_ids,
                          label_pos=label_pos,
                          char=char))
    # classification weight, for balancing the classes
    weight = [(max(label_count) / (lc + 100))**1 for lc in label_count]
    print(weight)
    weight = torch.FloatTensor([weight] * 8)
    logit_masks=torch.zeros(len(tokenizer.vocab.items()),len(label_list))
    vocab_list=list(tokenizer.vocab.keys())
    for i in range(len(vocab_list)):
        if vocab_list[i] in label_word:
            logit_masks[i]=masks[0][label_word[vocab_list[i]][0]]
    logit_masks=logit_masks.byte()
    return features, masks, weight, attention_mask, logit_masks


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_list(out, labels, positions):
    outputs = np.argmax(out, axis=2)
    res = []
    # print(out)
    # print(outputs)
    for i, p in enumerate(positions):
        # assert labels[i, p] != -1
        if labels[i, p] == -1:
            print(outputs[i], labels[i], positions[i])
        # print(outputs[i,p],labels[i,p])
        if outputs[i, p] == labels[i, p]:
            res.append(1)
        else:
            res.append(0)
    return res


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_inference",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--test_set",
                        default='story',
                        type=str,
                        #choices=['story', 'news', 'chat', 'train'],
                        help="Choose the test set.")
    parser.add_argument("--no_logit_mask",
                        action='store_true',
                        help="Whether not to use logit mask")
    parser.add_argument("--eval_every_epoch",
                        action='store_true',
                        help="Whether to evaluate for every epoch")
    parser.add_argument("--use_weight",
                        action='store_true',
                        help="Whether to use class-balancing weight")
    parser.add_argument("--hybrid_attention",
                        action='store_true',
                        help="Whether to use hybrid attention")
    parser.add_argument("--state_dir",
                        default="",
                        type=str,
                        help="Where to load state dict instead of using Google pre-trained model")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_inference:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")



    processor = DataProcessor(args.test_set)
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)
    logger.info("num_labels:" + str(num_labels))
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    max_epoch = -1
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        files = os.listdir(args.output_dir)
        for fname in files:
            if re.search(WEIGHTS_NAME, fname) and fname != WEIGHTS_NAME:
                max_epoch = max(max_epoch, int(fname.split('_')[-1]))
        if os.path.exists(os.path.join(args.output_dir, WEIGHTS_NAME + '_' + str(max_epoch))):
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '_' + str(max_epoch))
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME + '_0')
            config = BertConfig(output_config_file)
            model = BertForPolyphonyMulti(config, num_labels=num_labels)
            model.load_state_dict(torch.load(output_model_file))
        else:
            raise ValueError(
                "Output directory ({}) already exists but no model checkpoint was found.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.state_dir and os.path.exists(args.state_dir):
            state_dict=torch.load(args.state_dir)
            if isinstance(state_dict,dict) or isinstance(state_dict,collections.OrderedDict):
                assert 'model' in state_dict
                state_dict=state_dict['model']
            print("Using my own BERT state dict.")
        elif args.state_dir and not os.path.exists(args.state_dir):
            print("Warning: the state dict does not exist, using the Google pre-trained model instead.")
            state_dict=None
        else:
            state_dict=None
        model = BertForPolyphonyMulti.from_pretrained(args.bert_model,
                                                  cache_dir=cache_dir,
                                                  state_dict=state_dict,
                                                  num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    if os.path.exists(os.path.join(args.output_dir, OPTIMIZER_NAME+'_'+str(max_epoch))):
        output_optimizer_file = os.path.join(args.output_dir, OPTIMIZER_NAME+'_'+str(max_epoch))
        optimizer.load_state_dict(torch.load(output_optimizer_file))

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features, masks, weight, hybrid_mask, logit_masks = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        if args.eval_every_epoch:
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_features, masks, weight, hybrid_mask, logit_masks = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)

        if args.no_logit_mask:
            print("Remove logit mask")
            masks = None
        else:
            masks=masks.to(device)
        if not args.use_weight:
            weight=None
        if args.hybrid_attention:
            hybrid_mask = hybrid_mask.to(device)
        else:
            hybrid_mask=None
        print(weight)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_label_poss = torch.tensor([f.label_pos for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_label_poss)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, label_ids, label_poss = batch
                # print(masks.size())
                loss = model(input_ids, input_mask, label_ids, logit_masks=masks, weight=weight, hybrid_mask=hybrid_mask)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if args.eval_every_epoch:
                # evaluate for every epoch
                # save model and load for a single GPU
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '_' + str(ep))
                torch.save(model_to_save.state_dict(), output_model_file)
                output_optimizer_file = os.path.join(args.output_dir, OPTIMIZER_NAME + '_' + str(ep))
                torch.save(optimizer.state_dict(), output_optimizer_file)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME + '_' + str(ep))
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())

                # Load a trained model and config that you have fine-tuned
                config = BertConfig(output_config_file)
                model_eval = BertForPolyphonyMulti(config, num_labels=num_labels)
                model_eval.load_state_dict(torch.load(output_model_file))
                model_eval.to(device)

                if args.hybrid_attention:
                    hybrid_mask = hybrid_mask.to(device)
                else:
                    hybrid_mask=None

                if args.no_logit_mask:
                    print("Remove logit mask")
                    masks = None
                else:
                    masks = masks.to(device)
                chars = [f.char for f in eval_features]
                print(len(set(chars)), sorted(list(set(chars))))
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_label_poss = torch.tensor([f.label_pos for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_label_poss)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model_eval.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                res_list = []
                for input_ids, input_mask, label_ids, label_poss in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    label_ids = label_ids.to(device)
                    label_poss = label_poss.to(device)
                    with torch.no_grad():
                        tmp_eval_loss = model_eval(input_ids, input_mask, label_ids, logit_masks=masks, hybrid_mask=hybrid_mask)
                        logits = model_eval(input_ids, input_mask, label_ids, logit_masks=masks, cal_loss=False, hybrid_mask=hybrid_mask)
                    # print(logits.size())
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    res_list += accuracy_list(logits, label_ids, label_poss)
                    eval_loss += tmp_eval_loss.mean().item()

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                loss = tr_loss / nb_tr_steps if args.do_train else None
                acc = sum(res_list) / len(res_list)
                char_count = {k: [] for k in list(set(chars))}
                for i, c in enumerate(chars):
                    char_count[c].append(res_list[i])
                char_acc = {k: sum(char_count[k]) / len(char_count[k]) for k in char_count}

                result = {'epoch': ep + 1,
                          'eval_loss': eval_loss,
                          'eval_accuracy': acc,
                          'global_step': global_step,
                          'loss': loss}
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                output_eval_file = os.path.join(args.output_dir, "epoch_" + str(ep + 1) + ".txt")
                with open(output_eval_file, 'w') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n' + json.dumps(char_acc, ensure_ascii=False))

                # multi processing
                # if n_gpu > 1:
                #    model = torch.nn.DataParallel(model)

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_optimizer_file = os.path.join(args.output_dir, OPTIMIZER_NAME)
        torch.save(optimizer.state_dict(), output_optimizer_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForPolyphonyMulti(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        # model = BertForPolyphonyMulti.from_pretrained(args.bert_model, num_labels = num_labels)
        pass
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        config = BertConfig(output_config_file)
        model = BertForPolyphonyMulti(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features, masks, weight, hybrid_mask, logit_masks = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)

        if args.hybrid_attention:
            hybrid_mask=hybrid_mask.to(device)
        else:
            hybrid_mask=None
        if args.no_logit_mask:
            print("Remove logit mask")
            masks = None
        else:
            masks = masks.to(device)
        chars = [f.char for f in eval_features]
        print(len(set(chars)), sorted(list(set(chars))))
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_poss = torch.tensor([f.label_pos for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_label_poss)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        res_list = []
        # masks = masks.to(device)
        for input_ids, input_mask, label_ids, label_poss in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)
            label_poss = label_poss.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, input_mask, label_ids, logit_masks=masks, hybrid_mask=hybrid_mask)
                logits = model(input_ids, input_mask, label_ids, logit_masks=masks, cal_loss=False, hybrid_mask=hybrid_mask)
            # print(logits.size())
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            res_list += accuracy_list(logits, label_ids, label_poss)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if args.do_train else None
        acc = sum(res_list) / len(res_list)
        char_count = {k: [] for k in list(set(chars))}
        for i, c in enumerate(chars):
            char_count[c].append(res_list[i])
        char_acc = {k: sum(char_count[k]) / len(char_count[k]) for k in char_count}

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss,
                  'acc': acc}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            for key in sorted(char_acc.keys()):
                logger.info("  %s = %s", key, str(char_acc[key]))
                writer.write("%s = %s\n" % (key, str(char_acc[key])))
        print("mean accuracy", sum(char_acc[c] for c in char_acc) / len(char_acc))
        output_acc_file = os.path.join(args.output_dir, args.test_set + ".json")
        output_reslist_file = os.path.join(args.output_dir, args.test_set + "reslist.json")
        with open(output_acc_file, "w") as f:
            f.write(json.dumps(char_acc, ensure_ascii=False))
        with open(output_reslist_file, "w") as f:
            f.write(json.dumps(res_list, ensure_ascii=False))

    if args.do_inference and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features, masks, weight, hybrid_mask, logit_masks = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logit_masks=logit_masks.to(device)

        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        config = BertConfig(output_config_file)
        model = BertForPolyphonyInference(config, num_labels=num_labels, masks=logit_masks)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

        chars = [f.char for f in eval_features]
        print(len(set(chars)), sorted(list(set(chars))))
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_poss = torch.tensor([f.label_pos for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,all_label_poss)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        res_list = []
        inflist=[]
        # masks = masks.to(device)
        for input_ids, input_mask, label_poss in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids[:,:input_mask.sum(-1)].to(device)
            #input_mask = input_mask[:,:input_mask.sum(-1)].to(device)

            with torch.no_grad():
                logits = model(input_ids)
            # print(logits.size())
            logits = logits.detach().cpu().numpy()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            #output=logits.argmax(-1)
            #print(logits)
            logits=logits[0]
            inf_tmp=[]
            pos=int(label_poss[0])
            for i in range(len(logits)):
                word,phone=label_list[logits[i]].split('\t')
                if tokenizer.vocab[word]==input_ids[0][i]:
                    inf_tmp.append(phone)
                else:
                    inf_tmp.append('N')
            inflist.append('\t'.join(inf_tmp[1:-1]))
        output_reslist_file = os.path.join(args.output_dir, args.test_set + "reslist.json")
        with open(output_reslist_file,'w') as f:
            f.write('\n'.join(inflist))


if __name__ == "__main__":
    main()
