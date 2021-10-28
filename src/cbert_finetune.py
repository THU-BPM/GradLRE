import os
import shutil
import logging
import argparse
import random
import json
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW
import cbert_utils

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""cuda or cpu"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_space(sentence_train, dataset_name):
    if dataset_name == "tacred":
        return sentence_train
    else:
        sentence_train_new = []
        for sentence in sentence_train:
            sentence_train_new.append(sentence.replace("<e1>","<e1> ").replace("</e1>"," </e1>").replace("<e2>","<e2> ").replace("</e2>"," </e2>"))
        return sentence_train_new


def train_mlm(train_examples):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset.")
    parser.add_argument("--save_model_dir", default="cbert_model", type=str,
                        help="The cache dir for saved model.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path of pretrained bert model.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequence longer than this will be truncated, and sequences shorter \n"
                             "than this wille be padded.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--save_every_epoch", default=True, action='store_true')
    parser.add_argument('--use_aug', type=bool, default=False, help='whether to use data aug')

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    ## leveraging lastest bert module in Transformers to load pre-trained model (weights)
    masked_model = BertForMaskedLM.from_pretrained(args.bert_model)

    train_features, num_train_steps, train_dataloader = \
        cbert_utils.construct_train_dataloader(train_examples, args.max_seq_length, 
        args.train_batch_size, args.num_train_epochs, device)

    ## if you have a GPU, put everything on cuda
    masked_model.cuda()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    ## in Transformers, optimizer and schedules are splitted and instantiated like this:
    param_optimizer = list(masked_model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grounded_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grounded_parameters, lr=args.learning_rate, correct_bias=False)
    masked_model.train()

    os.makedirs(args.save_model_dir, exist_ok=True)

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        avg_loss = 0.

        for step, batch in enumerate(train_dataloader):
            # print(step)
            batch = tuple(t.cuda() for t in batch)
            _, input_ids, input_mask, segment_ids, masked_ids = batch
            """train generator at each batch"""
            optimizer.zero_grad()
            outputs = masked_model(input_ids, input_mask, segment_ids, labels=masked_ids)
            loss = outputs[0]
            # print(loss)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            if (step + 1) % 50 == 0:
                print("avg_loss: {}".format(avg_loss / 50))
                avg_loss = 0

    return masked_model
        
