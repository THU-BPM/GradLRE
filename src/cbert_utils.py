import random
import logging
from transformers import BertTokenizer

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# loading the pre-trained bert word tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels):
        self.init_ids = init_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels


def get_entity_pos(sentence_token_list):
    pos1 = []
    pos2 = []
    e1_flag = 0
    e2_flag = 0
    for i in range(len(sentence_token_list)):
        if sentence_token_list[i] == ">" and sentence_token_list[i-1] == "##1" and sentence_token_list[i-3] == "<":
            e1_flag = 1
            if e1_flag == 1 and e2_flag == 0:
                pos1.append(i+1-4)
            else:
                pos1.append(i+1-13)
        if sentence_token_list[i] == "<" and i+3 < len(sentence_token_list) and sentence_token_list[i+1] == "/" and sentence_token_list[i+3] == "##1":
            if e1_flag == 1 and e2_flag == 0:
                pos1.append(i-1-4)
            else:
                pos1.append(i-1-13)
        if sentence_token_list[i] == ">" and sentence_token_list[i-1] == "##2" and sentence_token_list[i-3] == "<":
            e2_flag = 1
            if e1_flag == 1 and e2_flag == 1:
                pos2.append(i+1-13)
            else:
                pos2.append(i+1-4)
        if sentence_token_list[i] == "<" and i+3 < len(sentence_token_list) and sentence_token_list[i+1] == "/" and sentence_token_list[i+3] == "##2":
            if e1_flag == 1 and e2_flag == 1:
                pos2.append(i-1-13)
            else:
                pos2.append(i-1-4)
    return pos1, pos2


def create_masked_lm_predictions(tokens, pos1, pos2, masked_lm_probs, masked_lm_labels, 
                                 max_predictions_per_seq, rng):
    """Creates the predictions for the masked LM objective."""

    #vocab_words = list(tokenizer.vocab.keys())
    mask_start_pos = tokens.index('[SEP]')
    
    cand_indexes = []
    for i in range(mask_start_pos, len(tokens)):
        if tokens[i] == "[CLS]" or tokens[i] == "[SEP]" or (len(pos1) == 2 and i in range(pos1[0],pos1[1]+1)) or (len(pos2) == 2 and i in range(pos2[0],pos2[1]+1)):
            continue
        cand_indexes.append(i)
    
    rng.shuffle(cand_indexes)
    len_cand = len(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq, 
                         max(1, int(round(len(tokens) * masked_lm_probs))))
    
    masked_lm_positions = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lm_positions) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        ## 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            ## 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            ## 10% of the time, replace with random word
            else:
                masked_token = tokens[cand_indexes[rng.randint(0, len_cand - 1)]]
                
        masked_lm_labels[index] = tokenizer.convert_tokens_to_ids([tokens[index]])[0]
        output_tokens[index] = masked_token
        masked_lm_positions.append(index)
    return output_tokens, masked_lm_positions, masked_lm_labels


def extract_features(tokens, max_seq_length):
    """extract features from tokens"""

    if len(tokens) > max_seq_length:
        tokens = tokens[0:max_seq_length]

    ## construct init_ids for each example
    init_ids = tokenizer.convert_tokens_to_ids(tokens)

    ## construct input_ids for each example, we replace the word_id using 
    ## the ids of masked words (mask words based on original sentence)
    masked_lm_probs = 0.15
    max_predictions_per_seq = 20
    rng = random.Random(12345)
    original_masked_lm_labels = [-100] * max_seq_length

    pos1, pos2 = get_entity_pos(tokens)
    (output_tokens, masked_lm_positions, 
    masked_lm_labels) = create_masked_lm_predictions(
            tokens, pos1, pos2, masked_lm_probs, original_masked_lm_labels, max_predictions_per_seq, rng)
    input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        init_ids.append(0)
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    assert len(init_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return tokens, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels


def convert_examples_to_features(examples, max_seq_length):
    """Loads a data file into a list of 'InputBatch's."""
    features = []
    for (ex_index, example) in enumerate(examples):
        # The convention in BERT is:
        # tokens:   [CLS] is this jack ##son ##ville ? [SEP]
        # type_ids: 0     0  0    0    0     0       0 0    
        tokens = tokenizer.tokenize(example)
        tokens, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels = \
            extract_features(tokens, max_seq_length)

        """consturct features"""
        features.append(
            InputFeature(
                init_ids=init_ids,        
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                masked_lm_labels=masked_lm_labels))

        """print examples"""
        if ex_index < 5:
            logger.info("[mlm_tune] *** Example ***")
            logger.info("[mlm_tune] tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("[mlm_tune] init_ids: %s" % " ".join([str(x) for x in init_ids]))
            logger.info("[mlm_tune] input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("[mlm_tune] input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("[mlm_tune] segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("[mlm_tune] masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))
    return features


def construct_train_dataloader(train_examples, max_seq_length, train_batch_size, num_train_epochs, device):
    """construct dataloader for training data"""

    num_train_steps = None
    train_features = convert_examples_to_features(
        train_examples, max_seq_length)
    num_train_steps = int(len(train_features) / train_batch_size * num_train_epochs)
    
    all_init_ids = torch.tensor([f.init_ids for f in train_features], dtype=torch.long, device=device)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long, device=device)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long, device=device)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long, device=device)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in train_features], dtype=torch.long, device=device)
    
    tensor_dataset = TensorDataset(all_init_ids, all_input_ids, all_input_mask, 
        all_segment_ids, all_masked_lm_labels)
    train_sampler = RandomSampler(tensor_dataset)
    train_dataloader = DataLoader(tensor_dataset, sampler=train_sampler, batch_size=train_batch_size)
    return train_features, num_train_steps, train_dataloader