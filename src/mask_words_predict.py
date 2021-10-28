import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import json
import os
import copy
import random
random.seed(1)

CUDA = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
device = torch.device("cuda")

# DATASET = 'SemEval'  # tacred,SemEval
lower = 1
upper = 4
lens = list(range(lower, upper + 1))
geometric_p = 0.2
len_distrib = [geometric_p * (1 - geometric_p)**(i - lower) for i in range(lower, upper + 1)] if geometric_p >= 0 else None
len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
print(len_distrib, lens)

#stop words list
stop_words = ['a', 'the', ',', '.', '(', ')', '[CLS]', '[SEP]']

# Loading pre-trained bert tokenizer model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# loading pre-trained bert model
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
model.to(device)

# loading pre-trained masked language model
# masked_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# masked_model.eval()
# masked_model.to(device)

def dist(x, y):
    """calculate the cos distance between two vectors
    """
    return np.sqrt(((x - y)**2).sum())

def cos_dist(x, y):
    return np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))

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

def get_random_mask_pos(sentence_token_list, pos1, pos2, p):
    # mask_start = sentence_token_list.index('[SEP]')
    mask_start = 0
    text_index_list = []
    for i in range(mask_start, len(sentence_token_list)):
        if sentence_token_list[i] not in stop_words and i not in range(pos1[0],pos1[1]+1) and i not in range(pos2[0],pos2[1]+1):
            text_index_list.append(i)
    mask_num = max(1, int(p * len(sentence_token_list)))  # whether text_index_list or sentence_token_list ??
    mask_num = min(mask_num, len(text_index_list))
    # print("#####")
    # print(mask_num)
    # print("#####")
    # print(text_index_list)
    random_mask_pos = set()
    while len(random_mask_pos) < mask_num:
        span_len = np.random.choice(lens, p=len_distrib)
        # print("----------------")
        # print(span_len)
        # print("----------------")
        span_len = min(span_len, len(text_index_list))
        start = np.random.choice(len(text_index_list)-(span_len-1))
        while sentence_token_list[text_index_list[start]].startswith('##'):
            start = np.random.choice(len(text_index_list)-(span_len-1))
        for i in range(start, start+span_len):
            if len(random_mask_pos) >= mask_num:
                break
            random_mask_pos.add(text_index_list[i])
    # random_mask_pos = random.sample(text_index_list, mask_num)
    return random_mask_pos

def add_point(sentence):
    words_list = sentence.split(" ")
    words_list.insert(len(words_list)-1,".")
    return " ".join(words_list)

def add_entity_flag(tokenized_text_without_mask, pos1, pos2):
    e1_start = ['<e1>']
    e1_end = ['</e1>']
    e2_start = ['<e2>']
    e2_end = ['</e2>']
    if pos1[0] < pos2[0]:
        tokenized_text_without_mask = tokenized_text_without_mask[:pos1[0]] + e1_start + tokenized_text_without_mask[pos1[0]:]
        tokenized_text_without_mask = tokenized_text_without_mask[:pos1[1]+1+1] + e1_end + tokenized_text_without_mask[pos1[1]+1+1:]
        tokenized_text_without_mask = tokenized_text_without_mask[:pos2[0]+2] + e2_start + tokenized_text_without_mask[pos2[0]+2:]
        tokenized_text_without_mask = tokenized_text_without_mask[:pos2[1]+1+3] + e2_end + tokenized_text_without_mask[pos2[1]+1+3:]
    else:
        tokenized_text_without_mask = tokenized_text_without_mask[:pos2[0]] + e2_start + tokenized_text_without_mask[pos2[0]:]
        tokenized_text_without_mask = tokenized_text_without_mask[:pos2[1]+1+1] + e2_end + tokenized_text_without_mask[pos2[1]+1+1:]
        tokenized_text_without_mask = tokenized_text_without_mask[:pos1[0]+2] + e1_start + tokenized_text_without_mask[pos1[0]+2:]
        tokenized_text_without_mask = tokenized_text_without_mask[:pos1[1]+1+3] + e1_end + tokenized_text_without_mask[pos1[1]+1+3:]
    return tokenized_text_without_mask

def get_entity_dist(tokenized_text_without_mask, pos1, pos2):
    # Mask Entity1
    for masked_index in range(pos1[0], pos1[1] + 1):
        tokenized_text_without_mask[masked_index] = '[MASK]'
        # print(tokenized_text_without_mask)
    # Convert tokens into vocabulary indexes
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_without_mask)
    segments_ids = [0] * len(tokenized_text_without_mask)
    # Convert the input into Pytorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # put data to GPU
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)

    # get emb_vector for entity1
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        sequence_output = outputs[0]
        # print(sequence_output.shape)       # torch.Size([1, 17, 768])

    entity1_emb = sequence_output[0, pos1[0]:pos1[1]+1].mean(0)

    # Mask Entity2
    for masked_index in range(pos2[0], pos2[1] + 1):
        tokenized_text_without_mask[masked_index] = '[MASK]'
    # print(tokenized_text_without_mask)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_without_mask)
    segments_ids = [0] * len(tokenized_text_without_mask)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)

    # get emb_vector for entity1
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        sequence_output = outputs[0]
        # print(sequence_output.shape)       # torch.Size([1, 17, 768])

    entity1_emb_2 = sequence_output[0, pos1[0]:pos1[1]+1].mean(0)
    return cos_dist(entity1_emb.cpu(), entity1_emb_2.cpu())


def get_cls_emb(tokenized_text_without_mask):
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_without_mask)
    segments_ids = [0] * len(tokenized_text_without_mask)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    # put data to cuda
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    
    # get emb_vector for entity1
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        sequence_output = outputs[0]
        # print(sequence_output.shape)       # torch.Size([1, 17, 768])
    return sequence_output[0, 0]

# import logging
# logging.basicConfig(level=logging.INFO)

def get_enhance_result(text, masked_model):
    masked_model.eval()
    masked_model.to(device)
    text = add_point(text) 
    tokenized_text_temp = tokenizer.tokenize(text)
    pos1,pos2 = get_entity_pos(tokenized_text_temp)

    text = text.replace("<e1>","").replace("</e1>","").replace("<e2>","").replace("</e2>","")
    text_enhance_list = {}

    tokenized_text = tokenizer.tokenize(text)
    # text_enhance_list.append(copy.deepcopy(tokenized_text))
    text_enhance_list["before mask"] = copy.deepcopy(tokenized_text)

    cls_emb1 = get_cls_emb(copy.deepcopy(tokenized_text))
    entity_effect1 = get_entity_dist(copy.deepcopy(tokenized_text), pos1, pos2)
    text_enhance_list["entity_effect1"] = entity_effect1

    random_mask_pos = get_random_mask_pos(tokenized_text, pos1, pos2, 0.15)

    # print(random_mask_pos)
    # Mask text tokens except entity
    for masked_index in random_mask_pos:
        tokenized_text[masked_index] = '[MASK]'
    # print(tokenized_text)
    # text_enhance_list.append(copy.deepcopy(tokenized_text))
    text_enhance_list["after mask"] = copy.deepcopy(tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)

    # predict all the masked token
    with torch.no_grad():
        outputs = masked_model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]
        # print(predictions.shape)       # torch.Size([1, 17, 30522])

    for i in random_mask_pos:
        predicted_index = torch.argmax(predictions[0, i]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        tokenized_text[i] = predicted_token
        # print(tokenized_text)

    # text_enhance_list.append(copy.deepcopy(tokenized_text))
    text_enhance_list["mask predict"] = copy.deepcopy(tokenized_text)
    text_enhance_list["mask predict add flag"] = add_entity_flag(copy.deepcopy(tokenized_text), pos1, pos2)
        
    cls_emb2 = get_cls_emb(copy.deepcopy(tokenized_text))
    entity_effect2 = get_entity_dist(copy.deepcopy(tokenized_text), pos1, pos2)
    text_enhance_list["entity_effect2"] = entity_effect2
    text_enhance_list["cls_dist"] = cos_dist(cls_emb1.cpu(), cls_emb2.cpu())
    # return text_enhance_list
    return text_enhance_list["mask predict add flag"], text_enhance_list["cls_dist"]
