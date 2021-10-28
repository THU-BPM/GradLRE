import json

def Read_SemEval_data(CATEGORY):
    sentence=[]
    label_name=[]
    interval = ' '
    for line in open('data/SemEval/%s/%s.txt'%(CATEGORY,CATEGORY)):
        token=[]
        line = line.split("	")
        line[1] = line[1].strip('\n')
        token.insert(0,'[CLS]')
        token.insert(1,line[1])
        token.insert(2, '[SEP]')
        token = interval.join(token)
        sentence.append(token)
    for line in open('data/SemEval/%s/%s_result_full.txt'%(CATEGORY,CATEGORY)):
        line = line.split("	")
        line[1] = line[1].strip('\n')
        label_name.append(line[1])
    return sentence,label_name

# def SemEval_relation2id():
#     SemEval_relation2id=[]
#     for line in open('data/SemEval/train/train_result_full.txt'):
#         line=line.split("	")
#         if line[1] in SemEval_relation2id:
#             continue
#         SemEval_relation2id.append(line[1])

def realtion2id(dataset_name,label_name):
    label_id=[]
    with open('data/%s/relation2id.json'% dataset_name,'r') as f:
        data = json.load(f)
    for i in range(len(label_name)):
        id=data[label_name[i]]
        label_id.append(id)
    return label_id



# def SemEval_INDEX():
#     SemEval_INDEX=[]
#     for i in range(len(SemEval_train_sentence)):
#         SemEval_INDEX.append(SemEval_train_sentence[i].find("<e1>"))
#         SemEval_INDEX.append(SemEval_train_sentence[i].find("<e2>"))
#         SemEval_train_sentence_index.append(SemEval_INDEX)
#     print(SemEval_train_sentence_index[0])

def Read_TACRED_data(CATEGORY):
    sentence = []
    label_name = []
    interval=' '
    with open('data/tacred/data/json/%s.json' % CATEGORY, 'r') as f:
        line=f.readline()
        data=json.loads(line)
    for i in range(len(data)):
        tokens=data[i]["token"]
        subj_start=data[i]["subj_start"]
        subj_end=data[i]["subj_end"]
        obj_start=data[i]["obj_start"]
        obj_end=data[i]["obj_end"]
        relation=data[i]["relation"]
        tokens.insert(subj_start, '<e1>')
        tokens.insert(subj_end + 2, '</e1>')
        tokens.insert(obj_start + 2, '<e2>')
        tokens.insert(obj_end + 4, '</e2>')
        tokens.insert(0,'[CLS]')
        tokens.insert(-1, '[SEP]')
        tokens=interval.join(tokens)
        sentence.append(tokens)
        label_name.append(relation)
    return sentence,label_name


if __name__ == '__main__':
    SemEval_train_sentence=[]
    SemEval_train_label_name=[]
    SemEval_train_label_id=[]
    SemEval_train_sentence,SemEval_train_label_name=Read_SemEval_data('train')
    SemEval_train_label_id=realtion2id('SemEval',SemEval_train_label_name)
    SemEval_test_sentence=[]
    SemEval_test_label_name=[]
    SemEval_test_label_id=[]
    SemEval_test_sentence,SemEval_test_label_name=Read_SemEval_data('test')
    SemEval_test_label_id=realtion2id('SemEval',SemEval_test_label_name)
    SemEval_dev_sentence=[]
    SemEval_dev_label_name=[]
    SemEval_dev_label_id=[]
    SemEval_dev_sentence=SemEval_train_sentence[0:800]
    SemEval_dev_label_name = SemEval_train_label_name[0:800]
    SemEval_dev_label_id = SemEval_train_label_id[0:800]
    SemEval_train_sentence=SemEval_train_sentence[800:]
    SemEval_train_label_name=SemEval_train_label_name[800:]
    SemEval_train_label_id=SemEval_train_label_id[800:]
    # print(len(SemEval_dev_label_id))
    # print(len(SemEval_dev_sentence))
    # print(len(SemEval_train_label_id))
    # print(len(SemEval_train_sentence))
    # print(len(SemEval_test_label_id))
    # print(len(SemEval_test_sentence))
    # with open('data/tacred/train_sentence.json','r') as f:
    #     data = json.load(f)
    # with open('data/tacred/train_label_id.json','r') as f:
    #     data_3 = json.load(f)
    # with open('data/tacred/dev_sentence.json','r') as f:
    #     data_1 = json.load(f)
    # with open('data/tacred/test_sentence.json','r') as f:
    #     data_2 = json.load(f)
    # print(len(data))
    # print(len(data_1))
    # print(len(data_2))
    # print(len(data_3))
    with open('data/SemEval/train_sentence.json', 'w') as fw:
        json.dump(SemEval_train_sentence, fw)
    with open('data/SemEval/train_label_id.json', 'w') as fw:
        json.dump(SemEval_train_label_id, fw)
    with open('data/SemEval/dev_sentence.json', 'w') as fw:
        json.dump(SemEval_dev_sentence, fw)
    with open('data/SemEval/dev_label_id.json', 'w') as fw:
        json.dump(SemEval_dev_label_id, fw)
    with open('data/SemEval/test_sentence.json', 'w') as fw:
        json.dump(SemEval_test_sentence, fw)
    with open('data/SemEval/test_label_id.json', 'w') as fw:
        json.dump(SemEval_test_label_id, fw)
    # TACRED_test_sentence=[]
    # TACRED_test_label_name=[]
    # TACRED_test_label_id=[]
    # TACRED_test_sentence,TACRED_test_label_name=Read_TACRED_data("test")
    # TACRED_test_label_id = realtion2id('tacred',TACRED_test_label_name)
    # TACRED_train_sentence=[]
    # TACRED_train_label_name=[]
    # TACRED_train_label_id=[]
    # TACRED_train_sentence,TACRED_train_label_name=Read_TACRED_data("train")
    # TACRED_train_label_id = realtion2id('tacred',TACRED_train_label_name)
    # TACRED_dev_sentence=[]
    # TACRED_dev_label_name=[]
    # TACRED_dev_label_id=[]
    # TACRED_dev_sentence,TACRED_dev_label_name=Read_TACRED_data("dev")
    # TACRED_dev_label_id = realtion2id('tacred',TACRED_dev_label_name)
    # with open('data/tacred/test_sentence.json', 'w') as fw:
    #     json.dump(TACRED_test_sentence, fw)
    # with open('data/tacred/test_label_id.json', 'w') as fw:
    #     json.dump(TACRED_test_label_id, fw)
    # with open('data/tacred/dev_sentence.json', 'w') as fw:
    #     json.dump(TACRED_dev_sentence, fw)
    # with open('data/tacred/dev_label_id.json', 'w') as fw:
    #     json.dump(TACRED_dev_label_id, fw)
    # with open('data/tacred/train_sentence.json', 'w') as fw:
    #     json.dump(TACRED_train_sentence, fw)
    # with open('data/tacred/train_label_id.json', 'w') as fw:
    #     json.dump(TACRED_train_label_id, fw)













