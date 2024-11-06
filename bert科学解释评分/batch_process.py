import torch

from mytokenization import tokenizer

from dataset import *

def collate_fn(data):
    labels = [i[0] for i in data]
    sents = [i[1] for i in data]

    #编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=50,
                                       return_tensors='pt',
                                       return_length=True)
    #input_ids:编码之后的数字
    #attention_mask:补0的位置是0，其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels

train_data[0].type
#collate_fn(train_data)
#collate_fn(test_data)