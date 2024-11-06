from transformers import BertTokenizer
from dataset import *

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', force_download=True)

# 列表保存每个批次的输出
all_input_ids_train = []
all_attention_masks_train = []
all_label_train = []

all_input_ids_test = []
all_attention_masks_test = []
all_label_test = []

#对训练数据tokenization
for text, label in train_data:
    text = [str(text) for text in text]
    train_input = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=text,
        truncation=True,
        padding='max_length',
        max_length=50,
        return_tensors='pt',
        return_length=True
    )

    input_ids = train_input['input_ids']  #input_ids:编码之后的数字
    attention_mask = train_input['attention_mask']  #attention_mask:补0的位置是0，其他位置是1

    #检查一下
    # print("Input IDs:")
    # print(input_ids)
    #
    # print("\nAttention Mask:")
    # print(attention_mask)

    # 保存到列表
    all_input_ids_train.append(input_ids)
    all_attention_masks_train.append(attention_mask)
    all_label_train.append(label)


#对测试数据tokenization
for text, label in test_data:
    text = [str(text) for text in text]
    test_input = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=text,
        truncation=True,
        padding='max_length',
        max_length=50,
        return_tensors='pt',
        return_length=True
    )

    input_ids = test_input['input_ids']
    attention_mask = test_input['attention_mask']

    # 保存到列表
    all_input_ids_test.append(input_ids)
    all_attention_masks_test.append(attention_mask)
    all_label_test.append(label)

# 将列表中的张量合并成一个张量
input_ids_train = torch.cat(all_input_ids_train, dim=0)
attention_mask_train = torch.cat(all_attention_masks_train, dim=0)
label_train = torch.cat(all_label_train, dim=0)

input_ids_test = torch.cat(all_input_ids_test, dim=0)
attention_mask_test = torch.cat(all_attention_masks_test, dim=0)
label_test = torch.cat(all_label_test, dim=0)

print(input_ids_train)