from transformers import BertModel
from mytokenization import *

#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')

#不训练，不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False) #不做微调

#模型试算
out = pretrained(input_ids=input_ids_train,
                 attention_mask=attention_mask_train
                 )
print(out.last_hidden_state.shape)