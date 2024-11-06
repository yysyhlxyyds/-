import torch.utils.data
from batch_process import *
from dataset import *


loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                     batch_size=32,
                                     collate_fn=collate_fn,
                                     shuffle=True
                                     )
loader_test = torch.utils.data.DataLoader(dataset=test_data,
                                     batch_size=32,
                                     collate_fn=collate_fn,
                                     shuffle=True
                                     )

for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_train):
    break

print(len(loader_train))
input_ids.shape, attention_mask.shape, token_type_ids.shape, labels