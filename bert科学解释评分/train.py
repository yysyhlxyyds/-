from transformers import AdamW
from classifier import *
from mytokenization import *
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW as TorchAdamW

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

#直接使用mytokenization中的input_ids、attention_mask、label创建新的数据集
train_data = CustomDataset(encodings={
    'input_ids_train': input_ids_train, 'attention_mask_train': attention_mask_train}, labels=label_train)

test_data = CustomDataset(encodings={
    'input_ids_test': input_ids_test, 'attention_mask_test': attention_mask_test}, labels=label_test)

# 创建 DataLoader
train_data_loader = DataLoader(train_data, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#训练
optimizer = TorchAdamW(classifier.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# 训练循环
classifier.train()
for i, batch in enumerate(train_data_loader):
    input_ids = batch['input_ids_train']
    attention_mask = batch['attention_mask_train']
    labels = batch['labels']

    out = classifier(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        print(i, loss.item(), accuracy)