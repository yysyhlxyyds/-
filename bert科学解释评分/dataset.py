import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

#设置随机数种子
np.random.seed(42)

# 读取CSV文件
def load_data(csv_file):
    return pd.read_csv(csv_file)

# 创建自定义Dataset
class TextDataset(Dataset):
    def __init__(self, dataframe, text_col, label_col):
        self.dataframe = dataframe
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx, self.dataframe.columns.get_loc(self.text_col)]
        label = self.dataframe.iloc[idx, self.dataframe.columns.get_loc(self.label_col)]
        return text, label

# 创建DataLoader
def get_dataloader(dataframe, text_col, label_col, batch_size=32):
    dataset = TextDataset(dataframe, text_col, label_col)
    return DataLoader(dataset, batch_size=batch_size)

csv_file = 'rawdata.csv'
text_col = 'text'
label_col = 'label'

# 读取数据
dataframe = load_data(csv_file)

# 打乱数据集
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# 划分数据集
train_df, test_df = train_test_split(dataframe, test_size=0.25, random_state=42)

train_data = get_dataloader(train_df, text_col, label_col, batch_size=32)
test_data = get_dataloader(test_df, text_col, label_col, batch_size=32)

#打印几个样本
print("\n训练数据加载器中的几个样本:")
for i, (texts, labels) in enumerate(train_data):
    print(f"批次 {i+1}:")
    print("文本样本:", texts)
    print("标签样本:", labels)
    # ins = isinstance(texts, tuple)
    # print(ins)
    if i == 0:  # 只打印第一个批次的样本
        break

