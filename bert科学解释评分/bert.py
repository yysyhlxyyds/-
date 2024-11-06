"""
安装所需要的包
pip install transformers torch datasets
"""

"""
使用HuggingFace 的datasets库中的imdb数据集
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import load_dataset
dataset = load_dataset(path='/rawdata.csv')
print(dataset)
"""
标记化数据，使用Transformers中的BertTokenizer对数据进行标记
"""
from transformers import BertTokenizer

#文本长度是默认的，可以自行设置，max_length=xxx
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#将文本转换为模型可以处理的格式，包括截断和填充
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

#使用map函数把数据处理应用到每个元素上，这里的batched=True参数表示tokenize_function将在数据集的批次上执行，而不是单独对每个样本执行。
tokenized_datasets = dataset.map(tokenize_function, batched=True)
                                                           
"""
划分训练集和验证集
"""
#按照8:2，把训练集进一步划分为训练集和测试集
train_testvalid = tokenized_datasets['train'].train_test_split(test_size=0.2)
train_dataset = train_testvalid['train']
valid_dataset = train_testvalid['test']

"""
为训练集和验证集创建数据加载器
"""
from torch.utils.data import DataLoader
#训练数据随机打乱，增加模型的泛化能力
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)

"""
加载模型，bert-base-uncased是BERT的基本版本，将文本的所有字母都转换为小写有助于减少模型的词汇大小
12个Transformer-Encoder，隐藏层维度768，总共有110M个参数
"""
from transformers import BertForSequenceClassification, AdamW
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


"""
定义模型微调过程的参数
"""
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    #微调后模型的存放位置
    output_dir='./results',
    #验证集在每个训练周期（epoch）结束后用于评估模型的性能，模型通过与验证集进行交互来调整自己的参数
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

"""
分类模型评价指标，准确度、精确度、召回率和 F1 分数等
"""
metrics = trainer.evaluate()

"""
使用微调后的模型进行预测
"""
predictions = trainer.predict(valid_dataset)
