from train import *


# 确保模型处于评估模式
classifier.eval()

# 切换到评估模式，关闭梯度计算
torch.no_grad()

# 初始化性能指标
total_accuracy = 0
total_loss = 0

# 迭代测试数据集
for i, batch in enumerate(test_data_loader):
    input_ids = batch['input_ids_test']
    attention_mask = batch['attention_mask_test']
    labels = batch['labels']

    # 前向传播
    out = classifier(input_ids=input_ids, attention_mask=attention_mask)

    # 计算损失
    loss = criterion(out, labels)
    total_loss += loss.item()

    # 计算准确率
    predictions = out.argmax(dim=1)
    total_accuracy += (predictions == labels).sum().item()

# 计算总准确率和损失
total_accuracy = total_accuracy / len(test_data)
average_loss = total_loss / len(test_data)

print(f"测试集上的准确率: {total_accuracy:.4f}")
print(f"测试集上的平均损失: {average_loss:.4f}")