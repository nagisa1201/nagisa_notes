import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):
    """生成y = Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b # 一维张量在矩阵乘法中会自动广播，若在右侧则被视为列向量，若在左侧则被视为行向量
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features) # 获取样本总数
    indices = list(range(num_examples)) # 创建一个索引列表，包含所有样本的索引
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices) 
    for i in range(0, num_examples, batch_size): # 以batch_size为步长遍历索引
        batch_indices = torch.tensor( # torch.tensor()将列表转换为张量
            indices[i: min(i + batch_size, num_examples)]) # 获取当前批次的索引
        # min(i + batch_size, num_examples) 确保不会超出样本总数
        yield features[batch_indices], labels[batch_indices] # yield语句返回当前批次的特征和标签,可以多次返回，调用时会继续从上次返回的地方开始执行

true_w = torch.tensor([2, -3.4]) # 真实权重，为一维张量
true_b = 4.2 # 真实偏置
features, labels = synthetic_data(true_w, true_b, 1000) # 生成1000个样本
# 此处X为(1000, 2)的二维张量，在y一步计算后为1000x2矩阵乘以2x1矩阵得到，y为(1000, 1)的二维张量
# b的偏置会加到矩阵的每一个元素上
print('features:', features[0],'\nlabel:', labels.shape, '\nlabel:', labels[0]) # 打印第一个样本的特征和标签
# 可视化
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=5) # :表示选取所有行，1表示选取第二列，若改为0:1则选取第一列，:是默认写法，全写应写为0:2
# s=1表示点的大小，前面两个表示x和y轴的值
plt.xlabel("Feature (x2)")
plt.ylabel("Label (y)")
plt.title("Synthetic Data: x2 vs y")
plt.show()


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break   

## 此代码文件写到读取数据集就不写了，因为觉得线性神经网络没有必要了解代码实现