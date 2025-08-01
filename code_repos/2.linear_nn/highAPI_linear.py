'''
Author: Nagisa 2964793117@qq.com
Date: 2025-08-01 17:13:44
LastEditors: Nagisa 2964793117@qq.com
LastEditTime: 2025-08-01 19:54:21
FilePath: \nagisa_notes\code_repos\2.linear_nn\highAPI_linear.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
from torch.utils import data 
import matplotlib.pyplot as plt
from torch import nn

def synthetic_data(w, b, num_examples):
    """生成y = Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b # 一维张量在矩阵乘法中会自动广播，若在右侧则被视为列向量，若在左侧则被视为行向量
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4]) 
true_b = 4.2 
features, labels = synthetic_data(true_w, true_b, 1000) # 生成1000个样本

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器，使用高级API"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 确定神经网络架构
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)