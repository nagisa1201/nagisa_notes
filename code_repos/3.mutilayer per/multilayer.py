import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
# features的维度:(n_train+n_test, 1)生成训练和测试数据集的特征
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features) # 打乱
# ​​构建多项式特征​​（每个样本扩展为20维多项式特征），200x1与1x20逐元素幂，np的广播机制
'''
第0列：所有元素取0次幂 → 都是1
第1列：所有元素取1次幂 → 原值
第2列：所有元素取2次幂 → features平方
...
第19列：所有元素取19次幂
广播是numpy对不同形状数组进行运算的机制，
它会自动扩展较小的数组以匹配较大数组的形状，使得小数组在横向或者纵向上进行重复。

将两个数组的维度大小右对齐，然后比较对应维度上的数值，如果数值相等或其中有一个为1或者为空，则能进行广播运算输出的维度大小为取数值大的数值。否则不能进行数组运算
'''
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
# 对多项式特征进行归一化处理
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w) # 用真实的权重生成标签
labels += np.random.normal(scale=0.1, size=labels.shape) # 添加噪声后的标签

# NumPy ndarray转换为tensor，列表表达式＋多重赋值
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) 
                                          for x in [true_w, features, poly_features, labels]]

# 分割训练集和测试集
train_features, test_features = poly_features[:n_train], poly_features[n_train:] # 取前n_train个样本作为训练集，其余作为测试集
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_features, train_labels.unsqueeze(1))
test_dataset = TensorDataset(test_features, test_labels.unsqueeze(1))

def evaluate_loss(net, data_loader, loss_func):
    """评估给定数据集上模型的损失"""
    total_loss = 0.0
    total_samples = 0
    
    net.eval()
    with torch.no_grad():
        for X, y in data_loader:
            out = net(X)
            l = loss_func(out, y)
            total_loss += l.sum().item()
            total_samples += X.size(0)
    
    return total_loss / total_samples

def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    # 创建数据加载器
    train_dataset = TensorDataset(train_features, train_labels.unsqueeze(1))
    test_dataset = TensorDataset(test_features, test_labels.unsqueeze(1))
    
    batch_size = min(10, train_labels.shape[0])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    # 用于记录损失
    train_losses = []
    test_losses = []
    epochs = []
    
    # 训练循环
    for epoch in range(num_epochs):
        net.train()
        epoch_train_loss = 0.0
        samples_processed = 0
        
        for X, y in train_loader:
            # 前向传播
            out = net(X)
            l = loss(out, y)
            
            # 反向传播
            trainer.zero_grad()
            l.backward()
            trainer.step()
            
            epoch_train_loss += l.item() * X.size(0)
            samples_processed += X.size(0)
        
        # 计算平均训练损失
        train_avg_loss = epoch_train_loss / samples_processed
        test_avg_loss = evaluate_loss(net, test_loader, loss)
        
        # 记录损失
        train_losses.append(train_avg_loss)
        test_losses.append(test_avg_loss)
        epochs.append(epoch + 1)
        
        # 每20个epoch打印一次
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_avg_loss:.6f}, Test Loss: {test_avg_loss:.6f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.ylim(1e-3, 1e2)
    plt.xlim(1, num_epochs)
    plt.legend()
    plt.grid(True)
    plt.title('Training and Test Loss')
    plt.show()
    
    print('Weight:', net[0].weight.data.numpy())

# 运行训练
train(train_features, test_features, train_labels, test_labels)