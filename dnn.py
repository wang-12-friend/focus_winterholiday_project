import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

train_Data = datasets.MNIST(
    root = 'E:/pycharm/py_project/MNIST_data',
    train = True,            # 是train集
    download = True,       # 如果该路径没有该数据集，就下载
    transform = transform    # 数据集转换参数
)
test_Data = datasets.MNIST(
    root = 'E:/pycharm/py_project/MNIST_data',     # 下载路径
    train = False,           # 是test集
    download = True,       # 如果该路径没有该数据集，就下载
    transform = transform    # 数据集转换参数
)

train_loader = DataLoader(train_Data, shuffle=True, batch_size=64)
test_loader = DataLoader(test_Data, shuffle=False, batch_size=64)

class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64,10)
        )
    def forward(self, x):
        y = self.net(x)
        return y
model = DNN().to('cuda:0')
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01    # 设置学习率
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate,
    momentum = 0.5
)

epochs = 5
losses = []  # 记录损失函数变化的列表

for epoch in range(epochs):
    for (x, y) in train_loader:  # 获取小批次的x与y
        x, y = x.to('cuda:0'), y.to('cuda:0')
        Pred = model(x)  # 一次前向传播（小批量）
        loss = loss_fn(Pred, y)  # 计算损失函数
        losses.append(loss.item())  # 记录损失函数的变化
        optimizer.zero_grad()  # 清理上一轮滞留的梯度
        loss.backward()  # 一次反向传播
        optimizer.step()  # 优化内部参数

correct = 0
total = 0
with torch.no_grad():
    for (x, y) in test_loader:
        x, y = x.to('cuda:0'), y.to('cuda:0')
        Pred = model(x)
        predicted = torch.max(Pred.data, dim=1)[1]
        correct += torch.sum( (predicted == y) )
        total += y.size(0)
print(f'测试集精准度: {100*correct/total} %')

Fig = plt.figure()
plt.plot(range(len(losses)), losses)
plt.show()