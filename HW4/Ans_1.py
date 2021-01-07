import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Parameters
input_size = 28 * 28 
hidden_size = 100
num_classes = 10
num_epoches = 5
batch_size = 64
learning_rate = 1e-3

# MNIST dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = dsets.MNIST(root = '../../data_sets/mnist',     #选择数据的根目录
                           train = True,                        #选择训练集
                           transform = transforms.ToTensor(),   #转换成tensor变量
                           download = True)                    #不从网络上download图片
test_dataset = dsets.MNIST(root = '../../data_sets/mnist',      #选择数据的根目录
                           train = False,                       #选择训练集
                           transform = transforms.ToTensor(),   #转换成tensor变量
                           download = True)                    #不从网络上download图片


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                           batch_size = batch_size, 
                                           shuffle = True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)


'''
# 全连接神经网络 FCN
class Net(nn.Module):
    def __init__(self, input_num, hidden_size, out_put):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(input_num, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, out_put)
        )
    def forward(self, input):
        return self.net(input)
'''
# 卷积神经网络 CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(320, 10)


    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*320
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        return x



#model = Net(input_size, hidden_size, num_classes)

model = Net()
model = model.to(device)

# optimization
Loss = nn.CrossEntropyLoss()                               # 使用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_record = np.array([])

for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for i, data in enumerate(train_loader): #利用enumerate取出一个可迭代对象的内容
        inputs = data[0].to(device)
        labels = data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = Loss(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            loss_record = np.append(loss_record, loss.data)
        if i % 100 == 0:
            print('current loss = %.5f' % loss.data)

# plot
Y1 = loss_record
X1 = np.linspace(1, len(Y1), len(Y1))
#plt.plot(X1,Y1,  linewidth=1.0, linestyle="-",label="Full Connection Network")
plt.plot(X1,Y1,  linewidth=1.0, linestyle="-",label="Convolutional Neural Network")

plt.legend()
plt.xlabel('Iteration Times')
plt.ylabel('Cross Entropy Loss')
plt.savefig('./Ans1_result.jpg')
plt.show()

# prediction
total = 0
correct = 0
for data in test_loader:
    inputs = data[0].to(device)
    labels = data[1].to(device)
    outputs = model(inputs)
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()

print('Accuracy = %.2f' % (100.0 * correct / total))


#if __name__ == "__main__":