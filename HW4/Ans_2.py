import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Parameters
num_epoches = 5
batch_size = 64
learning_rate = 1e-3
# MNIST dataset
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


transform_train = transforms.Compose([
# 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=1.5, contrast=1.4),
# 按0.5的概率水平翻转图片
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

 # 定义数据集

train_dataset = dsets.CIFAR10(root = '../../data_sets/CIFAR10',     #选择数据的根目录
                           train = True,                        #选择训练集
                           transform = transform_train,   #转换成tensor变量
                           download = True)                    #不从网络上download图片
test_dataset = dsets.CIFAR10(root = '../../data_sets/CIFAR10',      #选择数据的根目录
                           train = False,                       #选择训练集
                           transform = transform_test,   #转换成tensor变量
                           download = True)                    #不从网络上download图片

'''
train_dataset = dsets.CIFAR10(root = '../../data_sets/CIFAR10',     #选择数据的根目录
                           train = True,                        #选择训练集
                           transform = transforms.ToTensor(),   #转换成tensor变量
                           download = True)                    #不从网络上download图片
test_dataset = dsets.CIFAR10(root = '../../data_sets/CIFAR10',      #选择数据的根目录
                           train = False,                       #选择训练集
                           transform = transforms.ToTensor(),   #转换成tensor变量
                           download = True)                    #不从网络上download图片
'''

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                           batch_size = batch_size, 
                                           shuffle = True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle=True)
                                          
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_iter = iter(test_loader)
images, labels = next(data_iter)
idx = 15
image = images[idx].numpy()
image = np.transpose(image, (1,2,0))
plt.imshow(image)
classes[labels[idx].numpy()]

# 卷积神经网络 CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
model = model.to(device)

# optimization
Loss = nn.CrossEntropyLoss()                               # 使用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_record = np.array([])

for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for i, data in enumerate(train_loader): 
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