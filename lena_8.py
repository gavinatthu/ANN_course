import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from PIL import Image

img_path = "./lena.bmp"
# 灰度图归一化处理，从 [0, 255] 转化到 [0,1.0]
trans_1 = transforms.Compose([
    transforms.ToTensor()]
)
# PIL读取灰度图片
img = Image.open(img_path).convert("L")
print("img = ",img)
# 将图片转化为PIL并输出灰度形式
img_t = trans_1(img)
print("img_t = ",img_t)
X = img_t
Y = img_t

learn_rate = 0.01
# 使用批训练方式，将数据转换为dataset模式
dataset = TensorDataset(X, Y)
train_data = DataLoader(dataset, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(1, 100),nn.Sigmoid(),
            nn.Linear(100, 10),nn.Sigmoid(),
            nn.Linear(10, 1)
        )
    def forward(self, input):
        return self.net(input)

net=Net()

# 定义优化器和损失函数
optim=torch.optim.Adam(Net.parameters(net),learn_rate)
Loss=nn.MSELoss()

# 下面开始训练：
# 一共训练 1000次
for epoch in range(1000):
    loss=None
    y_predict = net(X) 
    loss=Loss(y_predict,batyYyyych_y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # 每100次 的时候打印一次日志
    if (epoch+1)%100==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))


# 使用训练好的模型进行预测
predict=net(torch.tensor(X, dtype=torch.float))