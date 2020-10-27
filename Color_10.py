import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 文件路径 
img_path = "./HMW1-dataset/dataset/color/train"
test_path = "./HMW1-dataset/dataset/color/test"

# 构建存储数据结果的空列表，img_list为输入数据，color_list为目标数据
img_list = []
color_list = []
# 构建测试集所需的空列表，test_list为test输入，taget_list为检验数据
test_list = []
target_list = []
# 获取图片的颜色向量
def byte(input):
    if input == "蓝":
        return [1, 0, 0, 0]
    elif input == "白":
        return [0, 1, 0, 0]
    elif input == "黑":
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

# 在文件夹中对每一张图片操作，调取os库
for filename in os.listdir(img_path):
    img = Image.open(img_path + "/" + filename).convert("RGB")
    # 初始化空列表rgb_list存储本图片中RGB数据
    rgb_list = []
    # 每一张图片在全图范围内随机选取100个像素
    for i in range(100):
        img_x = np.random.randint(0,img.size[0])
        img_y = np.random.randint(0,img.size[1])
        # 将每一个像素处的(R,G,B)值降维，并归一化列入列表rgb_list之后
        rgb = img.getpixel((img_x,img_y))
        rgb = np.array(rgb,dtype=float)/255
        rgb_list.append(rgb)
    # 将3*100的列表rgb_list降维成300*1 并列在列表img_list之后
    img_rgb = np.array(rgb_list).flatten()
    img_list.append(img_rgb)
    # 将该张图片的颜色向量列入列表#color_list之后
    color_list.append(byte(filename[7]))

# list转为tensor
# print(img_list)
# print(color_list)
img_tensor = torch.Tensor(img_list)
color_tensor = torch.Tensor(color_list)
print(img_tensor.size())
print(color_tensor.size())


X = img_tensor
Y = color_tensor

learn_rate = 0.008
# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(300, 10),nn.Sigmoid(),
            nn.Linear(10, 10),nn.Sigmoid(),
            nn.Linear(10, 4)
        )
    def forward(self, input):
        return self.net(input)

net=Net()

# 定义优化器和损失函数
optim=torch.optim.Adam(Net.parameters(net),learn_rate)
Loss=nn.MSELoss()

# 下面开始训练：
# 一共训练 1000次
for epoch in range(5000):
    loss = None
    y_predict = net(X) 
    loss = Loss(y_predict,Y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # 每1000次打印一次日志
    if (epoch+1)%1000==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))


# 检验网络结果
for filename in os.listdir(test_path):
    test = Image.open(test_path + "/" + filename).convert("RGB")
    rgb_list = []
    for i in range(100):
        img_x = np.random.randint(0,img.size[0])
        img_y = np.random.randint(0,img.size[1])
        rgb = img.getpixel((img_x,img_y))
        rgb = np.array(rgb,dtype=float)/255
        rgb_list.append(rgb)
    img_rgb = np.array(rgb_list).flatten()
    test_list.append(img_rgb)
    target_list.append(byte(filename[7]))
test_tensor = torch.Tensor(test_list)
target_tensor = torch.Tensor(target_list)
predict = net(test_tensor)

print(test_tensor.size())
print(target_tensor.size())

print("MSEloss of test=",Loss(predict,target_tensor).item())
