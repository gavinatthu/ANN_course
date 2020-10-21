from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import numpy as np
import torchimpor

# Input data x and target data y
x1 = np.linspace(-1,1,100)
x2 = np.linspace(-1,1,100)
x = np.vstack((x1,x2))
print(x)
x1, x2 = np.meshgrid(x1, x2)
# 高斯白噪声
# noise = 0.4 * np.random.normal(0, 0.1, (1,100))
# peaks 二元函数多项式
y =  1/8*(3 * (1 - 4*x1)**2 * np.exp(-((4*x1)**2) - (4*x2 + 1)**2)\
   - 10 * (4*x1/5 - (4*x1)**3 - (4*x2)**5)*np.exp(-(4*x1)**2 - (4*x2)**2)\
   - 1/3 * np.exp(-(4*x1 + 1)**2 - (4*x2)**2))

lr = 0.12
# 使用批训练方式，将数据转换为dataset模式
dataset=TensorDataset(torch.tensor(x1,dtype=torch.float),torch.tensor\
    (x2,dtype=torch.float),torch.tensor(y,dtype=torch.float))

# 将dataset置入dataloader中，批大小batch_size，随机打乱顺序shuffle，读取数据的线程数
dataloader=DataLoader(dataset,batch_size=20,shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_features=2,out_features=10),nn.Sigmoid(),
            nn.Linear(10,10),nn.Sigmoid(),
            nn.Linear(10,1)
        )

    def forward(self, input:(torch.FloatTensor,torch.FloatTensor)):
        return self.net(input)

net=Net()

# 定义优化器和损失函数
optim=torch.optim.Adam(Net.parameters(net),lr)
Loss=nn.MSELoss()


# 下面开始训练：
# 一共训练 1000次
for epoch in range(1000):
    loss=None
    for batch_x1,batch_x2,batch_y in dataloader:
        y_predict = net(batch_x1,batch_x2) 
        loss=Loss(y_predict,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # 每100次 的时候打印一次日志
    if (epoch+1)%100==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))


# 使用训练好的模型进行预测
predict=net(torch.tensor(X,dtype=torch.float))

# 绘图流程

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap='rainbow')
plt.show()
plt.plot(x,y,label="ideal", color='red',linestyle='--')
plt.plot(X,Y,label="fact", color='blue')
plt.plot(x,predict.detach().numpy(),label="predict", color='green')
plt.title("Hermit Fitting")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(fname="result.png", dpi=300,figsize=[10000,1000])
plt.show()
