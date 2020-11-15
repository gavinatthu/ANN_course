import numpy as np
import math as m

# 导入数据
X = np.array([[m.cos(m.pi/180*45),m.sin(m.pi/180*45)],
            [m.cos(m.pi/180*90),m.sin(m.pi/180*90)],
            [m.cos(m.pi/180*130),m.sin(m.pi/180*130)],
            [m.cos(m.pi/180*180),m.sin(m.pi/180*180)],
            [m.cos(m.pi/180*(-135)),m.sin(m.pi/180*(-135))]])



def sigmoid(x):
    '''sigmoid激活函数
    '''
    return 1/(1+np.exp(-x))

def normalization(M):
    """对行向量进行归一化
    :param M:行向量：【dim=len(M)】
    :return: 归一化后的行向量M
    """
    M=M/np.sqrt(np.dot(M,M.T))
    return M

def normalization_all(N):
    """对矩阵进行归一化
    :param N: 矩阵：【m,n】
    :return: 归一化后的矩阵M_all:【m,n】
    """
    M_all=[]
    for i in range(len(N)):
        K=normalization(N[i])
        M_all.append(K)
    return M_all

class competitive_network(object):
    def __init__(self,x_dim,output_num,a):
        '''类参数初始化
        '''
        #W = np.random.rand(output_num,x_dim) * (2) - 1
        W = np.array([[1.0,0.0],[0.0,-1.0]])
        self.W = normalization_all(W)
        self.a = a ## 权值更新参数

    def forward_propagation(self,x):
        '''前向传播
        input:self(object):类参数
              x(mat):一个训练样本
        output:argmax(int):被激活的权重向量指针
        '''
        z_layer = np.dot(self.W,x.T)
        a_layer = sigmoid(z_layer) 
        argmax = np.argmax(a_layer)
        return argmax
    
    def back_propagation(self,argmax,x):
        '''反向传播调整权重
        input:argmax(int):被激活的权重向量指针
              x(mat):一个训练样本
        '''
        self.W[argmax] = self.W[argmax]+ self.a * (x - self.W[argmax])
        self.W[argmax] = normalization(self.W[argmax])
        self.a -= self.decay
    
    def train(self,X,num_iter):
        '''模型训练
        input:X(mat):全部训练样本
              num_iter(int):迭代次数
        '''
        X = np.array(X)
        self.decay = self.a / num_iter
        for item in range(num_iter):
            for i in range(X.shape[0]):
                argmax = self.forward_propagation(X[i])
                self.back_propagation(argmax,X[i])
            print(self.W)
            
    def prediction(self,X_test):
        '''预测样本的类别
        input:self(object):类
              X_test(mat):测试样本
        output:predict_class(list):样本类别
        '''
        sample_num = np.shape(X_test)[0]
        predict_results = []
        for i in range(sample_num):
            predict_result = self.forward_propagation(X_test[i])
            predict_results.append(predict_result)
        return predict_results
        


if __name__ == '__main__':
    print('---------------------1.Load Data---------------------')
    data = X
    dataMat = np.mat(data)
    print('------------------2.Parameters Seting----------------')
    num_iter = 10
    x_dim = np.shape(dataMat)[1]
    output_num = 2
    a = 0.6
    print('-------------------3.Model Train---------------------')
    cnn = competitive_network(x_dim,output_num,a)
    cnn.train(dataMat,num_iter)
    print('-------------------4.Prediction----------------------')
    predict_results = cnn.prediction(X)
    print(predict_results)