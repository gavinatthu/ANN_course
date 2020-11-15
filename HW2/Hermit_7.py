import numpy as np
from sources.generator_distributions import mix_distributions, generate_normal_distributions, generate_expon_distributions, get_y_train
from sources import config
from math import sqrt
import time


x = np.linspace(-1,1,100).reshape(100,1)
X_values = x
y = 0.4 * (1 - 4 * x + 32 * x**2) * np.exp(-8 * x**2)
y_values = y


X = np.array([[0], [0.1], [0.2], [0.3], [0.4], [0.5],
            [0.6], [0.7], [0.8], [0.9], [1]])

D = np.array([[-0.5], [-0.4], [-0.3], [-0.2], [-0.1], [0],
            [0.1], [0.2], [0.3], [0.4], [0.5]])


def normalization(M):
    """对行向量进行归一化
    :param M:行向量：【dim=len(M)】
    :return: 归一化后的行向量M
    """
    M = M/np.sqrt(np.dot(M,M.T))
    return M


def normalization_all(N):
    """对矩阵进行归一化
    :param N: 矩阵：【m,n】
    :return: 归一化后的矩阵M_all:【m,n】
    """
    M_all=[]
    for i in range(len(N)):
        K = normalization(N[i])
        M_all.append(K)
    return M_all


class competitive_network(object):
    def __init__(self,x_dim, neu_dim, output_dim, lr, lr_out):
        '''类参数初始化
        '''
        V = np.ones((neu_dim, x_dim))/2
        W = np.ones((output_dim, neu_dim))/2
        Y = np.zeros(output_dim)
        V = normalization_all(V)
        self.V = V
        self.W = W
        self.Y = Y
        self.lr = lr
        self.lr_out = lr_out

    def forward_propagation(self, x):
        '''前向传播
        input:self(object):类参数
              x(mat):一个训练样本
        output:argmax(int):被激活的权重向量指针
        '''
        # (W,x)最大的一个神经元
        z_layer = np.dot(self.V, x.T)
        #a_layer = sigmoid(z_layer)
        argmax = np.argmax(z_layer)
        return argmax
    
    def back_propagation(self, argmax, X):
        '''反向传播调整权重
        input:argmax(int):被激活的权重向量指针
              x(mat):一个训练样本
        '''
        #print(self.V)
        self.V[argmax] = self.V[argmax] + self.lr * (X - self.V[argmax])
        self.V[argmax] = normalization(self.V[argmax])
        self.lr -= self.lr / (num_iter_1)


    def back_propagation_2(self, argmax, D):
        '''反向传播调整权重
        input:argmax(int):被激活的权重向量指针
              x(mat):一个训练样本
        '''
        self.W[argmax] = self.W[argmax] + self.lr_out * (D - self.W[argmax])
        self.W[argmax] = normalization(self.W[argmax])
        self.lr_out -= self.lr_out / (num_iter_2)




    def train(self,X,D,num_iter_1, num_iter_2):
        '''模型训练
        input:X(mat):全部训练样本
              num_iter(int):迭代次数
        '''
        X = np.array(X)
        D = np.array(D)
        # 学习速率减小
        for i in range(num_iter_1):
            for j in range(X.shape[0]):
                argmax_1 = self.forward_propagation(X[j])
                self.Y = np.zeros(output_dim)
                self.Y[argmax_1] = 1.0 #获胜神经元输出为1
                self.back_propagation(argmax_1, X[j])
        #print(self.W)
        print(self.V)

        for i in range(num_iter_2):
            for j in range(X.shape[0]):
                argmax = self.forward_propagation(X[j])
                self.Y = np.zeros(output_dim)
                self.Y[argmax] = 1.0 #获胜神经元输出为1
                self.back_propagation_2(argmax, D[j])
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
    num_iter_1 = 10
    num_iter_2 = 10
    x_dim = np.shape(X)[1]
    output_dim = 15
    neu_dim = 5
    lr = 1
    lr_out = 1
    cnn = competitive_network(x_dim, neu_dim, output_dim, lr, lr_out)
    cnn.train(X, D, num_iter_1, num_iter_2)
    #train_results = cnn.prediction(X)
    #print('train results = ',train_results, '\n')
