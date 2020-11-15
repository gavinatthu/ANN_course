import numpy as np
import math as m

# 导入数据
X = np.array([[0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0],
            [0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0],
            [1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]])

for j in range(3):
    for k in range(25):
        A = np.array(X)
        A[j][k] = 1 - A[j][k]
        X = np.concatenate((X, [A[j]]), axis=0)

train_set = np.concatenate((X[:8], X[28:33]))
train_set = np.concatenate((train_set, X[53:58]))

test_set = np.concatenate((X[8:28], X[33:53]))
test_set = np.concatenate((test_set, X[58:78]))


def sigmoid(x):
    return 1/(1+np.exp(-x))

def normalization(M):
    M=M/np.sqrt(np.dot(M,M.T))
    return M

def normalization_all(N):
    M_all=[]
    for i in range(len(N)):
        K=normalization(N[i])
        M_all.append(K)
    return M_all

class competitive_network(object):
    def __init__(self,x_dim,output_num,a):
        W = np.random.rand(output_num,x_dim)
        self.W = normalization_all(W)
        self.a = a

    def forward_propagation(self,x):
        z_layer = np.dot(self.W,x.T)
        a_layer = sigmoid(z_layer) 
        argmax = np.argmax(a_layer)
        return argmax
    
    def back_propagation(self,argmax,x):
        self.W[argmax] = self.W[argmax] + self.a * (x - self.W[argmax])
        self.W[argmax] = normalization(self.W[argmax])
        self.a -= self.a / (num_iter-1) 
    
    def train(self,X,num_iter):
        X = np.array(X)
        for item in range(num_iter):
            for i in range(X.shape[0]):
                argmax = self.forward_propagation(X[i])
                self.back_propagation(argmax,X[i])
        #print(self.W)
            
    def prediction(self,X_test):
        sample_num = np.shape(X_test)[0]
        predict_results = []
        for i in range(sample_num):
            predict_result = self.forward_propagation(X_test[i])
            predict_results.append(predict_result)
        return predict_results
        


if __name__ == '__main__':
    num_iter = 100
    x_dim = np.shape(X)[1]
    output_num = 3
    learn_rate = 0.1
    cnn = competitive_network(x_dim,output_num,learn_rate)
    cnn.train(train_set,num_iter)
    train_results = cnn.prediction(train_set)
    while train_results[1] == train_results[2]:
        print('train results = ',train_results,train_results[0]+train_results[1]+train_results[2])
        cnn = competitive_network(x_dim,output_num,learn_rate)
        cnn.train(train_set,num_iter)
        train_results = cnn.prediction(train_set)
    predict_results = cnn.prediction(test_set)
    print(train_results[0]+train_results[1]+train_results[2])
    print('test results = ',predict_results)