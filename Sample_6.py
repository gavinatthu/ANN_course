import numpy as np

X = np.array([[0,0,0,1],
            [0,0,1,0],
            [0,1,0,0],
            [1,0,0,0]])
bias1 = np.array([1,1,1,1])
bias2 = np.array([1,1,1,1])
# 在X第一列增加偏置1
X = np.insert(X, 0, values = bias1, axis = 1)
Y = np.array([[0,0,0,1],
            [0,0,1,0],
            [0,1,0,0],
            [1,0,0,0]])
# 输入权重矩阵W 4 x 3 ，说明隐含层有3个结点，包括1个偏置
W = 2 * np.random.random((5, 2)) - 1
# 输出权重矩阵V 4 x 2 ，说明输出层有2个结点，包含1个偏置
V = 2 * np.random.random((3, 1)) - 1
learn_rate = 0.8

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def loss(predict,label):
    return 0.5*(label-predict)**2

def update():
    global X, Y, W, V, bias2, learn_rate
    L1 = sigmoid(np.dot(X, W))
    # 在隐含层增加偏置
    L1 = np.insert(L1, 0, values = bias2, axis = 1)
    # 计算输出层的权值调整
    L2 = sigmoid(np.dot(L1, V))
    Diff = Y - L2   
    Delta_L2 = Diff * d_sigmoid(L2)
    Delta_V = np.dot(L1.T, Delta_L2)
    V = V + learn_rate * Delta_V
    # 计算隐含层的权值调整
    Delta_L1 = np.dot(Delta_L2, V.T) * d_sigmoid(L1)
    Delta_W = np.dot(X.T, Delta_L1)
    Delta_W = np.delete(Delta_W, 0, axis=1)
    W = W + learn_rate * Delta_W

def main():
    for i in range(10000):
        update()
        if i%1000 == 0:
            L1 = sigmoid(np.dot(X, W))
            L1 = np.insert(L1, 0, values = bias2, axis = 1)
            L2 = sigmoid(np.dot(L1, V))
            print('Times:', i, '  Current error:',np.mean(loss(Y,L2)))
        i += 1
    L1 = np.delete(L1, 0, axis=1)
    print(L1,"\n",L2)

if __name__ == "__main__":
    main()