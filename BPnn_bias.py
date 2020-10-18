import numpy as np

def init():
    X = np.array([[1,1,1],
                [1,1,2],
                [1,2,2],
                [2,2,2]])
    bias1 = np.array([-1,-1,-1,-1])
    bias2 = np.array([-1,-1,-1,-1])
    # 在X最后一列增加偏置-1
    X = np.insert(X, 3, values = bias1, axis = 1)
    Y = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])

    # 输入权重矩阵W 4 x 3 ，说明隐含层有3个结点，包括1个偏置
    W = 2 * np.random.random((4, 3)) - 1
    # 输出权重矩阵V 4 x 2 ，说明输出层有2个结点，包含了偏置之后，维度增加1
    V = 2 * np.random.random((4, 2)) - 1
    learn_rate = 0.15

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def loss(predict,label):
    return 0.5*(label-predict)**2

def update():
    global X, Y, W, V, bias2
    L1 = sigmoid(np.dot(X, W))
    L1 = np.insert(L1, 3, values = bias2, axis = 1)
    
    # 输出层的计算
    L2 = sigmoid(np.dot(L1, V))
    Diff = Y - L2   
    Delta_L2 = Diff * d_sigmoid(L2)
    Delta_V = np.dot(L1.T, Delta_L2)  #B相当于是输出层的输入，把每个节点的5次训练产生的delta_V累积了
    V = V + learn_rate * Delta_V

    # 计算隐含层的权的调整
    Delta_L1 = np.dot(Delta_C, V.T) * d_sigmoid(L1)
    Delta_W = np.dot(X.T, Delta_L1)  # 把多个样本的每个节点的偏差累积
    Delta_W = np.delete(Delta_W, 3, axis=1)
    W = W + learn_rate * Delta_W


def main():
    init()
    for i in range(10000):
        #--------------------------------------------
        # 隐含层的结点总输入
        update()
        if i%1000 == 0:
            L1 = sigmoid(np.dot(X, W))
            L1 = np.insert(L1, 3, values = bias2, axis = 1)
            L2 = sigmoid(np.dot(L1, V))
            print('Times:', i, '  Current error:',np.mean(loss(Y,L2)))

        #print('Delta_W=\n', Delta_W)
        i += 1
        #end of while
    print(C)

if __name__ == "__main__":
    main()