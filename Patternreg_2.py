import numpy as np

# learning rate
lr = 0.15

# Input data: Offset,x1,x2
X = np.array([[1/4,1/4],
            [3/4,1/8],
            [3/4,3/4],
            [1/2,1/8],
            [3/4,1/4],
            [1/4,3/4],
            [1/4,1/2],
            [1/2,1/2],
            [3/4,1/2]])

# Target data
Y = np.array([[1],[-1],[-1],
            [-1],[1],[-1],
            [-1],[-1],[1]])

# Weight matrix initialization, varying from -1 to 1
V = np.random.uniform(-1,1,size=(3,3))
W = np.random.uniform(-1,1,size=(3,1))
print('weight from input to hidden->V:',V)
print('weight from hidden to output->V:',W)

# Activation function, sigmoid or tanh
def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return x*(1-x)

def tanh(x):
    return (1-np.exp(-x))/(1+np.exp(-x))

def d_tanh(x):
    return 0.5*(1-x*x)

# Updating weight matrix
def update():
    global X,Y,W,V,lr

    # L1: data from input layer to hidden layer
    # L2: data from hidden layer to output layer
    L1 = sigmoid(np.dot(X,V))
    L2 = sigmoid(np.dot(L1,W))

    # Correction of L1 and L2 through gradient descent
    L2_c = (Y.T - L2)*d_sigmoid(L2)
    L1_c = L2_c.dot(W.T)*d_sigmoid(L1)

    # Correction of weight matrix
    W_C = lr * L1.T.dot(L2_c)
    V_C = lr * X.T.dot(L1_c)

    # Updated weight matrix
    W = W + W_C
    V = V + V_C


def main():
    for i in range(10000):
        update()
        if i%500 == 0:
            L1 = sigmoid(np.dot(X, V))
            L2 = sigmoid(np.dot(L1, W))
            print('Current error: ',np.mean(np.abs(Y.T - L2)))
    L1 = sigmoid(np.dot(X, V))
    L2 = sigmoid(np.dot(L1, W))
    print('Final approximation: ',L2)

if __name__ == "__main__":
    main()



