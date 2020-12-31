import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time

def Jacobi(A, B, x, e):
    '''Jacobi 迭代方法'''
    T0 = time.time()
    D = sparse.diags(A.diagonal())
    L = - sparse.tril(A, -1)
    U = -sparse.triu(A, 1)
    print('Solving B_J')
    B_J = sparse.linalg.inv(D) * (L + U)
    print('Solved B_J')
    x_next = x.copy()
    times = 0
    err_record = np.array([])
    while True:
        x_next = B_J * x + sparse.linalg.inv(D) * B
        err = max(abs(x_next - x))
        err_record = np.append(err_record,err)
        
        times += 1
        print('err = ', err)
        x = x_next.copy()
        if err < e:
            break
        elif times > 1000:
            print('Times out of range')
            break
    T1 = time.time()
    print('Iteration Times = ', times)
    print('err = ', err)
    print('CPU time = ', T1 - T0)
    #print(A * x)
    #print('X = ', x)
    return x, err_record

def C_Gradient(A, B, x, e):
    '''共轭梯度方法'''
    T0 = time.time()
    r = B - A * x
    p = r.copy()
    times = 0
    err_record = np.array([])
    while True:
        times += 1
        alpha = np.dot(r.T, r) / np.dot(p.T, A * p)
        x_next = x + alpha * p
        r_next = r - alpha * A * p
        #z_next = np.dot(np.linalg.inv(M), r_next)
        beta = np.dot(r_next.T, r_next) / np.dot(r.T, r)
        p_next = r_next + beta * p
        err = max(abs(x_next - x))
        err_record = np.append(err_record, err)
        print( 'err = ', err)
        #err = max(abs(x_next - x))
        if err < e:
            break
        elif times > 100:
            print('Times out of range')
            break
        x = x_next.copy()
        r = r_next.copy()
        p = p_next.copy()
    T1 = time.time()
    print('Iteration Times = ', times)
    print('err = ',err)
    print('CPU time = ', T1 - T0)
    #print(A * x)
    return x, err_record

def Generate_matrix(n):
    M = 3.0 * np.eye(n)
    for i in range(1,n):
        M[i][i - 1] = -1
        M[i - 1][i] = -1

    N = M.copy()
    for i in range(1, n + 1):
        N[i - 1][n - i] = 1 / 2
        N[i - 1][int(n/2) - 1] = M[i - 1][int(n/2) - 1]
        N[i - 1][int(n/2)] = M[i - 1][int(n/2)]
    return N

def Sparse_matrix(n):
    row = np.array([])
    col = np.array([])
    data = np.array([])
    for i in range(n):
        if i == 0:
            row = np.append(row, i * np.ones(3))
            col = np.append(col, np.array([0, 1, n - 1]))
            data = np.append(data, np.array([3, -1, 0.5]))
        elif i == n - 1:
            row = np.append(row, i * np.ones(3))
            col = np.append(col, np.array([0, n - 2, n - 1]))
            data = np.append(data, np.array([0.5, -1, 3]))           
        elif i == (int(n/2) - 1) or (i == int(n/2)):
            row = np.append(row, i * np.ones(3))
            col = np.append(col, np.array([i - 1, i, i + 1]))
            data = np.append(data, np.array([-1, 3, -1]))
        else:
            row = np.append(row, i * np.ones(4))
            col = np.append(col, np.array([i - 1, i, i + 1, n - i - 1]))
            data = np.append(data, np.array([-1, 3, -1, 0.5]))
    M_csr = csr_matrix((data, (row, col)), shape=(n, n))
    #print(M_csr)
    return M_csr

def Generate_B(n):
    B = 1.5 * np.ones(n)
    B[0] = 2.5
    B[n - 1] = 2.5
    B[int(n/2) - 1] = 1.0
    B[int(n/2)] = 1.0
    return B


def Plot(Y1, Y2):
    X1 = np.linspace(1, len(Y1), len(Y1))
    X2 = np.linspace(1, len(Y2), len(Y2))
    plt.plot(X1,np.log10(Y1),  linewidth=1.5, linestyle="-",label="Jacobi Method")
    plt.plot(X2,np.log10(Y2), linewidth=1.5, linestyle="-",label="C-G Method")
    plt.legend()
    plt.xlabel('Iteration Times')
    plt.ylabel('log10(err)')
    plt.savefig('./Ans2_result.jpg')
    plt.show()
    



if __name__ == "__main__":
    n = 100
    err = 1e-6
    #A1 = Generate_matrix(n)
    A2 = Sparse_matrix(n)
    B = Generate_B(n)
    x = np.zeros(n)
    #print('n = 100\n','Jacobi迭代法\n','普通方法定义矩阵：')
    #X1 = Jacobi(A1, B, x, err)
    print('定义稀疏矩阵Jacobi迭代法')
    X2 = Jacobi(A2, B, x, err)
    print('\n共轭梯度法:')
    X3 = C_Gradient(A2, B, x, err)
    Plot(X2[1],X3[1])