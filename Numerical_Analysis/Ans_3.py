import numpy as np


def C_Gradient(A, B, x, e):
    '''预处理共轭梯度方法'''
    r = B - np.dot(A, x)
    p = r.copy()
    times = 0
    while True:
        times += 1
        alpha = np.dot(r.T, r) / np.dot(p.T, np.dot(A, p))
        x_next = x + alpha * p
        r_next = r - alpha * np.dot(A, p)
        #z_next = np.dot(np.linalg.inv(M), r_next)
        beta = np.dot(r_next.T, r_next) / np.dot(r.T, r)
        p_next = r_next + beta * p
        err = max(abs(x_next - x))
        #err = max(abs(x_next - x))
        print('X = ',x, 'err = ',err)
        if err < e:
            break
        elif times > 10:
            print('Times out of range')
            break
        x = x_next.copy()
        r = r_next.copy()
        p = p_next.copy()
    print('X = ',x)
    print('Times = ', times)

if __name__ == "__main__":
    A0 = np.array([[2.51, 1.48, 4.53], [1.48, 0.93, -1.30], [2.68, 3.04, -1.48]])
    B0 = np.array([0.05, 1.03, -0.53])
    A = np.dot(A0.T, A0)
    B = np.dot(A0.T, B0)
    #X = np.linalg.solve(A, B)
    #print(X,np.dot(A,X))
    #A = np.array([[3.0, 1],[1, 2]])
    #B = np.array([5.0, 5])
    x = np.array([0.0, 0, 0])
    err = 1e-6
    print('Conjugate Gradient method:')
    C_Gradient(A, B, x, err)
