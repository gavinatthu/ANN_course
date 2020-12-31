import numpy as np


def Jacobi(A, B, x, e):
    '''Jacobi 迭代方法'''
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    B_J = np.dot(np.linalg.inv(D), (L + U))
    B_GS = np.dot(np.linalg.inv(D - L), U)
    x_next = x.copy()
    times = 0
    while True:
        x_next = np.dot(B_J, x) + np.dot(np.linalg.inv(D), B)
        err = max(abs(x_next - x))
        print('X = ', x, ' R = ', err)
        x = x_next.copy()
        times += 1
        if err < e:
            break
        elif times > 1000:
            print('Times out of range')
            break
    print('X = ', x)
    print('Times = ', times)

def Gauss_Seidel(A, B, x, e):
    '''Gauss——Seidel迭代方法'''
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    B_J = np.dot(np.linalg.inv(D), (L + U))
    B_GS = np.dot(np.linalg.inv(D - L), U)
    x_next = x.copy()
    times = 0
    while True:
        x_next = np.dot(B_GS, x) + np.dot(np.linalg.inv(D - L), B)
        err = max(abs(x_next - x))
        print('X = ', x, ' R = ', err)
        x = x_next.copy()
        times += 1
        if err < e:
            
            break
        elif times > 1000:
            print('Times out of range')
            break
    print('X = ', x)
    print('Times = ', times)


def Spectral_radius(A):
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    B_J = np.dot(np.linalg.inv(D), (L + U))
    B_GS = np.dot(np.linalg.inv(D - L), U)
    a, a1 = np.linalg.eig(B_J)
    b, b1 = np.linalg.eig(B_GS)
    return max(abs(a)), -np.log(max(abs(a))), max(abs(b)), -np.log(max(abs(b)))
    

def SOR(w, A, B, x, e):
    '''逐次超松弛迭代方法SOR, w->float: 松弛因子'''
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    M = (1.0 / w) * (D - w * L)
    N = (1.0 / w) * ((1.0 - w) * D + w * U)
    L_SOR = np.dot(np.linalg.inv(D - w * L), ((1.0 - w) * D + w * U))
    x_next = x.copy()
    times = 0
    while True:
        x_next = np.dot(np.dot(np.linalg.inv(M), N), x) + np.dot(np.linalg.inv(M), B)
        err = max(abs(x_next - x))
        x = x_next.copy()
        times += 1
        if err < e:
            break
        elif times > 1000:
            print('Times out of range')
            break
    print('Times = ', times)
    print('X = ', x_next, ' R = ', err)
    


if __name__ == "__main__":
    A = np.array([[11.0, 3, -2], [-1.0, 5, -3], [-2.0, -12, 19]])
    B = np.array([3.0, 6, -7])
    x = np.array([0.0, 0, 0])
    err = 1e-5
    radius = Spectral_radius(A)
    print('Jacobi method:')
    Jacobi(A, B, x, err)
    print('Rho = ', radius[0], 'Speed = ', radius[1])
    print('\nGauss_Seidel method:')
    Gauss_Seidel(A, B, x, err)
    print('Rho = ', radius[2], 'Speed = ', radius[3])

    print('\nSOR:')
    for w in [0.8, 1.0, 1.2, 1.4, 1.6]:
        print('w = ', w)
        SOR(w, A, B, x, err)
    w_opt = 2.0 / (1 + np.sqrt(1 - np.square(radius[0])))
    print('\n最优松弛因子\n','w_opt = ', w_opt)
    SOR(w_opt, A, B, x, err)