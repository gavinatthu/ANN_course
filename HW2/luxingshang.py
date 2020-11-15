
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl


# 高斯分布的近邻
def get_neighborhood(center, radix, domain):
    if radix < 1:
        radix = 1
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)
    return np.exp(-(distances*distances) / (2*(radix*radix)))

# 基于极差的归一化方法
def normalization(citys):
    x_min, y_min = citys.min(axis=0)[1:]
    x_max, y_max = citys.max(axis=0)[1:]
    citys[:,1] = (citys[:,1]-x_min)/(x_max-x_min)
    citys[:,2] = (citys[:,2]-y_min)/(y_max-y_min)
    return citys

# 找到与被选中城市距离最小的神经元
def get_nearest(city, network):
    nearest_n = -1
    min_dis = float('inf')
    for j in range(len(network)):
        dis = math.sqrt(sum(pow(city - network[j], 2)))
        if dis < min_dis:
            min_dis = dis
            nearest_n = j
    return nearest_n

# 训练过程
def train(citys, lr, n, iterations):
    # 初始化竞争层
    network = np.random.rand(n, 2)
    for i in range(iterations):
        #随机选择一个城市
        select_city = random.randint(0, len(citys)-1)
        city = citys[select_city][1:]
        nearest_n = get_nearest(city, network)
 
        #以该神经元为中心建立高斯分布
        gaussian = get_neighborhood(nearest_n, n // 5 , network.shape[0])
        network += gaussian[:, np.newaxis] * lr * (city - network)
        
        # 学习率和有效领域宽度的衰减规则
        n = n * math.pow(0.001, 1/iterations)
        lr = lr * math.pow(0.001, 1/iterations)

        if i%1000 == 0:
            plt.scatter(citys[:,1], citys[:,2], color='red', s=4)
            plt.plot(network[:,0], network[:,1], 'r.', ls='-', color='#0063ba', markersize=2)
            plt.savefig('./iter{}.jpg'.format(i), bbox_inches='tight', pad_inches=0, dpi=200)
            plt.close()
    return network

def main():
    cities = np.array([[1, 246.0, 53.0],
                [2, 408, 79],
                [3, 909, 89],
                [4, 115, 264],
                [5, 396, 335],
                [6, 185, 456],
                [7, 699, 252],
                [8, 963, 317],
                [9, 922, 389],
                [10, 649, 515]])
    lr = 0.8
    start = 0
    iterations = 10000
    n = len(cities)*8

    temp_citys = cities.copy()

    citys = normalization(cities)

    network = train(citys, lr, n, iterations)
 
    # 画图
    plt.scatter(citys[:,1], citys[:,2], color='red', s=4)
    plt.plot(network[:,0], network[:,1], 'r.', ls='-', color='#0063ba', markersize=2)
    plt.savefig('./iter{}.jpg'.format(iterations), bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

    new_citys = []

    for i in range(len(citys)):
        city = citys[i][1:]
        nearest_city = -1
        min_dis = float('inf')
        for j in range(len(network)):
            dis = math.sqrt(sum(pow(city - network[j], 2)))
            if dis < min_dis:
                min_dis = dis
                nearest_city = j
        new_citys.append([i, nearest_city])
 
    new_citys_ = sorted(new_citys, key=lambda x:x[1])
    new_citys_.append(new_citys_[0])
    new_citys_ = np.array(new_citys_)
    final_path = temp_citys[new_citys_[:,0],:][:,1:]
    path_lenght = 0
    for i in range(len(final_path)-1):
        path_lenght += math.sqrt(sum(pow(final_path[i] - final_path[i+1], 2)))
    print('final distance:{}'.format(path_lenght))
    plt.scatter(final_path[:, 0], final_path[:, 1], color='red', s=4)
    plt.plot(final_path[:, 0], final_path[:, 1], 'r.', ls='-', color='#0063ba', markersize=2)
    plt.savefig('./route.jpg', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
 

 
if __name__ == '__main__':
    main()
