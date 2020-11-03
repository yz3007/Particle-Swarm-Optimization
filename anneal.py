""""
Simulated Annealing algorithm 退火算法
https://www.imooc.com/article/details/id/30160
温度越高，，越允许搜索解进行波动，即使出现差的解，也会有更高的概率允许它出现，为跳出局部最优做准备。
温度低则越来越稳定。
"""
import numpy as np
import matplotlib.pyplot as plt
import math


class Anneal:
    def __init__(self, T=1000, Tmin=10, k=100):
        self.T = T  # init temperature
        self.Tmin = Tmin  # minimum val of temp
        self.k = k  # times of internal circulation

        self.x = np.random.uniform(low=0, high=100)  # init search point
        self.y = 0  # init result
        self.t = 0  # time

    def iterate(self):
        # outer loop
        while self.T > self.Tmin:
            # inner loop
            for i in range(self.k):
                self.y = self.fit_val(self.x)
                # generate a new x in the neighborhood of x by transform func
                xNew = self.x + np.random.uniform(low=-0.055, high=0.055) * self.T
                if 0 <= xNew <= 100:
                    yNew = self.fit_val(xNew)
                    if yNew - self.y < 0:
                        self.x = xNew
                    else:
                        # metroplolis principle
                        p = math.exp(-(yNew - self.y) / self.T)
                        r = np.random.uniform(low=0, high=1)
                        if r < p:
                            self.x = xNew
            self.t += 1
            # print(self.t)
            print('times', self.t, ':', self.x, self.y)
            self.T = 1000 / (1 + self.t)

        print(self.x, self.y)

    def fit_val(self, x):
        return x ** 3 - 60 * (x ** 2) - 4 * x + 6

    def draw(self):
        x = [i / 10 for i in range(1000)]
        y = [0] * 1000

        for i in range(1000):
            y[i] = self.fit_val(x[i])
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    anneal = Anneal()
    # anneal.draw()
    anneal.iterate()
