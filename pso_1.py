"""
Particle Swarm Optimization
Author： Yufei
"""
import os
import math
import cv2
import random
import numpy as np


# method 2 of pso.

class PSO:
    def __init__(self, target, alpha=0.1, particle_num=100, iteration_times=10000, ranks=20):
        self.img = target  # gray mode
        self.alpha = alpha
        self.particle_num = particle_num
        self.iteration_times = iteration_times
        self.ranks = ranks
        # magic number
        self.particles_keep_rate = 0.2
        self.offsprings_mutiplier = self.particle_num // self.ranks

        self.row, self.col = self.img.shape[:2]

        self.particles_loc = None

        # movement directions
        self.directions = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                self.directions.append((i, j))

        # init the particles
        self.particle_generation(particle_num)

    def iterate(self):
        for t in range(self.iteration_times):
            if t % 5 == 0:
                if pso.draw():
                    return
            # 矩阵运算优化
            res = sorted(enumerate(self.pbest), key=lambda x: x[1], reverse=True)
            if t % 5 == 0:
                print(res)
            suboptimum_idx = [idx for idx, val in res[:20]]

            ans = []
            for i, j in self.particles_loc[suboptimum_idx, :]:
                newones = self.offsprings(i, j)
                ans.extend(newones)

            self.particles_loc = np.array(ans)
            self.pbest = [self.fit_val(i, j) for i, j in self.particles_loc]

    def fit_val(self, i, j):
        """
        smaller the intensity is and smaller the row is is, the better the fit val
        :param i: row-th num
        :param j: col-th num
        :return: fintness val
        """
        # 0 - 255
        intensity = self.img[i, j]
        res = self.alpha * (255 - intensity) - i
        return res

    def offsprings(self, i, j):
        nums = 0
        res = []
        while True:
            idx = np.random.randint(len(self.directions))
            new_i, new_j = i + self.directions[idx][0], j + self.directions[idx][1]

            if 0 <= new_i < self.row and 0 <= new_j < self.col:
                res.append([new_i, new_j])
                nums += 1
                if nums == self.offsprings_mutiplier:
                    return res

    def particle_generation(self, p_nums):
        """
        init the particle swarm
        :param p_nums: particle nums
        :return:
        """
        row, col = self.img.shape[:2]
        # xy_min = [0, 0]
        # xy_max = [row, col]
        # data = np.random.uniform(low=xy_min, high=xy_max, size=(p_nums, 2))

        last_rows = int(row * self.particles_keep_rate)
        rows = np.random.randint(row - last_rows, row, size=p_nums)
        cols = np.random.randint(col, size=p_nums)
        self.particles_loc = np.column_stack((rows, cols))

        # compute fit val for each data points
        self.pbest = [self.fit_val(i, j) for i, j in self.particles_loc]

    def draw(self):
        cv2.namedWindow('render')
        bgr_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        for i, j in self.particles_loc:
            cv2.circle(bgr_img, (j, i), 1, (0, 255, 0), 1)
        cv2.imshow('render', bgr_img)
        if cv2.waitKey(1000) & 0xff == ord('q'):
            return True


if __name__ == '__main__':
    img = cv2.imread('case1.png', 0)
    # cv2.imshow('1', img)
    # cv2.waitKey(0)
    pso = PSO(img)
    # pso.draw()
    pso.iterate()

    print('hi')
