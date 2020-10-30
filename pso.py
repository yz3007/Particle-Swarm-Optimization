"""
Particle Swarm Optimization
Author： Yufei
"""
import os
import math
import cv2
import random
import numpy as np


# boundary limits

class PSO:
    def __init__(self, target, alpha=0.5, particle_num=100, p_lr_rate=0.5, g_lr_rate=1, iteration_times=1000):
        self.img = target  # gray mode
        self.alpha = alpha
        self.particle_num = particle_num
        self.p_lr_rate = p_lr_rate
        self.g_lr_rate = g_lr_rate
        self.iteration_times = iteration_times

        # magic number
        self.init_velocity_range = 2
        self.row, self.col = self.img.shape[:2]

        self.particles_loc = None
        self.particles_velocity = None
        # each particles
        self.pbest = None
        self.pbest_i_j = None
        # global range
        self.gbest = None
        self.gbest_i_j = None

        # init the particles
        self.particle_generation(particle_num)

    def iterate(self):
        for t in range(self.iteration_times):
            if t%100 == 0:
                pso.draw()
            # 矩阵运算优化
            for idx in range(self.particle_num):
                nxt_velocity = self.get_velocity(self.particles_velocity[idx], self.particles_loc[idx],
                                                 self.pbest_i_j[idx], self.gbest_i_j)
                nxt_pos = self.particles_loc[idx] + nxt_velocity
                # nxt_pos =
                # limits the pos range inside the image (r x c)
                nxt_pos[0] = np.clip(nxt_pos[0], 0, self.row - 1)
                nxt_pos[1] = np.clip(nxt_pos[1], 0, self.col - 1)

                nxt_pos = nxt_pos.astype(int)
                new_fit = self.fit_val(nxt_pos[0], nxt_pos[1])
                if new_fit > self.pbest[idx]:
                    self.pbest[idx] = new_fit
                    self.pbest_i_j[idx] = nxt_pos

                self.particles_loc[idx] = nxt_pos
                self.particles_velocity[idx] = nxt_velocity

            self.gbest = np.max(self.pbest)
            self.gbest_i_j = self.pbest_i_j[np.argmax(self.pbest)]
            print(self.gbest)
            print(self.gbest_i_j)
            print(self.fit_val(self.gbest_i_j[0], self.gbest_i_j[1]))

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

        rows = np.random.randint(row, size=p_nums)
        cols = np.random.randint(col, size=p_nums)
        self.particles_loc = np.column_stack((rows, cols))

        d_rows = np.random.randint(-self.init_velocity_range, self.init_velocity_range + 1, size=p_nums)
        d_cols = np.random.randint(-self.init_velocity_range, self.init_velocity_range + 1, size=p_nums)
        self.particles_velocity = np.column_stack((d_rows, d_cols))
        # compute fit val for each data points
        self.pbest = [self.fit_val(i, j) for i, j in self.particles_loc]
        self.gbest = np.max(self.pbest)
        self.pbest_i_j = np.copy(self.particles_loc)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.gbest_i_j = self.pbest_i_j[np.argmax(self.pbest)]

    def get_velocity(self, cur_v, cur_i_j, pbest_i_j, gbest_i_j):
        nxt_v = cur_v + self.p_lr_rate * random.random() * (pbest_i_j - cur_i_j) + self.g_lr_rate * random.random() * (
                gbest_i_j - cur_i_j)

        return nxt_v

    def update_position(self, cur_i_j, new_v):
        return cur_i_j + new_v

    def draw(self):
        cv2.namedWindow('render')
        bgr_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        for i, j in self.particles_loc:
            cv2.circle(bgr_img, (j, i), 1, (0, 255, 0), 1)
        cv2.imshow('render', bgr_img)
        cv2.waitKey(1000)


if __name__ == '__main__':
    img = cv2.imread('case1.png', 0)
    # cv2.imshow('1', img)
    # cv2.waitKey(0)
    pso = PSO(img)
    # pso.draw()
    pso.iterate()

    print('hi')
