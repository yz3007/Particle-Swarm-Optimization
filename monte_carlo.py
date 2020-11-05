"""
monte carlo
what is pi ?
"""
import numpy as np
import cv2


def main():
    # pi*r^2/(2r)**2 = the percent of points fall into the circle region
    size = 1 << 10
    x = np.random.uniform(0, 1, size)
    y = np.random.uniform(0, 1, size)
    points = np.column_stack((x, y))

    center = np.array([0.5, 0.5])

    res = np.sum(np.square(points - center), axis=-1)
    percent = len(np.where(res <= 0.25)[0]) / size

    return percent * 4


if __name__ == '__main__':
    res = main()
    print(res)
