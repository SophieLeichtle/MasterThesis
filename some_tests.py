import numpy as np
import matplotlib.pyplot as plt


def sampleEllipsis(x_goal, x_start, c_max):
    c_min = np.linalg.norm(x_start - x_goal)
    centre = (x_start + x_goal) / 2

    M = 1 / c_max * np.array([[x_goal[0] - x_start[0], 0], [x_goal[1] - x_start[1], 0]])
    u, s, vh = np.linalg.svd(M)

    C = u @ np.array([[1, 0], [0, np.linalg.det(u) * np.linalg.det(vh)]]) @ vh

    L = np.array([[c_max / 2, 0], [0, np.sqrt(c_max**2 - c_min**2) / 2]])

    r = np.sqrt(np.random.uniform(0, 1))
    a = np.pi * np.random.uniform(0, 2)

    x_circle = np.array([r * np.cos(a), r * np.sin(a)])

    x_rand = C @ L @ x_circle + centre
    return x_rand


def ellipsisTest():
    x_goal = np.array([1, -1])
    x_start = np.array([0, 0])
    c_max = 2

    x = []
    y = []

    for i in range(1000):
        point = sampleEllipsis(x_goal, x_start, c_max)
        x.append(point[0])
        y.append(point[1])

    plt.figure()
    plt.scatter(x, y)
    plt.scatter([0, 1], [0, 1])
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()


def losTest(x1, y1, x2, y2):
    start = np.array([x1, y1])
    end = np.array([x2, y2])
    mat = np.zeros((100, 100))
    length = np.linalg.norm(start - end)
    for i in range(0, int(length) - 1):
        point = start * ((length - i) / length) + end * (i / length)
        mat[int(point[0]), int(point[1])] = 1
    plt.figure()
    plt.imshow(mat)
    plt.show()
