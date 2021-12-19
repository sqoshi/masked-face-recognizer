import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def g(x, a, b):
    return a * x + b


def draw_red_class(ax):
    # x = [1, 1.5, 2, 3]
    # y = [(1, 2), (1.5,), (1, 2), (2.5,)]
    x = [1, 2, 3]
    y = [(0,), (0,), (0,)]
    # y = [(1,),(4**2,),(9**2,),(49**2,),(64**2,)]
    for xe, ye in zip(x, y):
        ax.scatter([xe] * len(ye), ye, color="red", s=100.5)


def draw_blue_class(ax):
    # x = [3, 3.5, 4, 2]
    # y = [(3,), (3.5, 2.5), (3,), (2.5,)]
    x = [4, 7, 8]
    y = [(0,), (0,), (0,)]  # , (0,), (0,)]

    # y = [(16**2,),(25**2,),(36**2,)]
    for xe, ye in zip(x, y):
        ax.scatter([xe] * len(ye), ye, color="blue", s=100.5)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    draw_red_class(ax)
    draw_blue_class(ax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Example problematic data for maximal margin method.")
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a, b = 0, 2.25
    # y = [ a*x+b for x in values]
    # y = [ 0 for _ in values ]
    # ax.plot(values,y, color="green")
    # a,b = -1,4
    # y = [ a*x+b for x in values]
    # ax.plot(values,y, color="purple", linestyle='--')
    # a,b = 601,-2010
    a, b = 0, 3.5
    y = [a * x + b for x in values]
    # ax.plot(values,y, color="green", linestyle='-')

    # plt.xticks([])
    # plt.yticks([])
    # ax.legend(["Red", "Blue"])
    # leg = ax.get_legend()
    # leg.legendHandles[0].set_color('red')
    # leg.legendHandles[1].set_color('blue')
    fig.savefig("samplefigure", bbox_inches="tight")
