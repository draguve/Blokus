from copy import deepcopy
from matplotlib import pyplot as plt

from pieces import *


def plot_piece(piece: Piece):
    plot_points(piece.possible_points.T)


def plot_points(points_to_plot):
    plt.xlim([0, 6])
    plt.ylim([0, 6])
    plt.scatter(points_to_plot[0], points_to_plot[1])
    plt.show()


def test():
    piece = TriominoB()
    piece2 = deepcopy(piece)
    piece.rotate(1)
    plot_piece(piece2)
    plot_piece(piece)


if __name__ == '__main__':
    test()
