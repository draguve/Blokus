from matplotlib import pyplot as plt
from pieces import *


def plot_piece(piece: Piece):
    plot_points(piece.possible_points.T)


def plot_points(points_to_plot):
    plt.scatter(points_to_plot[0], points_to_plot[1])
    plt.show()


def test():
    LTriomino = TriominoA()
    LTriomino.rotate(1)
    plot_piece(LTriomino)


if __name__ == '__main__':
    test()
