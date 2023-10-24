from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from pieces import *


# TODO: make this better, allow colors and delta to point as inputs
def plot_piece(piece: Piece):
    fig, ax = plt.subplots()
    plt.xlim([-1, 6])
    plt.ylim([-1, 6])
    for idx, toDraw in np.ndenumerate(piece.shape):
        if toDraw:
            ax.add_patch(Rectangle(idx, 1, 1, 0, facecolor="red"))
    plot_points(piece.possible_points.T)
    plt.show()


def plot_points(points_to_plot):
    plt.scatter(points_to_plot[0], points_to_plot[1])


def test():
    piece = TetrominoE()
    piece2 = deepcopy(piece)
    piece.rotate(1)
    plot_piece(piece2)
    plot_piece(piece)


if __name__ == '__main__':
    test()
