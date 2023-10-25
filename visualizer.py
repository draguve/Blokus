import copy
import math
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from pieces import *

colors = ["blue", "orange", "red", "green", "purple"]


# TODO: make this better, allow colors and delta to point as inputs
def plot_piece(piece: Piece):
    fig, ax = plt.subplots()
    plt.xlim([-1, 6])
    plt.ylim([-1, 6])
    plot_piece_box(ax, piece)
    plot_points(piece.possible_points.T)
    plt.show()


def plot_piece_box(ax, piece: Piece, pos=(0, 0), color="red"):
    for idx, toDraw in np.ndenumerate(piece.shape):
        if toDraw:
            ax.add_patch(Rectangle((idx[0] + pos[0], idx[1] + pos[1]), 1, 1, facecolor=color))


def plot_multiple_pieces(all_pieces, change_colors_after=2):
    color = 0
    color_idx = 0
    fig, ax = plt.subplots()
    number_of_figures_per_side = math.ceil(math.sqrt(len(all_pieces)))
    stride = 6
    limit = number_of_figures_per_side * stride + 1
    plt.xlim([-1, limit])
    plt.ylim([-1, limit])
    for i, piece in enumerate(all_pieces):
        x = (i % number_of_figures_per_side) * stride
        y = math.floor(i / number_of_figures_per_side) * stride
        plot_piece_box(ax, piece, (x, y), colors[color])
        color_idx += 1
        if color_idx == change_colors_after:
            color_idx = 0
            color += 1
            color = color % len(colors)


def plot_all_rotations():
    all_rotated_pieces = []
    all_pieces = get_all_pieces()
    for idx, piece in enumerate(all_pieces):
        all_rotated_pieces.append(piece)
        for i in range(1, 4):
            rot_piece = copy.deepcopy(piece)
            rot_piece.rotate(i)
            all_rotated_pieces.append(rot_piece)
        flipped = copy.deepcopy(piece)
        flipped.flip()
        all_rotated_pieces.append(flipped)
        for i in range(1, 4):
            rot_piece = copy.deepcopy(flipped)
            rot_piece.rotate(i)
            all_rotated_pieces.append(rot_piece)
    plot_multiple_pieces(all_rotated_pieces, 8)
    # plt.savefig("docs/all_rotations.png", format="png", dpi=1200)
    plt.show()


def plot_unique_pieces():
    unique_pieces = get_all_unique_pieces()
    total_unique_combinations = sum(len(v) for v in unique_pieces)
    print(f"There are {total_unique_combinations} unique combinations")
    fig, ax = plt.subplots()
    number_of_figures_per_side = math.ceil(math.sqrt(total_unique_combinations))
    stride = 6
    limit = number_of_figures_per_side * stride + 1
    plt.xlim([-1, limit])
    plt.ylim([-1, limit])
    location_idx = 0
    for i, piece_type in enumerate(unique_pieces):
        for j, piece in enumerate(piece_type):
            x = (location_idx % number_of_figures_per_side) * stride
            y = math.floor(location_idx / number_of_figures_per_side) * stride
            plot_piece_box(ax, piece, (x, y), colors[i % len(colors)])
            location_idx += 1
    # plt.savefig("docs/all_unique.png", format="png", dpi=1200)
    plt.show()


def plot_all_pieces():
    all_pieces = get_all_pieces()
    plot_multiple_pieces(all_pieces)
    plt.show()


def plot_points(points_to_plot):
    plt.scatter(points_to_plot[0], points_to_plot[1])


def test():
    # plot_all_pieces()
    # plot_all_rotations()
    plot_unique_pieces()


if __name__ == '__main__':
    test()
