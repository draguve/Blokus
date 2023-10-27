import copy
import math
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from pieces import *


colors = ["blue", "orange", "red", "green", "purple"]
dot_colors = ["#0000b2", "#b27300", "#b20000", "#003300"]


def plot_piece_base(piece):
    fig, ax = plt.subplots()
    plt.xlim([-1, 6])
    plt.ylim([-1, 6])
    plot_piece_box(ax, piece)
    plot_points(piece.possible_points)


# TODO: make this better, allow colors and delta to point as inputs
def plot_piece(piece: Piece):
    plot_piece_base(piece)
    plt.show()


def plot_piece_and_save(piece, filename):
    fig, ax = plt.subplots()
    plt.xlim([-1, 6])
    plt.ylim([-1, 6])
    plot_box(ax, piece, (0, 0), "red")
    plt.savefig(f"{filename}.png", format="png")
    plt.close()


def plot_piece_box(ax, piece: Piece, pos=(0, 0), color="red"):
    plot_box(ax, piece.shape, pos, color)


def plot_box(ax, piece: np.array, pos=(0, 0), color="red"):
    for idx, toDraw in np.ndenumerate(piece):
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


def plot_collision_masks():
    unique_pieces = get_all_unique_pieces()
    total_unique_combinations = sum(len(v) for v in unique_pieces)
    fig, ax = plt.subplots()
    number_of_figures_per_side = math.ceil(math.sqrt(total_unique_combinations))
    stride = 8
    limit = number_of_figures_per_side * stride + 1
    plt.xlim([-1, limit])
    plt.ylim([-1, limit])
    location_idx = 0
    for i, piece_type in enumerate(unique_pieces):
        for j, piece in enumerate(piece_type):
            x = (location_idx % number_of_figures_per_side) * stride
            y = math.floor(location_idx / number_of_figures_per_side) * stride
            plot_box(ax, piece.collision_mask, (x, y), "red")
            plot_box(ax, piece.shape, (x + 1, y + 1), "blue")
            plot_points(piece.possible_points, (x + 1, y + 1), "green", s=2)
            location_idx += 1
    # plt.savefig("docs/all_collision_masks.png", format="png", dpi=1200)
    plt.show()


def plot_all_pieces():
    all_pieces = get_all_pieces()
    plot_multiple_pieces(all_pieces)
    # plt.savefig("docs/all_pieces.png", format="png", dpi=1200)
    plt.show()


def plot_points(points_to_plot, origin_point=(0, 0), color="blue", s=None):
    delta = np.array(origin_point)
    if points_to_plot.shape[0] <= 0:
        return
    points_to_plot = (points_to_plot + delta).T
    plt.scatter(points_to_plot[0], points_to_plot[1], c=color, s=s)


def plot_board(board):
    fig, ax = plt.subplots()
    plt.xlim([-1, 21])
    plt.ylim([-1, 21])
    for i in range(4):
        plot_box(ax, board.playerBoards[i], (0, 0), colors[i])
    for i in range(4):
        s = 20 - 3 * i
        plot_points(np.array(board.open_board_points[i]), (0, 0), dot_colors[i], s=s)


def plot_show_board(board):
    plot_board(board)
    plt.show()


def plot_store_board(board, filename):
    plot_board(board)
    plt.savefig(f"{filename}.png", format="png")
    plt.close()


def plot_remaining_pieces(board, filename):
    fig, ax = plt.subplots()

    to_plot = 0
    for i in range(4):
        for upiece in board.available_pieces_per_player[i]:
            if len(upiece) == 0:
                continue
            to_plot += 1

    number_of_figures_per_side = math.ceil(math.sqrt(to_plot))
    stride = 6
    limit = number_of_figures_per_side * stride + 1
    plt.xlim([-1, limit])
    plt.ylim([-1, limit])
    index = 0
    for i in range(4):
        for upiece in board.available_pieces_per_player[i]:
            if len(upiece) == 0:
                continue
            piece_id = upiece[0]
            piece = board.all_unique_pieces[piece_id]
            x = (index % number_of_figures_per_side) * stride
            y = math.floor(index / number_of_figures_per_side) * stride
            plot_box(ax, piece, (x, y), colors[i])
            index += 1
    plt.savefig(f"{filename}.png", format="png")
    plt.close()


def test():
    triomino = TriominoA()
    plot_piece(triomino)
    plot_all_pieces()
    plot_all_rotations()
    plot_unique_pieces()
    plot_collision_masks()


if __name__ == '__main__':
    test()
