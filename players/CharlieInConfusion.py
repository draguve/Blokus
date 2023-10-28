import random

import numpy as np

import blokus
from players import Player


class CharlieInConfusion(Player):
    def __init__(self, board: blokus.BlokusBoard):
        super().__init__(board)
        piece_sizes = np.zeros(board.all_unique_pieces.shape[0])
        for i in range(board.all_unique_pieces.shape[0]):
            piece_sizes[i] = np.sum(board.all_unique_pieces[i])
        self.piece_sizes = piece_sizes
        self.center = np.array((board.board_size, board.board_size)) / 2

    def choose_move(self, board, board_point, piece_id, piece_point) -> int:
        these_option_sizes = self.piece_sizes[piece_id]
        idxs = np.flatnonzero(these_option_sizes[:] == np.max(these_option_sizes))

        distance_start = np.linalg.norm(self.center - board_point, axis=1)
        shape_all_pieces = board.all_piece_sizes[piece_id]
        distance_end = np.linalg.norm(self.center - board_point - piece_point + shape_all_pieces, axis=1)
        average_side = (distance_start+distance_end)/2
        best_ones = np.flatnonzero(average_side[idxs] == np.min(average_side[idxs]))
        random_index = random.randint(0, best_ones.shape[0] - 1)
        return idxs[best_ones[random_index]]
