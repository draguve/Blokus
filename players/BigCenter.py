import random

import numpy as np

import blokus
from players import Player


class BigCenterPlayer(Player):
    def __init__(self, board: blokus.BlokusBoard):
        super().__init__(board)
        piece_sizes = np.zeros(board.all_unique_pieces.shape[0])
        for i in range(board.all_unique_pieces.shape[0]):
            piece_sizes[i] = np.sum(board.all_unique_pieces[i])
        self.piece_sizes = piece_sizes
        self.center = np.array((board.board_size, board.board_size)) / 2

    def choose_move(self, board, board_point, uid) -> int:
        piece_point = board.unique_piece_id_to_join_point[uid]
        these_option_sizes = self.piece_sizes[board.unique_id_to_rotation_id[uid]]
        idxs = np.flatnonzero(these_option_sizes[:] == np.max(these_option_sizes))

        distance_start = np.linalg.norm(self.center - board_point, axis=1)
        shape_all_pieces = board.all_piece_sizes[board.unique_id_to_rotation_id[uid]]
        distance_end = np.linalg.norm(self.center - board_point - piece_point + shape_all_pieces, axis=1)
        longer_side = np.maximum(distance_start, distance_end)
        best_ones = np.flatnonzero(longer_side[idxs] == np.min(longer_side[idxs]))
        random_index = random.randint(0, best_ones.shape[0] - 1)
        return idxs[best_ones[random_index]]
