import random

import numpy as np

import board
from players import Player


class SmallestFirstPlayer(Player):
    def __init__(self, board: board.BlokusBoard):
        super().__init__(board)
        piece_sizes = np.zeros(board.all_unique_pieces.shape[0])
        for i in range(board.all_unique_pieces.shape[0]):
            piece_sizes[i] = np.sum(board.all_unique_pieces[i])
        self.piece_sizes = piece_sizes

    def choose_move(self, board, board_point, uid) -> int:
        these_option_sizes = self.piece_sizes[board.unique_id_to_rotation_id[uid]]
        idxs = np.flatnonzero(these_option_sizes[:] == np.min(these_option_sizes))
        random_index = random.randint(0, idxs.shape[0] - 1)
        return idxs[random_index]
