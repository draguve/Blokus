import random
import numpy as np
import blokus
from players import Player


class AimCenterPlayer(Player):
    def __init__(self, board: blokus.BlokusBoard):
        super().__init__(board)
        self.center = np.array((board.board_size, board.board_size)) / 2

    def choose_move(self, board: blokus.BlokusBoard, board_point, piece_id, piece_point) -> int:
        distance_start = np.linalg.norm(self.center - board_point, axis=1)
        shape_all_pieces = board.all_piece_sizes[piece_id]
        distance_end = np.linalg.norm(self.center - board_point - piece_point + shape_all_pieces, axis=1)
        longer_side = np.maximum(distance_start, distance_end)
        idxs = np.flatnonzero(longer_side == np.min(longer_side))
        random_index = random.randint(0, idxs.shape[0] - 1)
        return idxs[random_index]
