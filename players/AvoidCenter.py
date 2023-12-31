import random
import numpy as np
import board
from players import Player


class AvoidCenterPlayer(Player):
    def __init__(self, board: board.BlokusBoard):
        super().__init__(board)
        self.center = np.array((board.board_size, board.board_size)) / 2

    def choose_move(self, board: board.BlokusBoard, board_point, uid) -> int:
        piece_point = board.unique_piece_id_to_join_point[uid]
        distance_start = np.linalg.norm(self.center - board_point, axis=1)
        shape_all_pieces = board.all_piece_sizes[board.unique_id_to_rotation_id[uid]]
        distance_end = np.linalg.norm(self.center - board_point - piece_point + shape_all_pieces, axis=1)
        smaller_side = np.minimum(distance_start, distance_end)
        idxs = np.flatnonzero(smaller_side == np.max(smaller_side))
        random_index = random.randint(0, idxs.shape[0] - 1)
        return idxs[random_index]
