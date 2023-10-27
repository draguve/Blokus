import random

from players import Player


class RandomPlayer(Player):
    def __init__(self, board):
        super().__init__(board)

    def choose_move(self, board, board_point, piece_id, piece_point_id) -> int:
        rand = random.randint(0, board_point.shape[0] - 1)
        return rand