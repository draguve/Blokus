import numpy as np


class BlokusBoard:
    def __init__(self, boardSize=20):
        self._player_turn = 0
        self.playerBoards = np.zeros((4, boardSize, boardSize), dtype=bool)

    def _player_turn(self):
        return self._player_turn
