from mcts import MCTS
from games.blokus import Game

class SelfPlay:
    def __init__(self):
        self.mcts = MCTS(
            780,
            32,
            800
        )
        self.game = Game()
