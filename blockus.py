import copy
import numpy as np

from pieces import get_all_unique_pieces


class BlokusBoard:
    def __init__(self, boardSize=20):
        self._player_turn = 0
        self.playerBoards = np.zeros((4, boardSize, boardSize), dtype=bool)
        self.full_board = np.zeros((boardSize, boardSize), dtype=np.uint8)
        # TODO: can prob cache this if required(to reduce the amount of mem it takes for each match)
        self.available_pieces_per_player = [get_all_unique_pieces()]
        self.available_pieces_per_player.extend([copy.deepcopy(self.available_pieces_per_player[0]) for _ in range(3)])
        # maybe this should be a list, we need to iterate and REMOVE items from the list after we place an item at that location
        self.positions_available_per_player = [
            np.array(((0, 0),)),
            np.array(((20, 0),)),
            np.array(((20, 20),)),
            np.array(((0, 20),))
        ]
        # how it should work when placing a block, remove point where we place the block. add all the other points of that block to this list(after adding the 'origin' of the block to all the vectors),
        pass

    def _player_turn(self):
        return self._player_turn

    # to check if a piece
    def _check_collide_with_wall(self):
        pass

    def check_if_move_valid(self):
        pass

    def current_player_get_all_valid_moves(self):
        pass

    # need define input and output
    def current_player_submit_move(self):
        pass

    def visualize_board(self):
        pass

    # for the first turn can check just by making sure there is a block on the (0,0) point(check this)
    def check_if_moves_possible(self):
        pass


def test():
    board = BlokusBoard()


if __name__ == '__main__':
    test()
