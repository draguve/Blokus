import hashlib

class Player:
    def __init__(self, board):
        pass

    # TODO: can remove passing the board everytime
    def choose_move(self, board, board_point,uid) -> int:
        raise NotImplementedError

    def get_player_id(self):
        return str(hashlib.md5(type(self).__name__.encode('utf-8')).hexdigest())
