class Player:
    def __init__(self):
        raise NotImplementedError

    def choose_move(self, board, board_point, piece_id, piece_point_id) -> int:
        raise NotImplementedError
