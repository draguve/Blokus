import copy
# import cupy as cp
# from cupyx.scipy import signal
import numpy as np
from scipy import signal

collision_filter = np.zeros((3, 3), dtype=np.uint8)
collision_filter[:, 1] = True
collision_filter[1, :] = True

rotation_matrices = np.zeros((4, 2, 2), dtype=np.float32)
for k in range(4):
    degrees = k * 90
    angle = np.deg2rad(degrees)
    rotation_matrices[k, :, :] = np.array([[np.cos(angle), -np.sin(angle)],
                                           [np.sin(angle), np.cos(angle)]])


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class Piece(metaclass=PostInitCaller):
    def __init__(self, boundingBoxSize=(1, 1)):
        self.collision_mask = None
        self.possible_points = None
        self.shape = np.zeros(boundingBoxSize, dtype=bool)
        self.bounding_box_size = np.array(self.shape.shape)
        self.rotate_point = self.bounding_box_size / 2
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = True

    def __post_init__(self):
        self.collision_mask = signal.convolve2d(self.shape, collision_filter).astype(bool)
        self.possible_points = self.possible_points.astype(np.int8)

    def is_unique(self, k, is_flipped):
        if is_flipped:
            if self.differentFlip:
                return self._is_unique_angle(angle=k)
            return False
        return self._is_unique_angle(angle=k)

    def _is_unique_angle(self, angle):
        match (angle, self.different90, self.different180, self.different270):
            case (0 | 2, _, False, _):
                return False
            case (1, False, _, _):
                return False
            case (3, _, _, False):
                return False
        return True

    # rotates anticlockwise k times
    def rotate(self, k=1):
        k = k % 4
        self.shape = np.rot90(self.shape, k=k)
        self.collision_mask = np.rot90(self.collision_mask, k=k)
        R = rotation_matrices[k, :, :]
        o = self.rotate_point
        p = self.possible_points
        self.bounding_box_size = np.array(self.shape.shape)
        self.rotate_point = self.bounding_box_size / 2
        self.possible_points = np.rint(((R @ (p - o).T).T + self.rotate_point)).astype(np.int8)

    def flip(self):
        self.shape = np.fliplr(self.shape)
        self.collision_mask = np.fliplr(self.collision_mask)
        points = np.zeros((self.possible_points.shape[0], 3))
        points[:, 0:2] = self.possible_points
        o = np.zeros((3,))
        o[0:2] = self.rotate_point
        R = np.array([[1, 0, 0, ],
                      [0, -1, 0, ],
                      [0, 0, -1]])
        self.possible_points = np.rint((R @ (points - o).T).T + o)[:, 0:2].astype(np.int8)


class Monomino(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(1, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])
        self.differentFlip = False
        self.different90 = False
        self.different180 = False
        self.different270 = False


class Domino(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(2, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (2, 0), (0, 1), (2, 1)])
        self.different90 = True
        self.different180 = False
        self.different270 = False
        self.differentFlip = False


class TriominoA(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(2, 2))
        self.shape[0, :] = True
        self.shape[:, 0] = True
        self.possible_points = np.array([(0, 0), (2, 0), (2, 1), (1, 2), (0, 2)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = False


class TriominoB(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (3, 0), (0, 1), (3, 1)])
        self.different90 = True
        self.different180 = False
        self.different270 = False
        self.differentFlip = False


class TetrominoA(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(4, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (4, 0), (0, 1), (4, 1)])
        self.different90 = True
        self.different180 = False
        self.different270 = False
        self.differentFlip = False


class TetrominoB(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[:, 0] = True
        self.shape[2, :] = True
        self.possible_points = np.array([(0, 0), (0, 1), (2, 2), (3, 2), (3, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = True


class TetrominoC(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[0:2, 0] = True
        self.shape[1:3, 1] = True
        self.possible_points = np.array([(0, 0), (0, 1), (1, 2), (3, 2), (3, 1), (2, 0)])
        self.different90 = True
        self.different180 = False
        self.different270 = False
        self.differentFlip = True


class TetrominoD(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(2, 2))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (2, 0), (0, 2), (2, 2)])
        self.differentFlip = False
        self.different90 = False
        self.different180 = False
        self.different270 = False


class TetrominoE(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[1, :] = True
        self.shape[:, 0] = True
        self.possible_points = np.array([(0, 0), (0, 1), (1, 2), (2, 2), (3, 1), (3, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = False


class PentominoF(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[:, 1] = True
        self.shape[1, 0:2] = True
        self.shape[0, 1:3] = True
        self.possible_points = np.array([(1, 0), (0, 1), (0, 3), (1, 3), (3, 2), (3, 1), (2, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = True


class PentominoI(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(5, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (5, 0), (0, 1), (5, 1)])
        self.different90 = True
        self.different180 = False
        self.different270 = False
        self.differentFlip = False


class PentominoL(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(4, 2))
        self.shape[:, 0] = True
        self.shape[0, :] = True
        self.possible_points = np.array([(0, 0), (0, 2), (1, 2), (4, 1), (4, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = True


class PentominoN(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(4, 2))
        self.shape[0:2, 0] = True
        self.shape[1:4, 1] = True
        self.possible_points = np.array([(0, 0), (0, 1), (1, 2), (4, 2), (4, 1), (2, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = True


class PentominoP(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[0:2, 0:2] = True
        self.shape[2, 0] = True
        self.possible_points = np.array([(0, 0), (0, 2), (2, 2), (3, 1), (3, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = True


class PentominoT(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[:, 1] = True
        self.shape[0, :] = True
        self.possible_points = np.array([(0, 0), (0, 3), (1, 3), (3, 2), (3, 1), (1, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = False


class PentominoU(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[:, 1] = True
        self.shape[0, :] = True
        self.shape[2, :] = True
        self.possible_points = np.array([(0, 0), (0, 2), (3, 2), (3, 0), (2, 0), (1, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = False


class PentominoV(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[:, 0] = True
        self.shape[2, :] = True
        self.possible_points = np.array([(0, 0), (0, 1), (2, 3), (3, 3), (3, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = False


class PentominoW(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[0, 0:2] = True
        self.shape[1, 1:3] = True
        self.shape[2, 2] = True
        self.possible_points = np.array([(0, 0), (0, 2), (1, 3), (3, 3), (3, 2), (2, 1), (1, 0)])
        self.different90 = True
        self.different180 = True
        self.different270 = True
        self.differentFlip = False


class PentominoX(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[1, :] = True
        self.shape[:, 1] = True
        self.possible_points = np.array([(1, 0), (0, 1), (0, 2), (1, 3), (2, 3), (3, 2), (3, 1), (2, 0)])
        self.different90 = False
        self.different180 = False
        self.different270 = False
        self.differentFlip = False


class PentominoY(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(4, 2))
        self.shape[2, :] = True
        self.shape[:, 0] = True
        self.possible_points = np.array([(0, 0), (0, 1), (2, 2), (3, 2), (4, 1), (4, 0)])
        self.different90 = True
        self.different180 = True


class PentominoZ(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[:, 1] = True
        self.shape[0, 0:2] = True
        self.shape[2, 1:3] = True
        self.possible_points = np.array([(0, 0), (0, 2), (2, 3), (3, 3), (3, 1), (1, 0)])
        self.different90 = True
        self.different180 = False
        self.different270 = False
        self.differentFlip = True


def get_all_pieces():
    return [Monomino(), Domino(), TriominoA(), TriominoB(), TetrominoA(), TetrominoB(), TetrominoC(), TetrominoD(),
            TetrominoE(), PentominoF(), PentominoI(), PentominoL(), PentominoN(), PentominoP(), PentominoT(),
            PentominoU(), PentominoV(), PentominoW(), PentominoX(), PentominoY(), PentominoZ()]


def get_all_unique_pieces():
    unique_pieces = []
    all_pieces = get_all_pieces()
    for idx, piece in enumerate(all_pieces):
        combinations = [piece]
        for i in range(1, 4):
            should_draw = piece.is_unique(k=i, is_flipped=False)
            if not should_draw:
                continue
            rot_piece = copy.deepcopy(piece)
            rot_piece.rotate(i)
            combinations.append(rot_piece)
        flipped = copy.deepcopy(piece)
        flipped.flip()
        if piece.differentFlip:
            combinations.append(flipped)
            for i in range(1, 4):
                should_draw = piece.is_unique(k=i, is_flipped=True)
                if not should_draw:
                    continue
                rot_piece = copy.deepcopy(flipped)
                rot_piece.rotate(i)
                combinations.append(rot_piece)
        unique_pieces.append(combinations)
    return unique_pieces


def test():
    mono = Monomino()
    mono.rotate()


if __name__ == '__main__':
    test()
