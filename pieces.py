import numpy as np


# TODO: Need to able to flip all options
class Piece:
    def __init__(self, boundingBoxSize=(1, 1)):
        self.possible_points = None
        self.shape = np.zeros(boundingBoxSize, dtype=bool)
        self.bounding_box_size = np.array(self.shape.shape)
        self.rotate_point = self.bounding_box_size / 2
        self.different90 = False
        self.different180 = False

    # rotates anticlockwise k times
    def rotate(self, k=1):
        k = k % 4
        match (k, self.different90, self.different180):
            case (0, _, _):
                return
            case (1 | 3, False, _):
                return
            case (2, _, False):
                return
        self.shape = np.rot90(self.shape, k=k)
        degrees = k * 90
        angle = np.deg2rad(degrees)
        # TODO: cache this
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = self.rotate_point
        p = self.possible_points
        self.bounding_box_size = np.array(self.shape.shape)
        self.rotate_point = self.bounding_box_size / 2
        self.possible_points = np.rint(((R @ (p - o).T).T + self.rotate_point))


class Monomino(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(1, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])


class Domino(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(2, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (2, 0), (0, 1), (2, 1)])
        self.different90 = True


class TriominoA(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(2, 2))
        self.shape[0, :] = True
        self.shape[:, 0] = True
        self.possible_points = np.array([(0, 0), (2, 0), (2, 1), (1, 2), (0, 2)])
        self.different90 = True
        self.different180 = True


class TriominoB(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (3, 0), (0, 1), (3, 1)])
        self.different90 = True


class TetrominoA(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(4, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (4, 0), (0, 1), (4, 1)])
        self.different90 = True


class TetrominoB(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[:, 0] = True
        self.shape[2, :] = True
        self.possible_points = np.array([(0, 0), (0, 1), (2, 2), (3, 2), (3, 0)])
        self.different90 = True
        self.different180 = True


class TetrominoC(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[0:2, 0] = True
        self.shape[1:3, 1] = True
        self.possible_points = np.array([(0, 0), (0, 1), (1, 2), (3, 2), (3, 1), (2, 0)])
        self.different90 = True
        self.different180 = True


class TetrominoD(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(2, 2))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (2, 0), (0, 2), (2, 2)])


class TetrominoE(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[1, :] = True
        self.shape[:, 0] = True
        self.possible_points = np.array([(0, 0), (0, 1), (1, 2), (2, 2), (3, 1), (3, 0)])
        self.different90 = True
        self.different180 = True


class PentominoF(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[:, 1] = True
        self.shape[1, 0:2] = True
        self.shape[0, 1:3] = True
        self.possible_points = np.array([(1, 0), (0, 1), (0, 3), (1, 3), (3, 2), (3, 1), (2, 0)])
        self.different90 = True
        self.different180 = True


class PentominoI(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(5, 1))
        self.shape[:, :] = True
        self.possible_points = np.array([(0, 0), (5, 0), (0, 1), (5, 1)])
        self.different90 = True


class PentominoL(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(4, 2))
        self.shape[:, 0] = True
        self.shape[0, :] = True
        self.possible_points = np.array([(0, 0), (0, 2), (1, 2), (4, 1), (4, 0)])
        self.different90 = True
        self.different180 = True


class PentominoN(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(5, 2))
        self.shape[0:2, 0] = True
        self.shape[1:5, 1] = True
        self.possible_points = np.array([(0, 0), (0, 1), (1, 2), (5, 2), (5, 1), (2, 0)])
        self.different90 = True
        self.different180 = True


class PentominoP(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[0:2, 0:2] = True
        self.shape[2, 0] = True
        self.possible_points = np.array([(0, 0), (0, 2), (2, 2), (3, 1), (3, 0)])
        self.different90 = True
        self.different180 = True


class PentominoT(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[:, 1] = True
        self.shape[0, :] = True
        self.possible_points = np.array([(0, 0), (0, 3), (1, 3), (3, 2), (3, 1), (1, 0)])
        self.different90 = True
        self.different180 = True


class PentominoU(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 2))
        self.shape[:, 1] = True
        self.shape[0, :] = True
        self.shape[2, :] = True
        self.possible_points = np.array([(0, 0), (0, 2), (3, 2), (3, 0), (2, 0), (1, 0)])
        self.different90 = True
        self.different180 = True


class PentominoV(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[:, 0] = True
        self.shape[2, :] = True
        self.possible_points = np.array([(0, 0), (0, 1), (2, 3), (3, 3), (3, 0)])
        self.different90 = True
        self.different180 = True


class PentominoW(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[0, 0:2] = True
        self.shape[1, 1:3] = True
        self.shape[2, 2] = True
        self.possible_points = np.array([(0, 0), (0, 2), (1, 3), (3, 3), (3, 2), (2, 1), (1, 0)])
        self.different90 = True
        self.different180 = True


class PentominoX(Piece):
    def __init__(self):
        super().__init__(boundingBoxSize=(3, 3))
        self.shape[1, :] = True
        self.shape[:, 1] = True
        self.possible_points = np.array([(1, 0), (0, 1), (0, 2), (1, 3), (2, 3), (3, 2), (3, 1), (2, 0)])


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
        self.different180 = True


def test():
    mono = Monomino()
    mono.rotate(1)

    domino = Domino()
    domino.rotate(3)


if __name__ == '__main__':
    test()
