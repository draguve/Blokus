import numpy as np


class Piece:
    def __init__(self, boundingBoxSize=(1, 1)):
        self.possible_points = None
        self.shape = np.zeros(boundingBoxSize, dtype=bool)
        self.bounding_box_size = np.array(self.shape.shape)
        self.rotate_point = self.bounding_box_size / 2
        self.different90 = False
        self.different180 = False

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


def test():
    mono = Monomino()
    mono.rotate(1)

    domino = Domino()
    domino.rotate(3)


if __name__ == '__main__':
    test()
