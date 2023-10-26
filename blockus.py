import copy
import numpy as np
from numba import cuda
import time
from functools import wraps

from pieces import get_all_unique_pieces

MAX_BOUNDING_BOX_FOR_SHAPE = 5
MAX_BOUNDING_BOX_FOR_MASK = 7
THREADS_PER_BLOCK = 1


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


# need to remove the points that dont exist
@cuda.jit(cache=True)
def check_if_piece_possible(
        masking_board,
        full_board,
        all_pieces,
        all_piece_lengths,
        all_masks,
        target_board_points,
        target_piece_points,
        target_piece_ids,
        is_valid_placement
):
    pos = cuda.grid(1)
    if pos < target_piece_ids.shape[0]:
        this_piece = all_pieces[pos]
        this_piece_shape = all_piece_lengths[pos]
        this_piece_mask = all_masks[pos]

        target_board_point = target_board_points[pos]
        target_piece_point = target_piece_points[pos]
        location_of_piece_origin_x = target_board_point[0] - target_piece_point[0]
        location_of_piece_origin_y = target_board_point[1] - target_piece_point[1]
        # out of bound check here
        if location_of_piece_origin_x < 0 or location_of_piece_origin_y < 0:
            is_valid_placement[pos] = False
            return  # check if i should do this return here
        location_of_piece_bbox_x = location_of_piece_origin_x + this_piece_shape[0]
        location_of_piece_bbox_y = location_of_piece_origin_y + this_piece_shape[1]
        if location_of_piece_bbox_x > 20 or location_of_piece_bbox_y > 20:  # fix this constant
            is_valid_placement[pos] = False
            return

        output = True
        for i in range(this_piece_shape[0]):
            for j in range(this_piece_shape[1]):
                is_clash = this_piece[i, j] and full_board[
                    location_of_piece_origin_x + i, location_of_piece_origin_y + j]
                output = output and not is_clash

        for i in range(this_piece_shape[0] + 2):  # +2 for the shape of the mask
            for j in range(this_piece_shape[1] + 2):
                is_clash = this_piece_mask[i, j] and masking_board[
                    location_of_piece_origin_x + i, location_of_piece_origin_y + j]
                output = output and not is_clash
        is_valid_placement[pos] = output


class BlokusBoard:
    def __init__(self, boardSize=20):
        self._player_turn = 0
        self.maskingBoards = np.zeros((4, boardSize + 2, boardSize + 2), dtype=bool)
        self.playerBoards = self.maskingBoards[:, 1:21, 1:21]
        self.full_board = np.zeros((boardSize, boardSize), dtype=np.int64)
        # TODO: can prob cache this if required(to reduce the amount of mem it takes for each match)
        all_unique_pieces = get_all_unique_pieces()
        total_number_shapes = sum(len(v) for v in all_unique_pieces)
        max_length_of_join_points = max(max([r.possible_points.shape[0] for r in p]) for p in all_unique_pieces)
        # keeping the zeroth piece empty for now
        self.all_unique_pieces = np.zeros(
            (total_number_shapes + 1, MAX_BOUNDING_BOX_FOR_SHAPE, MAX_BOUNDING_BOX_FOR_SHAPE), dtype=bool)
        self.all_unique_masks = np.zeros(
            (total_number_shapes + 1, MAX_BOUNDING_BOX_FOR_MASK, MAX_BOUNDING_BOX_FOR_MASK), dtype=bool)
        self.all_piece_sizes = np.zeros((total_number_shapes + 1, 2), dtype=np.int64)
        self.all_mask_shapes = np.zeros((total_number_shapes + 1, 2), dtype=np.int64)
        self.all_join_points = np.zeros((total_number_shapes + 1, max_length_of_join_points, 2), dtype=np.int64)
        self.all_join_points_size = np.zeros(total_number_shapes + 1, dtype=np.int64)

        index = 1
        all_pieces = []
        for piece in all_unique_pieces:
            all_rotations = []
            for rotation in piece:
                piece_shape = rotation.shape.shape
                mask_shape = rotation.collision_mask.shape
                join_points_shape = rotation.possible_points.shape[0]
                self.all_unique_pieces[index, :piece_shape[0], :piece_shape[1]] = rotation.shape
                self.all_unique_masks[index, :mask_shape[0], :mask_shape[1]] = rotation.collision_mask
                self.all_piece_sizes[index] = rotation.bounding_box_size
                # fix this later
                self.all_mask_shapes[index] = rotation.bounding_box_size + 2
                self.all_join_points[index, :join_points_shape, :] = rotation.possible_points
                self.all_join_points_size[index] = join_points_shape
                all_rotations.append(index)
                index += 1
            all_pieces.append(all_rotations)

        self.available_pieces_per_player = [all_pieces]
        self.available_pieces_per_player.extend([copy.deepcopy(self.available_pieces_per_player[0]) for _ in range(3)])
        # maybe this should be a list, we need to iterate and REMOVE items from the list after we place an item at that location
        self.positions_available_per_player = [
            np.array(((-3, -3),)),
            np.array(((20, 0),)),
            np.array(((20, 20),)),
            np.array(((0, 20),))
        ]
        # how it should work when placing a block, remove point where we place the block. add all the other points of that block to this list(after adding the 'origin' of the block to all the vectors),
        self.max_length_of_join_points = max_length_of_join_points

    def player_turn(self):
        return self._player_turn

    def check_if_move_valid(self):
        pass

    @timeit
    def current_player_get_all_valid_moves(self):
        number_of_pieces_available = sum(len(v) for v in self.available_pieces_per_player[self._player_turn])
        possible_points = self.positions_available_per_player[self._player_turn]
        number_possible_points = possible_points.shape[0]
        number_of_possible_positions_per_point = number_of_pieces_available * self.max_length_of_join_points
        target_piece_ids = np.zeros(number_of_possible_positions_per_point, dtype=np.int64)
        target_join_points = np.zeros((number_of_possible_positions_per_point, 2), dtype=np.int64)
        index = 0
        for piece in self.available_pieces_per_player[self._player_turn]:
            for rotation in piece:
                num_of_join_points = self.all_join_points_size[rotation]
                start_target_location = index
                end_target_location = index + num_of_join_points
                target_piece_ids[start_target_location:end_target_location] = rotation
                target_join_points[start_target_location:end_target_location, :] = self.all_join_points[rotation][
                                                                                   0:num_of_join_points]
                index += num_of_join_points
        max_target_join_points = index
        target_piece_ids = target_piece_ids[:max_target_join_points]
        target_join_points = target_join_points[:max_target_join_points, :]
        target_board_points = np.repeat(possible_points, max_target_join_points, axis=0).astype(np.int64)
        all_target_piece_ids = np.concatenate([target_piece_ids] * number_possible_points, axis=0)
        target_piece_points = np.concatenate([[target_join_points]] * number_possible_points, axis=0).reshape(
            (number_possible_points * max_target_join_points, 2))
        is_valid_placement = np.ones(max_target_join_points, dtype=bool)
        blocks_per_grid = (all_target_piece_ids.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
        check_if_piece_possible[blocks_per_grid, THREADS_PER_BLOCK](
            self.maskingBoards[self._player_turn],
            self.full_board,
            self.all_unique_pieces,
            self.all_piece_sizes,
            self.all_unique_masks,
            target_board_points,
            target_piece_points,
            target_piece_ids,
            is_valid_placement
        )

    # need define input and output
    def current_player_submit_move(self):
        # remove possible points if one of the location(x or y) is 0, nothing can connect there
        pass

    def visualize_board(self):
        pass

    # for the first turn can check just by making sure there is a block on the (0,0) point(check this)
    def check_if_moves_possible(self):
        pass


def check():
    board = BlokusBoard()
    board.current_player_get_all_valid_moves()
    board.current_player_get_all_valid_moves()
    board.current_player_get_all_valid_moves()


if __name__ == '__main__':
    check()
