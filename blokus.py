import copy
import random
from numba import typed

import numba
import numpy as np
from numba import cuda, uint8
from numba import jit
from util import timeit

from pieces import get_all_unique_pieces
import visualizer

MAX_BOUNDING_BOX_FOR_SHAPE = 5
MAX_BOUNDING_BOX_FOR_MASK = 7
THREADS_PER_BLOCK = 1


@jit(nopython=True, parallel=True)
def is_point_surrounded(player_board, point):
    return np.sum(player_board[point[0] - 1:point[0] + 1, point[1] - 1:point[1] + 1]) > 1


# one issue is the surrounded technique does not work in the case when a vertex gets surrounded after placing the new tile and only get removed after the next iteration
@jit(nopython=True, parallel=True)
def update_open_points(new_open_points, current_open_vertexes, board_point, piece_origin, current_board,
                       new_piece_join_vertexes, piece_vertex):
    index_new_vertexes = 0
    found = False
    number_possible_vertexes = current_open_vertexes.shape[0]
    board_shape = current_board.shape
    for i in range(number_possible_vertexes):
        if np.all(current_open_vertexes[i] == board_point):
            found = True
        else:
            if is_point_surrounded(current_board, current_open_vertexes[i]):
                continue
            new_open_points[index_new_vertexes] = current_open_vertexes[i]
            index_new_vertexes += 1
    if not found:
        return False, None
    number_new_piece_join_vertexes = new_piece_join_vertexes.shape[0]
    possible_join_vertexes = new_piece_join_vertexes + piece_origin
    piece_vertex_where_added = piece_vertex + piece_origin
    for i in range(number_new_piece_join_vertexes):
        if np.all(possible_join_vertexes[i] == piece_vertex_where_added):
            continue
        if possible_join_vertexes[i, 0] == 0 or possible_join_vertexes[i, 1] == 0:
            continue
        if possible_join_vertexes[i, 0] == board_shape[0] or possible_join_vertexes[i, 1] == board_shape[1]:
            continue
        if is_point_surrounded(current_board, possible_join_vertexes[i]):
            continue
        new_open_points[index_new_vertexes] = possible_join_vertexes[i]
        index_new_vertexes += 1
    return True, new_open_points[:index_new_vertexes]


@jit(nopython=True)
def check_if_piece_possible(
        masking_board,
        full_board,
        all_pieces,
        all_piece_lengths,
        all_masks,
        target_board_vertexes,
        target_piece_vertexes,
        target_piece_ids,
        is_valid_placement
):
    for pos in range(is_valid_placement.shape[0]):
        id = target_piece_ids[pos]
        this_piece = all_pieces[id]
        this_piece_shape = all_piece_lengths[id]
        this_piece_mask = all_masks[id]

        target_board_point = target_board_vertexes[pos]
        target_piece_point = target_piece_vertexes[pos]
        location_of_piece_origin_x = target_board_point[0] - target_piece_point[0]
        location_of_piece_origin_y = target_board_point[1] - target_piece_point[1]
        # out of bound check here
        if location_of_piece_origin_x < 0 or location_of_piece_origin_y < 0:
            is_valid_placement[pos] = False
            continue
        location_of_piece_bbox_x = location_of_piece_origin_x + this_piece_shape[0]
        location_of_piece_bbox_y = location_of_piece_origin_y + this_piece_shape[1]
        if location_of_piece_bbox_x > 20 or location_of_piece_bbox_y > 20:  # fix this constant
            is_valid_placement[pos] = False
            continue

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
        self.board_size = boardSize
        self.maskingBoards = np.zeros((4, boardSize + 2, boardSize + 2), dtype=bool)
        self.playerBoards = self.maskingBoards[:, 1:21, 1:21]
        self.full_board = np.zeros((boardSize, boardSize), dtype=np.int8)
        all_unique_pieces = get_all_unique_pieces()
        total_number_shapes = sum(len(v) for v in all_unique_pieces)
        max_length_of_join_points = max(max([r.possible_points.shape[0] for r in p]) for p in all_unique_pieces)
        # keeping the zeroth piece empty for now
        self.all_unique_pieces = np.zeros(
            (total_number_shapes + 1, MAX_BOUNDING_BOX_FOR_SHAPE, MAX_BOUNDING_BOX_FOR_SHAPE), dtype=bool)
        self.all_unique_masks = np.zeros(
            (total_number_shapes + 1, MAX_BOUNDING_BOX_FOR_MASK, MAX_BOUNDING_BOX_FOR_MASK), dtype=bool)
        self.all_piece_sizes = np.zeros((total_number_shapes + 1, 2), dtype=np.int8)
        self.all_mask_sizes = np.zeros((total_number_shapes + 1, 2), dtype=np.int8)
        self.all_join_points = np.zeros((total_number_shapes + 1, max_length_of_join_points, 2), dtype=np.int8)
        self.all_join_points_length = np.zeros(total_number_shapes + 1, dtype=np.int8)
        self.piece_id_to_unique_piece_index = {}
        index = 1
        u_index = 0
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
                self.all_mask_sizes[index] = rotation.bounding_box_size + 2
                self.all_join_points[index, :join_points_shape, :] = rotation.possible_points
                self.all_join_points_length[index] = join_points_shape
                all_rotations.append(index)
                self.piece_id_to_unique_piece_index[index] = u_index
                index += 1
            all_pieces.append(all_rotations)

            u_index += 1

        self.available_pieces_per_player = [all_pieces]
        self.available_pieces_per_player.extend([copy.deepcopy(self.available_pieces_per_player[0]) for _ in range(3)])
        # maybe this should be a list, we need to iterate and REMOVE items from the list after we place an item at that location
        self.open_board_points = [
            np.array(((0, 0),)),
            np.array(((20, 0),)),
            np.array(((20, 20),)),
            np.array(((0, 20),)),
        ]
        # how it should work when placing a block, remove point where we place the block. add all the other points of that block to this list(after adding the 'origin' of the block to all the vectors),
        self.max_length_of_join_points = max_length_of_join_points

    def player_turn(self):
        return self._player_turn

    def check_if_move_valid(self, board_point: np.array, piece_id: 0, piece_point: np.array):
        piece_id = np.array((piece_id,), dtype=np.int8)
        # piece_point = self.all_join_points[piece_id, piece_point_id]
        is_valid_placement = np.ones((1, 1), dtype=bool)
        check_if_piece_possible(
            self.maskingBoards[self._player_turn],
            self.full_board,
            self.all_unique_pieces,
            self.all_piece_sizes,
            self.all_unique_masks,
            np.atleast_2d(board_point),
            np.atleast_2d(piece_point),
            piece_id,
            is_valid_placement
        )
        return is_valid_placement[0, 0]

    # @timeit
    def current_player_get_all_valid_moves(self):
        current_player = self._player_turn

        number_of_pieces_available = sum(len(v) for v in self.available_pieces_per_player[current_player])
        possible_points = self.open_board_points[current_player]
        number_possible_points = possible_points.shape[0]
        if number_possible_points == 0:
            return None, None, None
        number_of_possible_positions_per_point = number_of_pieces_available * self.max_length_of_join_points
        target_piece_ids = np.zeros(number_of_possible_positions_per_point, dtype=np.int8)
        target_join_points = np.zeros((number_of_possible_positions_per_point, 2), dtype=np.int8)
        index = 0
        for piece in self.available_pieces_per_player[self._player_turn]:
            for rotation in piece:
                num_of_join_points = self.all_join_points_length[rotation]
                start_target_location = index
                end_target_location = index + num_of_join_points
                target_piece_ids[start_target_location:end_target_location] = rotation
                target_join_points[start_target_location:end_target_location, :] = self.all_join_points[rotation][
                                                                                   0:num_of_join_points]
                index += num_of_join_points
        max_target_join_points = index
        target_piece_ids = target_piece_ids[:max_target_join_points]
        target_join_points = target_join_points[:max_target_join_points, :]
        target_board_points = np.repeat(possible_points, max_target_join_points, axis=0).astype(np.int8)
        all_target_piece_ids = np.concatenate([target_piece_ids] * number_possible_points, axis=0)
        target_piece_points = np.concatenate([[target_join_points]] * number_possible_points, axis=0).reshape(
            (number_possible_points * max_target_join_points, 2))
        is_valid_placement = np.ones(max_target_join_points * number_possible_points, dtype=bool)
        check_if_piece_possible(
            self.maskingBoards[self._player_turn],
            self.full_board,
            self.all_unique_pieces,
            self.all_piece_sizes,
            self.all_unique_masks,
            target_board_points,
            target_piece_points,
            all_target_piece_ids,
            is_valid_placement
        )
        valid_results = is_valid_placement.nonzero()[0]
        if len(valid_results) < 1:
            return None, None, None
        return target_board_points[valid_results], all_target_piece_ids[valid_results], target_piece_points[
            valid_results]

    # need define input and output
    def current_player_submit_move(self, board_point: np.array, piece_id: int, piece_point_location: np.array):
        current_player = self._player_turn

        piece_shape = self.all_piece_sizes[piece_id]
        piece_origin = board_point - piece_point_location
        piece_bb_end = piece_origin + piece_shape
        if not self.check_if_move_valid(board_point, piece_id, piece_point_location):
            print("Submitted move found to be invalid????")
            return False
        current_open_points = self.open_board_points[current_player]
        current_num_of_positions = self.open_board_points[current_player].shape[
                                       0] - 1  # minus the one we are gonna take away
        this_piece_add_num_positions = self.all_join_points_length[piece_id] - 1
        max_new_points = np.zeros((current_num_of_positions + this_piece_add_num_positions, 2), dtype=np.uint8)
        new_piece_join_points = self.all_join_points[piece_id][0:self.all_join_points_length[piece_id]]
        update_success, new_piece_join_points = update_open_points(max_new_points, current_open_points, board_point,
                                                                   piece_origin,
                                                                   self.playerBoards[current_player],
                                                                   new_piece_join_points, piece_point_location)
        if not update_success:
            print("Could not find board point as valid??????")
            return False
        self.open_board_points[current_player] = new_piece_join_points
        self.playerBoards[self._player_turn, piece_origin[0]:piece_bb_end[0], piece_origin[1]:piece_bb_end[1]] |= \
            self.all_unique_pieces[piece_id][0:piece_shape[0], 0:piece_shape[1]]
        self.full_board[self.playerBoards[self._player_turn]] = self._player_turn + 1

        u_index = self.piece_id_to_unique_piece_index[piece_id]
        self.available_pieces_per_player[self._player_turn][u_index] = []
        self._player_turn = (self._player_turn + 1) % 4
        return True

    def current_player_skip_turn(self):
        self._player_turn = (self._player_turn + 1) % 4

# def check():
#     board = BlokusBoard()
#     for i in range(10):
#         results = board.current_player_get_all_valid_moves()
#         if results[0] is None:
#             board.current_player_skip_turn()
#             continue
#         rand = random.randint(0, results[1].shape[0] - 1)
#         board.current_player_submit_move(results[0][rand], results[1][rand], results[2][rand])
#         # visualizer.plot_store_board(board, f"board_{i}")
#
#
# if __name__ == '__main__':
#     check()
