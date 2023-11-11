import copy
import random
from numba import typed, njit

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


@njit(cache=True)
def is_point_surrounded(player_board, point):
    return np.sum(player_board[point[0] - 1:point[0] + 1, point[1] - 1:point[1] + 1]) > 1


@njit(cache=True)
def get_available_uids(all_pieces):
    number_points = 0
    for i, value in np.ndenumerate(all_pieces):
        number_points += value
    index = 0
    player_available_uids = np.zeros(number_points, dtype=np.int32)
    for (i,), value in np.ndenumerate(all_pieces):
        if value:
            player_available_uids[index] = i
            index += 1
    return player_available_uids


@njit(cache=True)
def get_open_points_from_board(player_open_positions):
    number_points = 0
    for (i, j), i_ij in np.ndenumerate(player_open_positions):
        number_points += i_ij
    index = 0
    player_open_points = np.zeros((number_points, 2), dtype=np.int8)
    for (i, j), i_ij in np.ndenumerate(player_open_positions):
        if i_ij:
            player_open_points[index][0] = i
            player_open_points[index][1] = j
            index += 1
    return player_open_points


@njit(cache=True)  # TODO: maybe check if surrounded by the full board to cull cases where surrounded by enemy pieces
def update_open_positions(
        player_open_positions,
        new_piece_board_point,
        new_piece_uid,

        unique_id_to_rotation_id,
        all_piece_sizes,
        unique_piece_id_to_join_point,
        all_join_points,
        all_join_points_lengths,
        current_player_board
):
    board_shape = current_player_board.shape

    rotation_id = unique_id_to_rotation_id[new_piece_uid]
    piece_shape = all_piece_sizes[rotation_id]
    piece_point_location = unique_piece_id_to_join_point[new_piece_uid]

    piece_origin = new_piece_board_point - piece_point_location
    piece_bb_end = piece_origin + piece_shape
    new_piece_join_points = all_join_points[rotation_id][0:all_join_points_lengths[rotation_id]]

    number_new_piece_join_vertexes = new_piece_join_points.shape[0]
    possible_join_vertexes = new_piece_join_points + piece_origin
    piece_vertex_where_added = piece_point_location + piece_origin

    for i in range(number_new_piece_join_vertexes):
        if possible_join_vertexes[i, 0] == 0 or possible_join_vertexes[i, 1] == 0:
            continue
        if possible_join_vertexes[i, 0] == board_shape[0] or possible_join_vertexes[i, 1] == board_shape[1]:
            continue
        player_open_positions[possible_join_vertexes[i][0], possible_join_vertexes[i][1]] = True

    player_open_positions[piece_vertex_where_added[0], piece_vertex_where_added[1]] = False
    player_open_positions[new_piece_board_point[0], new_piece_board_point[1]] = False

    possible_points = get_open_points_from_board(player_open_positions)
    number_possible_points = possible_points.shape[0]

    for i in range(number_possible_points):
        if is_point_surrounded(current_player_board, possible_points[i]):
            player_open_positions[possible_points[i][0], possible_points[i][1]] = False


@njit(cache=True)
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


@jit(forceobj=True, cache=True)  # maybe fix this later has an issue with the repeat and the list multiplication
def get_all_valid_moves(
        current_player,
        available_uids_per_player,
        open_board_locations,
        unique_id_to_rotation_id,
        unique_piece_id_to_join_point,
        masking_boards,
        full_board,
        all_unique_pieces,
        all_piece_sizes,
        all_unique_masks,
):
    # possible_uids = np.array(available_uids_per_player[current_player].nonzero()[0])
    possible_uids = get_available_uids(available_uids_per_player[current_player])
    number_possible_uids = possible_uids.shape[0]

    possible_points = get_open_points_from_board(open_board_locations[current_player])
    number_possible_points = possible_points.shape[0]

    if number_possible_points == 0:
        return None, None

    target_board_points = np.repeat(possible_points, number_possible_uids, axis=0).astype(np.int8)
    target_uids = np.concatenate([possible_uids] * number_possible_points, axis=0)
    target_rotation_ids = unique_id_to_rotation_id[target_uids]
    target_join_points = unique_piece_id_to_join_point[target_uids]
    is_valid_placement = np.ones(number_possible_uids * number_possible_points, dtype=bool)

    check_if_piece_possible(
        masking_boards[current_player],
        full_board,
        all_unique_pieces,
        all_piece_sizes,
        all_unique_masks,
        target_board_points,
        target_join_points,
        target_rotation_ids,
        is_valid_placement
    )
    valid_results = is_valid_placement.nonzero()[0]
    if len(valid_results) < 1:
        return None, None
    return target_board_points[valid_results], target_uids[valid_results]


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

        self.unique_id_to_piece_id = np.zeros((total_number_shapes * max_length_of_join_points), dtype=int)
        self.unique_id_to_rotation_id = np.zeros((total_number_shapes * max_length_of_join_points), dtype=int)
        self.unique_piece_id_to_join_point = np.zeros((total_number_shapes * max_length_of_join_points, 2), dtype=int)
        self.piece_id_to_uid_start_stop = np.zeros((len(all_unique_pieces), 2), dtype=int)
        self.piece_id_to_rotation_id = np.zeros(len(all_unique_pieces), dtype=int)

        rotation_id_index = 1  # 1-100ish one for each rotation
        piece_id_index = 0  # 0-21
        unique_id_index = 0  # 0-800ish every combination of piece, rotation, and point
        for piece in all_unique_pieces:

            self.piece_id_to_uid_start_stop[piece_id_index, 0] = unique_id_index

            for rotation in piece:
                piece_shape = rotation.shape.shape
                mask_shape = rotation.collision_mask.shape

                number_of_join_points = rotation.possible_points.shape[0]

                self.all_unique_pieces[rotation_id_index, :piece_shape[0], :piece_shape[1]] = rotation.shape
                self.all_unique_masks[rotation_id_index, :mask_shape[0], :mask_shape[1]] = rotation.collision_mask
                self.all_piece_sizes[rotation_id_index] = rotation.bounding_box_size
                self.all_mask_sizes[rotation_id_index] = rotation.bounding_box_size + 2

                start_unique_id = unique_id_index
                stop_unique_id = unique_id_index + number_of_join_points

                self.all_join_points[rotation_id_index, :number_of_join_points, :] = rotation.possible_points
                self.all_join_points_length[rotation_id_index] = number_of_join_points

                self.unique_piece_id_to_join_point[start_unique_id:stop_unique_id] = rotation.possible_points
                self.unique_id_to_piece_id[start_unique_id:stop_unique_id] = piece_id_index
                self.unique_id_to_rotation_id[start_unique_id:stop_unique_id] = rotation_id_index
                self.piece_id_to_rotation_id[piece_id_index] = rotation_id_index

                rotation_id_index += 1
                unique_id_index = stop_unique_id

            self.piece_id_to_uid_start_stop[piece_id_index, 1] = unique_id_index
            piece_id_index += 1

        self.total_number_of_uids = unique_id_index
        self.unique_id_to_piece_id = self.unique_id_to_piece_id[:self.total_number_of_uids]
        self.unique_id_to_rotation_id = self.unique_id_to_rotation_id[:self.total_number_of_uids]
        self.unique_piece_id_to_join_point = self.unique_piece_id_to_join_point[:self.total_number_of_uids]

        self.available_uids_per_player = np.ones((4, self.total_number_of_uids), dtype=bool)

        self.open_board_locations = np.zeros((4, boardSize + 1, boardSize + 1), dtype=bool)
        self.open_board_locations[0, 0, 0] = True
        self.open_board_locations[1, 20, 0] = True
        self.open_board_locations[2, 20, 20] = True
        self.open_board_locations[3, 0, 20] = True
        # how it should work when placing a block, remove point where we place the block. add all the other points of that block to this list(after adding the 'origin' of the block to all the vectors),
        self.max_length_of_join_points = max_length_of_join_points

    def player_turn(self):
        return self._player_turn

    def re_init(self):
        self._player_turn = 0
        self.maskingBoards[:] = False
        self.full_board[:] = 0

        self.available_uids_per_player[:] = True

        self.open_board_locations[:] = False
        self.open_board_locations[0, 0, 0] = True
        self.open_board_locations[1, self.board_size, 0] = True
        self.open_board_locations[2, self.board_size, self.board_size] = True
        self.open_board_locations[3, 0, self.board_size] = True

    # @timeit
    def check_if_move_valid(self, board_point: np.array, unique_id: 0):
        unique_id = np.array((unique_id,), dtype=np.int64)
        # piece_point = self.all_join_points[piece_id, piece_point_id]
        unique_id_point = self.unique_piece_id_to_join_point[unique_id]
        rotation_id = self.unique_id_to_rotation_id[unique_id]
        is_valid_placement = np.ones((1, 1), dtype=bool)
        check_if_piece_possible(
            self.maskingBoards[self._player_turn],
            self.full_board,
            self.all_unique_pieces,
            self.all_piece_sizes,
            self.all_unique_masks,
            np.atleast_2d(board_point),
            np.atleast_2d(unique_id_point),
            rotation_id,
            is_valid_placement
        )
        return is_valid_placement[0, 0]

    def number_of_tokens(self):
        return ((self.board_size + 1) * (self.board_size + 1)) + self.total_number_of_uids

    # @timeit
    def current_player_get_all_valid_moves(self):
        current_player = self._player_turn

        return get_all_valid_moves(
            current_player,
            self.available_uids_per_player,
            self.open_board_locations,
            self.unique_id_to_rotation_id,
            self.unique_piece_id_to_join_point,
            self.maskingBoards,
            self.full_board,
            self.all_unique_pieces,
            self.all_piece_sizes,
            self.all_unique_masks
        )

    # @timeit
    def current_player_submit_move(self, board_point: np.array, unique_id: int):
        current_player = self._player_turn

        rotation_id = self.unique_id_to_rotation_id[unique_id]
        piece_point_location = self.unique_piece_id_to_join_point[unique_id]

        piece_shape = self.all_piece_sizes[rotation_id]
        piece_origin = board_point - piece_point_location
        piece_bb_end = piece_origin + piece_shape

        if not self.check_if_move_valid(board_point, unique_id):
            print("Submitted move found to be invalid????")
            return False

        if not self.open_board_locations[current_player][board_point[0], board_point[1]]:
            print("Could not find board point as valid??????")
            return False

        self.playerBoards[self._player_turn, piece_origin[0]:piece_bb_end[0], piece_origin[1]:piece_bb_end[1]] |= \
            self.all_unique_pieces[rotation_id][0:piece_shape[0], 0:piece_shape[1]]
        self.full_board[self.playerBoards[self._player_turn]] = self._player_turn + 1

        update_open_positions(
            self.open_board_locations[current_player],
            board_point,
            unique_id,

            self.unique_id_to_rotation_id,
            self.all_piece_sizes,
            self.unique_piece_id_to_join_point,
            self.all_join_points,
            self.all_join_points_length,
            self.playerBoards[current_player]
        )

        piece_id = self.unique_id_to_piece_id[unique_id]
        start_stop = self.piece_id_to_uid_start_stop[piece_id]

        self.available_uids_per_player[current_player, start_stop[0]:start_stop[1]] = False
        self._player_turn = (self._player_turn + 1) % 4
        return True

    def current_player_skip_turn(self):
        self._player_turn = (self._player_turn + 1) % 4


def check():
    board = BlokusBoard()
    for i in range(10):
        results = board.current_player_get_all_valid_moves()
        if results[0] is None:
            board.current_player_skip_turn()
            continue
        rand = random.randint(0, results[1].shape[0] - 1)
        board.current_player_submit_move(results[0][rand], results[1][rand])
        # visualizer.plot_store_board(board, f"board_{i}")


if __name__ == '__main__':
    check()
