import datetime
import pathlib
import random

import numpy as np
import visualizer
import board as blokus_board
import torch
import visualizer
from util import timeit


class Game:
    def __init__(self, seed=None):
        self.selected_pos_action = None
        self.selected_pos = None
        self.board = blokus_board.BlokusBoard(20)
        self.position_selected = False
        self.no_more_moves = np.zeros(4, dtype=bool)

    # @timeit
    def step(self, action):
        # return self.get_observation(), reward, done
        if not self.position_selected:
            x = action % 21
            y = action // 21
            self.position_selected = True
            self.selected_pos_action = action
            self.selected_pos = np.array([x, y])
            return self.get_observation(), 0, False
        else:
            self.position_selected = False
            uid = action - (21 * 21)
            self.board.current_player_submit_move(self.selected_pos, uid)

            # keep going until there are valid moves
            while self.board.current_player_get_all_valid_moves()[0] is None:
                self.no_more_moves[self.board.player_turn()] = True
                if self.is_finished():
                    break
                self.board.current_player_skip_turn()

            r_id = self.board.unique_id_to_rotation_id[uid]
            score = np.sum(self.board.all_unique_pieces[r_id])

            return self.get_observation(), score, self.is_finished()

    # @timeit
    def legal_actions(self):
        if not self.position_selected:
            valid_options = self.board.current_player_get_all_valid_moves()
            available_positions = valid_options[0].astype(int)
            valid_position_actions = available_positions[:, 0] + available_positions[:, 1] * 21
            return np.unique(valid_position_actions).tolist()
        else:
            # this can be optimized
            valid_options = self.board.current_player_get_all_valid_moves()
            available_positions = valid_options[0].astype(int)
            valid_position_actions = available_positions[:, 0] + available_positions[:, 1] * 21
            valid_uid_options = valid_options[1][(valid_position_actions == self.selected_pos_action).nonzero()]
            return (21 * 21 + valid_uid_options).tolist()

    def reset(self):
        selection_step = 0
        self.selected_pos_action = None
        self.selected_pos = None
        self.position_selected = False
        self.no_more_moves[:] = False
        self.board.re_init()
        return self.get_observation()

    def is_finished(self):
        return np.all(self.no_more_moves)

    def render(self):
        visualizer.plot_board(self.board)

    def get_observation(self):
        obs = np.zeros((5, 21, 21), dtype=bool)
        # fit available piece data into the edges
        obs[0:4, 20, :] = self.board.available_uids_per_player[:, self.board.uid_to_piece_id_index]
        obs[0:4, 0:20, 0:20] = self.board.playerBoards
        obs[self.board.player_turn(), 0, 20] = True
        if self.position_selected:
            obs[4, self.selected_pos[0], self.selected_pos[1]] = True
        return obs

    def packed_obs(self):
        obs = self.get_observation()
        packed = np.packbits(obs)
        return packed

    def to_play(self):
        # need to check if there are any valid positions here
        current_player_id = self.board.player_turn() % 2
        return current_player_id


def main():
    bboard = Game()
    for i in range(21 * 4 * 2):
        check = bboard.legal_actions()
        rand = random.randint(0, len(check) - 1)
        obs, reward, finished = bboard.step(check[rand])
        print(obs.shape, reward, finished, i)
        # visualizer.plot_store_board(bboard.board, f"../match_replays/board_{i}")
        if finished:
            break


if __name__ == '__main__':
    main()
