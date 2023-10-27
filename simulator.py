from blockus import BlokusBoard
from players.RandomPlayer import RandomPlayer
from util import random_id, check_for_dir
import numpy as np
import os
from visualizer import plot_store_board, plot_remaining_pieces

VISUALIZE_EVERY_STEP = True
visualizer_location = "match_replays/"


class BlokusSim:
    def __init__(self, players):
        if len(players) < 2:
            raise NotImplementedError
        if len(players) == 3:
            raise NotImplementedError
        self.board = BlokusBoard()
        self.game_id = random_id(10)
        self.players = players
        self.step = 0
        self.no_more_moves = np.zeros(4, dtype=bool)

    def run_steps(self, num_steps):
        for i in range(num_steps):
            self.visualize_board()
            if self.is_finished():
                print(f"{self.game_id} match finished")
                break
            self.step += 1
            current_player_id = self.board.player_turn() % len(self.players)
            current_player = self.players[current_player_id]

            if self.no_more_moves[self.board.player_turn()]:
                self.board.current_player_skip_turn()
                continue

            possible_moves = self.board.current_player_get_all_valid_moves()
            if possible_moves[0] is None:
                self.no_more_moves[self.board.player_turn()] = True
                self.board.current_player_skip_turn()
                continue

            chosen_move = current_player.choose_move(self.board, *possible_moves)
            result = self.board.current_player_submit_move(possible_moves[0][chosen_move],
                                                           possible_moves[1][chosen_move],
                                                           possible_moves[2][chosen_move])
            if not result:
                self.board.current_player_skip_turn()
        self.visualize_board("end")
        self.visualize_remaining()

    def visualize_board(self, extra=""):
        if VISUALIZE_EVERY_STEP:
            check_for_dir(f"{visualizer_location}/{self.game_id}/")
            plot_store_board(self.board, f"{visualizer_location}/{self.game_id}/match_{self.step}{extra}")

    def visualize_remaining(self):
        if VISUALIZE_EVERY_STEP:
            check_for_dir(f"{visualizer_location}/{self.game_id}/")
            plot_remaining_pieces(self.board, f"{visualizer_location}/{self.game_id}/remaining_{self.step}")

    def get_current_score(self):
        score = np.zeros(len(self.players), dtype=int)
        for i in range(4):
            score[i % len(self.players)] += np.sum(self.board.playerBoards[i])
        return score

    def is_finished(self):
        return np.all(self.no_more_moves)


def check():
    players = [RandomPlayer(), RandomPlayer()]
    sim = BlokusSim(players)
    sim.run_steps(21 * 4)
    sim.board.current_player_get_all_valid_moves()
    print(f"Matches {sim.get_current_score()}")


if __name__ == '__main__':
    check()