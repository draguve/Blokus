import visualizer
from mcts import MCTS, Model
from games.blokus import Game


class SelfPlay:
    def __init__(self, max_num_simulations=800, hidden_state_size=32, mcts_batch_size=16):
        # TODO: make the game and model an argument
        self.game = Game()
        self.model = Model(self.game.total_number_of_possible_tokens(), hidden_state_size)
        self.mcts = MCTS(
            self.game.total_number_of_possible_tokens(),
            hidden_state_size,
            max_num_simulations
        )
        self.mcts_batch_size = mcts_batch_size

    def continous_self_play(self):
        pass

    def play_game(self, temp_thresh):
        for i in range(21 * 4 * 2):
            self.mcts.reinit()
            to_invert = self.game.to_play() == 1
            self.mcts.run_batched(self.model, self.game.get_observation(), self.game.legal_actions(), i,
                                  self.mcts_batch_size, to_invert)
            # TODO No need to check to find the action if there is only one legal action
            action = self.mcts.select_root_action(temp_thresh)
            obs, reward, finished = self.game.step(action)
            visualizer.plot_store_board(self.game.board, f"match_replays/board_{i}")
            if finished:
                break


def main():
    play = SelfPlay(400, 32, 16)
    play.play_game(0)


if __name__ == '__main__':
    main()
