from mcts import MCTS, Model
from games.blokus import Game


class SelfPlay:
    def __init__(self, max_num_simulations=800, hidden_state_size=32, mcts_batch_size=16):
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

    def play_game(self):
        for i in range(21 * 4 * 2):
            self.mcts.reinit()
            to_invert = self.game.to_play() == 1
            self.mcts.run_batched(self.model, self.game.get_observation(), self.game.legal_actions(), i,
                                  self.mcts_batch_size,to_invert)
            obs, reward, finished = self.game.step(check[rand])
            if finished:
                break

    def select_action(self):

