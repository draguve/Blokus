import torch
from torch import nn, Tensor
import numpy as np
from RetnetDecoder import RetNetDecoder, RNetDecoderLayer
from RetnetEncoder import RetNetEncoder
from yet_another_retnet.retnet import RetNetDecoderLayer as RetNetEncoderLayer

from games.blokus import Game


class ReinforcementNet(nn.Module):
    def __init__(self, input_shape, embedding_size=64, policy_size=400, n_heads=4, dropout=0.1, dim_ff=512,
                 num_layers=3, support_scalar_size=10, device="cpu", dtype=torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if embedding_size % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embedding_size}) must be divisible by num_heads ({n_heads})"
            )

        head_dim = embedding_size // n_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )

        self.support_scalar_size = support_scalar_size
        total_input_size = np.prod(input_shape)
        
        represent_net = nn.Sequential(
            nn.Linear(total_input_size, embedding_size, device=device, dtype=dtype),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )
        self.represent = represent_net
        encoder_layer = RetNetEncoderLayer(embedding_size, n_heads, dim_ff, dropout)
        self.encoder = RetNetEncoder(encoder_layer, num_layers, device, dtype, num_heads=4,
                                     head_dim=head_dim)

        policy_head_layer = RNetDecoderLayer(embedding_size, n_heads, dim_ff, dropout)
        self.policy_head = RetNetDecoder(policy_head_layer, num_layers)
        self.policy_ff = nn.Sequential(
            nn.Linear(embedding_size, policy_size, device=device, dtype=dtype),
            nn.Dropout(dropout),
            nn.Softmax()
        )
        self.policy_emb = nn.Embedding(policy_size, embedding_size)
        # self.softmax = nn.Softmax()

    def init_infer(
            self, obs, prev_action, seq_idx, prev_encoder_state,
            prev_policy_state, prev_cross_policy_state,
            prev_reward_state, prev_cross_reward_state,
            prev_value_state,prev_cross_value_state,
    ):
        representation = self.represent(torch.atleast_2d(obs))
        prev_emb_policy = torch.atleast_2d(self.policy_emb(prev_action))
        mem, next_encoder_states = self.encoder.forward_recurrent(representation, seq_idx, prev_encoder_state)
        emb_policy, next_policy_state, next_policy_cross_state = self.policy_head.forward_recurrent(prev_emb_policy,
                                                                                                    mem,
                                                                                                    seq_idx,
                                                                                                    prev_policy_state,
                                                                                                    prev_cross_policy_state)

        policy_logits = self.policy_ff(emb_policy)
        return policy_logits
        # return value, reward, self.softmax(policy_logits), representation


def main():
    game = Game()
    input_shape = game.get_observation().shape
    net = ReinforcementNet(
        input_shape,
        embedding_size=64,
        policy_size=game.total_number_of_possible_tokens(),
        n_heads=4,
        dropout=0.1,
        dim_ff=512,
        num_layers=3,
        device="cpu",
        dtype=torch.float32
    )

    net.init_infer(
        torch.flatten(Tensor(game.get_observation())),
        torch.IntTensor(game.no_more_moves_token),
        0,
        None,
        None,
        None
    )


if __name__ == '__main__':
    main()
