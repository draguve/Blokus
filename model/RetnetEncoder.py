import torch
from einops import rearrange
from torch import Tensor, nn
from copy import deepcopy
from typing import Callable, List, Optional, Sequence, Tuple, Union
from yet_another_retnet.retnet import RetNetDecoderLayer


class RetNetEncoder(nn.Module):
    def __init__(self, encoder_layer: RetNetDecoderLayer, num_layers: int, device, dtype, num_heads, head_dim):
        super().__init__()
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward_parallel(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            assert isinstance(layer, RetNetDecoderLayer)
            x = layer.forward_parallel(x)
        return x

    def forward_recurrent(
            self, x: Tensor, seq_idx: Optional[Union[Tensor, int]] = 0, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if len(x.shape) != 2:
            raise ValueError(
                f"Unexpected shape of input, expected [batches,emb_dim]"
            )
        if torch.is_tensor(seq_idx):
            seq_idx = rearrange(seq_idx, "b -> b () ()")
        batches, _ = x.shape
        if prev_state is None:
            prev_state = [None] * self.num_layers
        else:
            if len(prev_state.shape) != 5:
                raise ValueError(
                    f"Unexpected shape of prev_states"
                )
            test_state_layers, batch_state, _, _, _ = prev_state.shape
            if test_state_layers != self.num_layers:
                raise ValueError(
                    f"Expected {len(self.layers)} previous states, got incorrect amount"
                )
            assert batches == batch_state

        states = torch.zeros([self.num_layers, batches, self.num_heads, self.head_dim, self.head_dim],
                             dtype=self.dtype, device=self.device)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            assert isinstance(layer, RetNetDecoderLayer)
            x, states[i] = layer.forward_recurrent(x, seq_idx, prev_state[i])
        return x, states

    def forward_chunkwise(
            self, x: Tensor, seq_idx: Optional[Union[Tensor, int]] = 0, prev_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if len(x.shape) != 3:
            raise ValueError(
                f"Unexpected shape of input, expected [batches,seq_len,emb_dim]"
            )
        if torch.is_tensor(seq_idx):
            seq_idx = rearrange(seq_idx, "b -> b () () ()")
        batches, _, _ = x.shape
        if prev_state is None:
            prev_state = [None] * self.num_layers
        else:
            if len(prev_state.shape) != 5:
                raise ValueError(
                    f"Unexpected shape of prev_states or prev_cross_states"
                )
            test_state_layers, batch_state, _, _, _ = prev_state.shape
            if test_state_layers != self.num_layers:
                raise ValueError(
                    f"Expected {len(self.layers)} previous states, got incorrect amount"
                )
            assert batches == batch_state

        states = torch.zeros([self.num_layers, batches, self.num_heads, self.head_dim, self.head_dim],
                             dtype=self.dtype, device=self.device)
        for i in range(self.num_layers):
            layer = self.layers[i]
            assert isinstance(layer, RetNetDecoderLayer)
            x, states[i] = layer.forward_chunkwise(x, seq_idx, prev_state[i])
        return x, states

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)
