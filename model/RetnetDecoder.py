from torch import nn
from copy import deepcopy
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
from einops import rearrange
from torch import Tensor, nn

from yet_another_retnet.retention import (
    ActivationString,
    MultiScaleRetention,
    _get_activation_fn,
)


class RNetDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
            norm_first: bool = True,
            layer_norm_eps: float = 1e-6,
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        # retention block
        self.num_heads = nhead
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        if d_model % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({d_model}) must be divisible by num_heads ({self.num_heads})"
            )

        self.head_dim = d_model // self.num_heads
        if not self.head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {self.head_dim}) must be divisible by 8"
            )
        self.retention = MultiScaleRetention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        self.cross_retention = MultiScaleRetention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        # feedforward block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Check that we're following the same initialization as the paper
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward_parallel(self, x: Tensor, mem: Tensor) -> Tensor:
        def _retention_block(x: Tensor) -> Tensor:
            x, _ = self.retention.forward_parallel(x, x, x)
            return self.dropout(x)

        def _cross_retention_block(x: Tensor, mem: Tensor) -> Tensor:
            x, _ = self.cross_retention.forward_parallel(x, mem, mem)
            return self.dropout(x)

        if self.norm_first:
            x = x + _retention_block(self.norm1(x))
            x = x + _cross_retention_block(self.norm2(x), mem)
            x = x + self._feedforward_block(self.norm3(x))
        else:
            x = x + self.norm1(_retention_block(x))
            x = x + self.norm2(_cross_retention_block(x, mem))
            x = x + self.norm3(self._feedforward_block(x))
        return x

    def forward_recurrent(
            self, x: Tensor, mem: Tensor, seq_idx: Optional[Union[Tensor, int]] = 0,
            prev_state: Optional[Tensor] = None,
            cross_prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_recurrent(
                x, x, x, seq_idx=seq_idx, prev_state=prev_state
            )
            return self.dropout(x), state

        def _cross_retention_block(x: Tensor, mem: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.cross_retention.forward_recurrent(
                x, mem, mem, seq_idx=seq_idx, prev_state=cross_prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            y, cross_state = _cross_retention_block(self.norm2(x), mem)
            x = x + y
            x = x + self._feedforward_block(self.norm3(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            y, cross_state = _cross_retention_block(x, mem)
            x = x + self.norm2(y)
            x = x + self.norm3(self._feedforward_block(x))

        return x, state, cross_state

    def forward_chunkwise(
            self, x: Tensor, mem, start_idx: Optional[Union[Tensor, int]] = 0, prev_state: Optional[Tensor] = None,
            cross_prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_chunkwise(
                x, x, x, start_idx=start_idx, prev_state=prev_state
            )
            return self.dropout(x), state

        def _cross_retention_block(x: Tensor, mem: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.cross_retention.forward_chunkwise(
                x, mem, mem, start_idx=start_idx, prev_state=cross_prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            y, cross_state = _cross_retention_block(self.norm2(x), mem)
            x = x + y
            x = x + self._feedforward_block(self.norm3(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            y, cross_state = _cross_retention_block(x, mem)
            x = x + self.norm2(y)
            x = x + self.norm3(self._feedforward_block(x))

        return x, state, cross_state

    def forward(self, x: Tensor, mem: Tensor) -> Tensor:
        return self.forward_parallel(x, mem)


class RetNetDecoder(nn.Module):
    def __init__(self, decoder_layer: RNetDecoderLayer, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.device = decoder_layer.device
        self.dtype = decoder_layer.dtype
        self.num_heads = decoder_layer.num_heads
        self.head_dim = decoder_layer.head_dim
        self.layers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward_parallel(self, x: Tensor, mem: Tensor) -> Tensor:
        for layer in self.layers:
            assert isinstance(layer, RNetDecoderLayer)
            x = layer.forward_parallel(x, mem)
        return x

    def forward_recurrent(
            self, x: Tensor, mem: Tensor, seq_idx: Optional[Union[Tensor, int]] = 0,
            prev_state: Optional[Tensor] = None,
            cross_prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if len(x.shape) != 2:
            raise ValueError(
                f"Unexpected shape of input, expected [batches,seq_len,emb_dim]"
            )
        if torch.is_tensor(seq_idx):
            seq_idx = rearrange(seq_idx, "b -> b () ()")
        batches, _ = x.shape
        if prev_state is None or cross_prev_state is None:
            prev_state = [None] * self.num_layers
            cross_prev_state = [None] * self.num_layers
        else:
            if len(prev_state.shape) != 5 or len(cross_prev_state.shape) != 5:
                raise ValueError(
                    f"Unexpected shape of prev_states or prev_cross_states"
                )
            test_state_layers, batch_state, _, _, _ = prev_state.shape
            test_cross_state_layers, batch_cross_state, _, _, _ = cross_prev_state.shape
            if test_state_layers != test_cross_state_layers and test_state_layers != self.num_layers:
                raise ValueError(
                    f"Expected {len(self.layers)} previous states, got incorrect amount"
                )
            if batch_state != batch_cross_state:
                raise ValueError(
                    f"Cross state and state do not have the same number of batches"
                )
            batches = batch_state

        states = torch.zeros([self.num_layers, batches, self.num_heads, self.head_dim, self.head_dim],
                             dtype=self.dtype, device=self.device)
        cross_states = torch.zeros([self.num_layers, batches, self.num_heads, self.head_dim, self.head_dim],
                                   dtype=self.dtype, device=self.device)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            assert isinstance(layer, RNetDecoderLayer)
            x, states[i], cross_states[i] = layer.forward_recurrent(x, mem, seq_idx, prev_state[i],
                                                                    cross_prev_state[i])
        return x, states, cross_states

    def forward_chunkwise(
            self, x: Tensor, mem: Tensor, seq_idx: Optional[Union[Tensor, int]] = 0,
            prev_state: Optional[Tensor] = None,
            cross_prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if len(x.shape) != 3:
            raise ValueError(
                f"Unexpected shape of input, expected [batches,seq_len,emb_dim]"
            )
        if torch.is_tensor(seq_idx):
            seq_idx = rearrange(seq_idx, "b -> b () () ()")
        batches, _, _ = x.shape
        if prev_state is None or cross_prev_state is None:
            prev_state = [None] * self.num_layers
            cross_prev_state = [None] * self.num_layers
        else:
            if len(prev_state.shape) != 5 or len(cross_prev_state.shape) != 5:
                raise ValueError(
                    f"Unexpected shape of prev_states or prev_cross_states"
                )
            test_state_layers, batch_state, _, _, _ = prev_state.shape
            test_cross_state_layers, batch_cross_state, _, _, _ = cross_prev_state.shape
            if test_state_layers != test_cross_state_layers and test_state_layers != self.num_layers:
                raise ValueError(
                    f"Expected {len(self.layers)} previous states, got incorrect amount"
                )
            if batch_state != batch_cross_state:
                raise ValueError(
                    f"Cross state and state do not have the same number of batches"
                )
            batches = batch_state

        states = torch.zeros([self.num_layers, batches, self.num_heads, self.head_dim, self.head_dim],
                             dtype=self.dtype, device=self.device)
        cross_states = torch.zeros([self.num_layers, batches, self.num_heads, self.head_dim, self.head_dim],
                                   dtype=self.dtype, device=self.device)
        for i in range(self.num_layers):
            layer = self.layers[i]
            assert isinstance(layer, RNetDecoderLayer)
            x, states[i], cross_states[i] = layer.forward_chunkwise(x, mem, seq_idx, prev_state[i],
                                                                    cross_prev_state[i])
        return x, states, cross_states

    def forward(self, x: Tensor, mem: Tensor) -> Tensor:
        return self.forward_parallel(x, mem)


def main():
    size = (10, 20, 64)
    layer = RNetDecoderLayer(64, 4, 256, 0)
    layers = RetNetDecoder(layer, 4)

    input_test1 = torch.rand(*size)
    input_test1_mem = torch.rand(*size)
    layers.eval()

    output1, state1, cross_state1 = layers.forward_chunkwise(input_test1, input_test1_mem)

    input_test2 = torch.rand(*size)
    input_test2_mem = torch.rand(*size)
    output2, state2, cross_state2 = layers.forward_chunkwise(input_test2, input_test2_mem, 20, state1, cross_state1)

    input_test2 = torch.rand(*size)
    seq_lens = torch.randint(0, 20, size=[10])
    input_test2_mem = torch.rand(*size)
    output2, state2, cross_state2 = layers.forward_chunkwise(input_test2, input_test2_mem, seq_lens, state1,
                                                             cross_state1)

    input_test3 = torch.rand([10, 64])
    input_test3_mem = torch.rand([10, 64])
    output3, state3, cross_state3 = layers.forward_recurrent(input_test3, input_test3_mem, 40, state2, cross_state2)

    print("Test")


main()
