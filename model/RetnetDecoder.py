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


class RetNetDecoderLayer(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        # retention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.retention = MultiScaleRetention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        self.cross_retention = MultiScaleRetention(  # type: ignore
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
