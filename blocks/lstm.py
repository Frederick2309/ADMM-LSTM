""" lstm.py (c) Yang Yang, 2024-2025
  This file implements an LSTM model suitable for ADMM-based training.
"""

import torch
from torch import nn
from torch.nn import init
from typing import Dict


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, with_grad: bool = False) -> None:
        """
        This class implements an LSTM-Linear block.
        :param with_grad: When using gradient-based optimization, this option should be True.
        """
        super().__init__()
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.init_parameters()
        self.sigmoid, self.tanh = nn.Sigmoid(), nn.Tanh()
        self.with_grad = with_grad

    def init_parameters(self) -> None:
        for map_to in ['i', 'f', 'g', 'o']:
            setattr(self, f'x2{map_to}', nn.Parameter(torch.randn(self.input_size, self.hidden_size)))
            setattr(self, f'h2{map_to}', nn.Parameter(torch.randn(self.hidden_size, self.hidden_size)))
        setattr(self, 'out', nn.Parameter(torch.randn(self.hidden_size, self.output_size)))
        for param in self.parameters():
            init.xavier_normal_(param)

    def get_weight(self, map_from: str, map_to: str) -> nn.Parameter:
        return getattr(self, f'{map_from}2{map_to}').clone().detach()

    def set_weight(self, map_from: str, map_to: str, value: torch.Tensor) -> None:
        setattr(self, f'{map_from}2{map_to}', nn.Parameter(value.clone().detach()))

    def get_wy(self) -> nn.Parameter:
        return getattr(self, 'out').clone().detach()

    def set_wy(self, value: torch.Tensor) -> None:
        setattr(self, 'out', nn.Parameter(value))

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, h: torch.Tensor = None) -> torch.Tensor:
        if self.with_grad:
            return self.grad_forward(x, c, h)
        return self.init_gate_variables(x, c, h)['a']

    def grad_forward(self, x: torch.Tensor, c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert x.size(2) == self.input_size
        batch, seq_len, _ = x.size()
        if c is None:
            c = torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device)
        if h is None:
            h = torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device)
        for t in range(seq_len):
            x_t = x[:, t, :]
            i = self.sigmoid(x_t @ self.x2i + h @ self.h2i)
            f = self.sigmoid(x_t @ self.x2f + h @ self.h2f)
            g = self.tanh(x_t @ self.x2g + h @ self.h2g)
            o = self.sigmoid(x_t @ self.x2o + h @ self.h2o)
            c = f * c + i * g
            h = o * self.tanh(c)
        return h @ self.out

    def init_gate_variables(self, x: torch.Tensor, c: torch.Tensor = None, h: torch.Tensor = None) \
            -> Dict[str, torch.Tensor]:
        assert x.size(2) == self.input_size
        batch, seq_len, _ = x.size()
        if c is None:
            c = torch.zeros(batch, seq_len + 1, self.hidden_size, dtype=x.dtype, device=x.device)
        if h is None:
            h = torch.zeros(batch, seq_len + 1, self.hidden_size, dtype=x.dtype, device=x.device)
        i = torch.zeros(batch, seq_len + 1, self.hidden_size, dtype=x.dtype, device=x.device)
        f = torch.zeros(batch, seq_len + 1, self.hidden_size, dtype=x.dtype, device=x.device)
        g = torch.zeros(batch, seq_len + 1, self.hidden_size, dtype=x.dtype, device=x.device)
        o = torch.zeros(batch, seq_len + 1, self.hidden_size, dtype=x.dtype, device=x.device)
        for t in range(1, seq_len + 1):
            x_t = x[:, t - 1, :]
            h_before = h[:, t - 1, :]
            i[:, t, :] = self.sigmoid(x_t @ self.x2i + h_before @ self.h2i)
            f[:, t, :] = self.sigmoid(x_t @ self.x2f + h_before @ self.h2f)
            g[:, t, :] = self.tanh(x_t @ self.x2g + h_before @ self.h2g)
            o[:, t, :] = self.sigmoid(x_t @ self.x2o + h_before @ self.h2o)
            c[:, t, :] = f[:, t, :] * c[:, t - 1, :] + i[:, t, :] * g[:, t, :]
            h[:, t, :] = o[:, t, :] * self.tanh(c[:, t, :])
        return {
            'i': i, 'f': f, 'g': g, 'o': o, 'c': c, 'h': h, 'a': h[:, seq_len, :] @ self.out
        }
