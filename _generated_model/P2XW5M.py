import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.retiarii.nn.pytorch

import torch


class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._first_layer = torch.nn.modules.linear.Linear(in_features=33, out_features=112)
        self._mapping_ = {'_first_layer': 'first_layer'}

    def forward(self, x__1):
        _first_layer = self._first_layer(x__1)
        return _first_layer