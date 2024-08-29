import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.retiarii.nn.pytorch

import nni


class _model__layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.mutable_all_0 = nni.retiarii.nn.pytorch.Flatten()
        self.mutable_all_fc_0 = nni.retiarii.nn.pytorch.Linear(in_features=33, out_features=32)
        self.mutable_all_fc_activation_0 = nni.retiarii.nn.pytorch.Hardshrink()
        self.mutable_all_fc_1 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_1 = nni.retiarii.nn.pytorch.Hardshrink()
        self.mutable_all_fc_2 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_2 = nni.retiarii.nn.pytorch.Hardshrink()
        self.mutable_all_fc_3 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_3 = nni.retiarii.nn.pytorch.Hardshrink()
        self.mutable_all_fc_4 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=1)
        self._mapping_ = {'mutable_all_0': 'layers.0', 'mutable_all_fc_0': None, 'mutable_all_fc_activation_0': None, 'mutable_all_fc_1': None, 'mutable_all_fc_activation_1': None, 'mutable_all_fc_2': None, 'mutable_all_fc_activation_2': None, 'mutable_all_fc_3': None, 'mutable_all_fc_activation_3': None, 'mutable_all_fc_4': None}

    def forward(self, input__1):
        mutable_all_0 = self.mutable_all_0(input__1)
        mutable_all_fc_0 = self.mutable_all_fc_0(mutable_all_0)
        mutable_all_fc_activation_0 = self.mutable_all_fc_activation_0(mutable_all_fc_0)
        mutable_all_fc_1 = self.mutable_all_fc_1(mutable_all_fc_activation_0)
        mutable_all_fc_activation_1 = self.mutable_all_fc_activation_1(mutable_all_fc_1)
        mutable_all_fc_2 = self.mutable_all_fc_2(mutable_all_fc_activation_1)
        mutable_all_fc_activation_2 = self.mutable_all_fc_activation_2(mutable_all_fc_2)
        mutable_all_fc_3 = self.mutable_all_fc_3(mutable_all_fc_activation_2)
        mutable_all_fc_activation_3 = self.mutable_all_fc_activation_3(mutable_all_fc_3)
        mutable_all_fc_4 = self.mutable_all_fc_4(mutable_all_fc_activation_3)
        return mutable_all_fc_4



class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = _model__layers()
        self._mapping_ = {'_layers': 'layers'}

    def forward(self, x__1):
        _layers = self._layers(x__1)
        return _layers