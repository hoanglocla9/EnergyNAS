import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.retiarii.nn.pytorch

import nni


class _model__layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.mutable_all_0 = nni.retiarii.nn.pytorch.Conv1d(kernel_size=7, in_channels=1, out_channels=1, padding=3)
        self.mutable_all_conv_activation_0 = nni.retiarii.nn.pytorch.LogSigmoid()
        self.mutable_all_pooling = nni.retiarii.nn.pytorch.AvgPool1d(kernel_size=7, stride=1, padding=3)
        self.mutable_all_flatten = nni.retiarii.nn.pytorch.Flatten()
        self.mutable_all_fc_0 = nni.retiarii.nn.pytorch.Linear(in_features=33, out_features=16)
        self.mutable_all_fc_activation_0 = nni.retiarii.nn.pytorch.Softplus()
        self.mutable_all_fc_1 = nni.retiarii.nn.pytorch.Linear(in_features=16, out_features=16)
        self.mutable_all_fc_activation_1 = nni.retiarii.nn.pytorch.Softplus()
        self.mutable_all_fc_2 = nni.retiarii.nn.pytorch.Linear(in_features=16, out_features=1)
        self._mapping_ = {'mutable_all_0': 'layers.0', 'mutable_all_conv_activation_0': None, 'mutable_all_pooling': None, 'mutable_all_flatten': None, 'mutable_all_fc_0': None, 'mutable_all_fc_activation_0': None, 'mutable_all_fc_1': None, 'mutable_all_fc_activation_1': None, 'mutable_all_fc_2': None}

    def forward(self, input__1):
        mutable_all_0 = self.mutable_all_0(input__1)
        mutable_all_conv_activation_0 = self.mutable_all_conv_activation_0(mutable_all_0)
        mutable_all_pooling = self.mutable_all_pooling(mutable_all_conv_activation_0)
        mutable_all_flatten = self.mutable_all_flatten(mutable_all_pooling)
        mutable_all_fc_0 = self.mutable_all_fc_0(mutable_all_flatten)
        mutable_all_fc_activation_0 = self.mutable_all_fc_activation_0(mutable_all_fc_0)
        mutable_all_fc_1 = self.mutable_all_fc_1(mutable_all_fc_activation_0)
        mutable_all_fc_activation_1 = self.mutable_all_fc_activation_1(mutable_all_fc_1)
        mutable_all_fc_2 = self.mutable_all_fc_2(mutable_all_fc_activation_1)
        return mutable_all_fc_2



class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = _model__layers()
        self._mapping_ = {'_layers': 'layers'}

    def forward(self, x__1):
        _layers = self._layers(x__1)
        return _layers