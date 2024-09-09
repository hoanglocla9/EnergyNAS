import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.retiarii.nn.pytorch

import nni


class _model__layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.mutable_all_0 = nni.retiarii.nn.pytorch.api.Placeholder(label='mutable_all', n_input_features=33, n_neurons_option=range(4, 4097, 4), fc_n_layer_option=range(2, 10), dropout_option=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9], fc_activation_fn_option=['None', '__torch__.nni.retiarii.nn.pytorch.ReLU', '__torch__.nni.retiarii.nn.pytorch.Tanh', '__torch__.nni.retiarii.nn.pytorch.LeakyReLU', '__torch__.nni.retiarii.nn.pytorch.Hardshrink', '__torch__.nni.retiarii.nn.pytorch.Hardsigmoid', '__torch__.nni.retiarii.nn.pytorch.Hardtanh', '__torch__.nni.retiarii.nn.pytorch.Hardswish', '__torch__.nni.retiarii.nn.pytorch.LogSigmoid', '__torch__.nni.retiarii.nn.pytorch.PReLU', '__torch__.nni.retiarii.nn.pytorch.ELU', '__torch__.nni.retiarii.nn.pytorch.ReLU6', '__torch__.nni.retiarii.nn.pytorch.RReLU', '__torch__.nni.retiarii.nn.pytorch.CELU', '__torch__.nni.retiarii.nn.pytorch.SiLU', '__torch__.nni.retiarii.nn.pytorch.Softplus', '__torch__.nni.retiarii.nn.pytorch.Softshrink', '__torch__.nni.retiarii.nn.pytorch.Softsign', '__torch__.nni.retiarii.nn.pytorch.Tanhshrink'])
        self.mutable_all_fc_0 = nni.retiarii.nn.pytorch.Linear(in_features=33, out_features=1528)
        self.mutable_all_fc_activation_0 = nni.retiarii.nn.pytorch.ReLU6()
        self.mutable_all_fc_dropout_0 = nni.retiarii.nn.pytorch.Dropout(p=0.8)
        self.mutable_all_fc_1 = nni.retiarii.nn.pytorch.Linear(in_features=1528, out_features=1528)
        self.mutable_all_fc_activation_1 = nni.retiarii.nn.pytorch.ReLU6()
        self.mutable_all_fc_dropout_1 = nni.retiarii.nn.pytorch.Dropout(p=0.8)
        self.mutable_all_fc_2 = nni.retiarii.nn.pytorch.Linear(in_features=1528, out_features=1)
        self._mapping_ = {'mutable_all_0': 'layers.0', 'mutable_all_fc_0': None, 'mutable_all_fc_activation_0': None, 'mutable_all_fc_dropout_0': None, 'mutable_all_fc_1': None, 'mutable_all_fc_activation_1': None, 'mutable_all_fc_dropout_1': None, 'mutable_all_fc_2': None}

    def forward(self, input__1):
        mutable_all_0 = self.mutable_all_0(input__1)
        mutable_all_fc_0 = self.mutable_all_fc_0(mutable_all_0)
        mutable_all_fc_activation_0 = self.mutable_all_fc_activation_0(mutable_all_fc_0)
        mutable_all_fc_dropout_0 = self.mutable_all_fc_dropout_0(mutable_all_fc_activation_0)
        mutable_all_fc_1 = self.mutable_all_fc_1(mutable_all_fc_dropout_0)
        mutable_all_fc_activation_1 = self.mutable_all_fc_activation_1(mutable_all_fc_1)
        mutable_all_fc_dropout_1 = self.mutable_all_fc_dropout_1(mutable_all_fc_activation_1)
        mutable_all_fc_2 = self.mutable_all_fc_2(mutable_all_fc_dropout_1)
        return mutable_all_fc_2



class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = _model__layers()
        self._mapping_ = {'_layers': 'layers'}

    def forward(self, x__1):
        _layers = self._layers(x__1)
        return _layers