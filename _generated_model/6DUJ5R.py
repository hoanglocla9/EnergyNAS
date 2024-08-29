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
        self.mutable_all_fc_activation_0 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_1 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_1 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_2 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_2 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_3 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_3 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_4 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_4 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_5 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_5 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_6 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_6 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_7 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_7 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_8 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_8 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_9 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_9 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_10 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_10 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_11 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_11 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_12 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=32)
        self.mutable_all_fc_activation_12 = nni.retiarii.nn.pytorch.ReLU()
        self.mutable_all_fc_13 = nni.retiarii.nn.pytorch.Linear(in_features=32, out_features=1)
        self._mapping_ = {'mutable_all_0': 'layers.0', 'mutable_all_fc_0': None, 'mutable_all_fc_activation_0': None, 'mutable_all_fc_1': None, 'mutable_all_fc_activation_1': None, 'mutable_all_fc_2': None, 'mutable_all_fc_activation_2': None, 'mutable_all_fc_3': None, 'mutable_all_fc_activation_3': None, 'mutable_all_fc_4': None, 'mutable_all_fc_activation_4': None, 'mutable_all_fc_5': None, 'mutable_all_fc_activation_5': None, 'mutable_all_fc_6': None, 'mutable_all_fc_activation_6': None, 'mutable_all_fc_7': None, 'mutable_all_fc_activation_7': None, 'mutable_all_fc_8': None, 'mutable_all_fc_activation_8': None, 'mutable_all_fc_9': None, 'mutable_all_fc_activation_9': None, 'mutable_all_fc_10': None, 'mutable_all_fc_activation_10': None, 'mutable_all_fc_11': None, 'mutable_all_fc_activation_11': None, 'mutable_all_fc_12': None, 'mutable_all_fc_activation_12': None, 'mutable_all_fc_13': None}

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
        mutable_all_fc_activation_4 = self.mutable_all_fc_activation_4(mutable_all_fc_4)
        mutable_all_fc_5 = self.mutable_all_fc_5(mutable_all_fc_activation_4)
        mutable_all_fc_activation_5 = self.mutable_all_fc_activation_5(mutable_all_fc_5)
        mutable_all_fc_6 = self.mutable_all_fc_6(mutable_all_fc_activation_5)
        mutable_all_fc_activation_6 = self.mutable_all_fc_activation_6(mutable_all_fc_6)
        mutable_all_fc_7 = self.mutable_all_fc_7(mutable_all_fc_activation_6)
        mutable_all_fc_activation_7 = self.mutable_all_fc_activation_7(mutable_all_fc_7)
        mutable_all_fc_8 = self.mutable_all_fc_8(mutable_all_fc_activation_7)
        mutable_all_fc_activation_8 = self.mutable_all_fc_activation_8(mutable_all_fc_8)
        mutable_all_fc_9 = self.mutable_all_fc_9(mutable_all_fc_activation_8)
        mutable_all_fc_activation_9 = self.mutable_all_fc_activation_9(mutable_all_fc_9)
        mutable_all_fc_10 = self.mutable_all_fc_10(mutable_all_fc_activation_9)
        mutable_all_fc_activation_10 = self.mutable_all_fc_activation_10(mutable_all_fc_10)
        mutable_all_fc_11 = self.mutable_all_fc_11(mutable_all_fc_activation_10)
        mutable_all_fc_activation_11 = self.mutable_all_fc_activation_11(mutable_all_fc_11)
        mutable_all_fc_12 = self.mutable_all_fc_12(mutable_all_fc_activation_11)
        mutable_all_fc_activation_12 = self.mutable_all_fc_activation_12(mutable_all_fc_12)
        mutable_all_fc_13 = self.mutable_all_fc_13(mutable_all_fc_activation_12)
        return mutable_all_fc_13



class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = _model__layers()
        self._mapping_ = {'_layers': 'layers'}

    def forward(self, x__1):
        _layers = self._layers(x__1)
        return _layers