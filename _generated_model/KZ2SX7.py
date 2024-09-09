import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.retiarii.nn.pytorch

import torch


class _model__resnetblocks__blocks__0__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_0_0 = torch.nn.modules.batchnorm.BatchNorm1d(num_features=64)
        self._mapping_ = {'layerchoice_resnetblock_0_0': 'resnetblocks.blocks.0.batchnorm_1.0'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_0_0 = self.layerchoice_resnetblock_0_0(_inputs[0])
        return layerchoice_resnetblock_0_0



class _model__resnetblocks__blocks__0(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__0__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.1)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.2)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.0.batchnorm_1', '_linear_1': 'resnetblocks.blocks.0.linear_1', '_relu_1': 'resnetblocks.blocks.0.relu_1', '_dropout_1': 'resnetblocks.blocks.0.dropout_1', '_linear_2': 'resnetblocks.blocks.0.linear_2', '_dropout_2': 'resnetblocks.blocks.0.dropout_2'}

    def forward(self, x__1):
        _Constant14 = 1
        _aten__squeeze17 = x__1.squeeze(dim=_Constant14)
        _aten__squeeze15 = x__1.squeeze(dim=_Constant14)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze17)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add23 = _aten__squeeze15.add(other=_dropout_2, alpha=_Constant14)
        return _aten__add23



class _model__resnetblocks__blocks__1__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_1_1 = torch.nn.modules.linear.Identity()
        self._mapping_ = {'layerchoice_resnetblock_1_1': 'resnetblocks.blocks.1.batchnorm_1.1'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_1_1 = self.layerchoice_resnetblock_1_1(_inputs[0])
        return layerchoice_resnetblock_1_1



class _model__resnetblocks__blocks__1(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__1__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.9)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.9)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.1.batchnorm_1', '_linear_1': 'resnetblocks.blocks.1.linear_1', '_relu_1': 'resnetblocks.blocks.1.relu_1', '_dropout_1': 'resnetblocks.blocks.1.dropout_1', '_linear_2': 'resnetblocks.blocks.1.linear_2', '_dropout_2': 'resnetblocks.blocks.1.dropout_2'}

    def forward(self, x__1):
        _Constant24 = 1
        _aten__squeeze27 = x__1.squeeze(dim=_Constant24)
        _aten__squeeze25 = x__1.squeeze(dim=_Constant24)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze27)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add33 = _aten__squeeze25.add(other=_dropout_2, alpha=_Constant24)
        return _aten__add33



class _model__resnetblocks__blocks__2__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_2_1 = torch.nn.modules.linear.Identity()
        self._mapping_ = {'layerchoice_resnetblock_2_1': 'resnetblocks.blocks.2.batchnorm_1.1'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_2_1 = self.layerchoice_resnetblock_2_1(_inputs[0])
        return layerchoice_resnetblock_2_1



class _model__resnetblocks__blocks__2(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__2__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.8)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.9)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.2.batchnorm_1', '_linear_1': 'resnetblocks.blocks.2.linear_1', '_relu_1': 'resnetblocks.blocks.2.relu_1', '_dropout_1': 'resnetblocks.blocks.2.dropout_1', '_linear_2': 'resnetblocks.blocks.2.linear_2', '_dropout_2': 'resnetblocks.blocks.2.dropout_2'}

    def forward(self, x__1):
        _Constant34 = 1
        _aten__squeeze37 = x__1.squeeze(dim=_Constant34)
        _aten__squeeze35 = x__1.squeeze(dim=_Constant34)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze37)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add43 = _aten__squeeze35.add(other=_dropout_2, alpha=_Constant34)
        return _aten__add43



class _model__resnetblocks__blocks__3__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_3_0 = torch.nn.modules.batchnorm.BatchNorm1d(num_features=64)
        self._mapping_ = {'layerchoice_resnetblock_3_0': 'resnetblocks.blocks.3.batchnorm_1.0'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_3_0 = self.layerchoice_resnetblock_3_0(_inputs[0])
        return layerchoice_resnetblock_3_0



class _model__resnetblocks__blocks__3(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__3__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.1)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.6)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.3.batchnorm_1', '_linear_1': 'resnetblocks.blocks.3.linear_1', '_relu_1': 'resnetblocks.blocks.3.relu_1', '_dropout_1': 'resnetblocks.blocks.3.dropout_1', '_linear_2': 'resnetblocks.blocks.3.linear_2', '_dropout_2': 'resnetblocks.blocks.3.dropout_2'}

    def forward(self, x__1):
        _Constant44 = 1
        _aten__squeeze47 = x__1.squeeze(dim=_Constant44)
        _aten__squeeze45 = x__1.squeeze(dim=_Constant44)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze47)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add53 = _aten__squeeze45.add(other=_dropout_2, alpha=_Constant44)
        return _aten__add53



class _model__resnetblocks__blocks__4__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_4_0 = torch.nn.modules.batchnorm.BatchNorm1d(num_features=64)
        self._mapping_ = {'layerchoice_resnetblock_4_0': 'resnetblocks.blocks.4.batchnorm_1.0'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_4_0 = self.layerchoice_resnetblock_4_0(_inputs[0])
        return layerchoice_resnetblock_4_0



class _model__resnetblocks__blocks__4(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__4__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.6)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.6)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.4.batchnorm_1', '_linear_1': 'resnetblocks.blocks.4.linear_1', '_relu_1': 'resnetblocks.blocks.4.relu_1', '_dropout_1': 'resnetblocks.blocks.4.dropout_1', '_linear_2': 'resnetblocks.blocks.4.linear_2', '_dropout_2': 'resnetblocks.blocks.4.dropout_2'}

    def forward(self, x__1):
        _Constant54 = 1
        _aten__squeeze57 = x__1.squeeze(dim=_Constant54)
        _aten__squeeze55 = x__1.squeeze(dim=_Constant54)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze57)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add63 = _aten__squeeze55.add(other=_dropout_2, alpha=_Constant54)
        return _aten__add63



class _model__resnetblocks__blocks__5__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_5_1 = torch.nn.modules.linear.Identity()
        self._mapping_ = {'layerchoice_resnetblock_5_1': 'resnetblocks.blocks.5.batchnorm_1.1'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_5_1 = self.layerchoice_resnetblock_5_1(_inputs[0])
        return layerchoice_resnetblock_5_1



class _model__resnetblocks__blocks__5(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__5__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.1)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.2)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.5.batchnorm_1', '_linear_1': 'resnetblocks.blocks.5.linear_1', '_relu_1': 'resnetblocks.blocks.5.relu_1', '_dropout_1': 'resnetblocks.blocks.5.dropout_1', '_linear_2': 'resnetblocks.blocks.5.linear_2', '_dropout_2': 'resnetblocks.blocks.5.dropout_2'}

    def forward(self, x__1):
        _Constant64 = 1
        _aten__squeeze65 = x__1.squeeze(dim=_Constant64)
        _aten__squeeze67 = x__1.squeeze(dim=_Constant64)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze67)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add73 = _aten__squeeze65.add(other=_dropout_2, alpha=_Constant64)
        return _aten__add73



class _model__resnetblocks__blocks__6__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_6_1 = torch.nn.modules.linear.Identity()
        self._mapping_ = {'layerchoice_resnetblock_6_1': 'resnetblocks.blocks.6.batchnorm_1.1'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_6_1 = self.layerchoice_resnetblock_6_1(_inputs[0])
        return layerchoice_resnetblock_6_1



class _model__resnetblocks__blocks__6(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__6__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.9)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.6)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.6.batchnorm_1', '_linear_1': 'resnetblocks.blocks.6.linear_1', '_relu_1': 'resnetblocks.blocks.6.relu_1', '_dropout_1': 'resnetblocks.blocks.6.dropout_1', '_linear_2': 'resnetblocks.blocks.6.linear_2', '_dropout_2': 'resnetblocks.blocks.6.dropout_2'}

    def forward(self, x__1):
        _Constant74 = 1
        _aten__squeeze77 = x__1.squeeze(dim=_Constant74)
        _aten__squeeze75 = x__1.squeeze(dim=_Constant74)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze77)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add83 = _aten__squeeze75.add(other=_dropout_2, alpha=_Constant74)
        return _aten__add83



class _model__resnetblocks__blocks__7__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_7_0 = torch.nn.modules.batchnorm.BatchNorm1d(num_features=64)
        self._mapping_ = {'layerchoice_resnetblock_7_0': 'resnetblocks.blocks.7.batchnorm_1.0'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_7_0 = self.layerchoice_resnetblock_7_0(_inputs[0])
        return layerchoice_resnetblock_7_0



class _model__resnetblocks__blocks__7(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__7__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.1)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.6)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.7.batchnorm_1', '_linear_1': 'resnetblocks.blocks.7.linear_1', '_relu_1': 'resnetblocks.blocks.7.relu_1', '_dropout_1': 'resnetblocks.blocks.7.dropout_1', '_linear_2': 'resnetblocks.blocks.7.linear_2', '_dropout_2': 'resnetblocks.blocks.7.dropout_2'}

    def forward(self, x__1):
        _Constant84 = 1
        _aten__squeeze85 = x__1.squeeze(dim=_Constant84)
        _aten__squeeze87 = x__1.squeeze(dim=_Constant84)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze87)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add93 = _aten__squeeze85.add(other=_dropout_2, alpha=_Constant84)
        return _aten__add93



class _model__resnetblocks__blocks__8__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_8_0 = torch.nn.modules.batchnorm.BatchNorm1d(num_features=64)
        self._mapping_ = {'layerchoice_resnetblock_8_0': 'resnetblocks.blocks.8.batchnorm_1.0'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_8_0 = self.layerchoice_resnetblock_8_0(_inputs[0])
        return layerchoice_resnetblock_8_0



class _model__resnetblocks__blocks__8(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__8__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.1)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.9)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.8.batchnorm_1', '_linear_1': 'resnetblocks.blocks.8.linear_1', '_relu_1': 'resnetblocks.blocks.8.relu_1', '_dropout_1': 'resnetblocks.blocks.8.dropout_1', '_linear_2': 'resnetblocks.blocks.8.linear_2', '_dropout_2': 'resnetblocks.blocks.8.dropout_2'}

    def forward(self, x__1):
        _Constant94 = 1
        _aten__squeeze95 = x__1.squeeze(dim=_Constant94)
        _aten__squeeze97 = x__1.squeeze(dim=_Constant94)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze97)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add103 = _aten__squeeze95.add(other=_dropout_2, alpha=_Constant94)
        return _aten__add103



class _model__resnetblocks__blocks__9__batchnorm_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnetblock_9_1 = torch.nn.modules.linear.Identity()
        self._mapping_ = {'layerchoice_resnetblock_9_1': 'resnetblocks.blocks.9.batchnorm_1.1'}

    def forward(self, *_inputs):
        layerchoice_resnetblock_9_1 = self.layerchoice_resnetblock_9_1(_inputs[0])
        return layerchoice_resnetblock_9_1



class _model__resnetblocks__blocks__9(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm_1 = _model__resnetblocks__blocks__9__batchnorm_1()
        self._linear_1 = torch.nn.modules.linear.Linear(in_features=64, out_features=32)
        self._relu_1 = torch.nn.modules.activation.ReLU()
        self._dropout_1 = torch.nn.modules.dropout.Dropout(p=0.6)
        self._linear_2 = torch.nn.modules.linear.Linear(in_features=32, out_features=64)
        self._dropout_2 = torch.nn.modules.dropout.Dropout(p=0.9)
        self._mapping_ = {'_batchnorm_1': 'resnetblocks.blocks.9.batchnorm_1', '_linear_1': 'resnetblocks.blocks.9.linear_1', '_relu_1': 'resnetblocks.blocks.9.relu_1', '_dropout_1': 'resnetblocks.blocks.9.dropout_1', '_linear_2': 'resnetblocks.blocks.9.linear_2', '_dropout_2': 'resnetblocks.blocks.9.dropout_2'}

    def forward(self, x__1):
        _Constant104 = 1
        _aten__squeeze105 = x__1.squeeze(dim=_Constant104)
        _aten__squeeze107 = x__1.squeeze(dim=_Constant104)
        _batchnorm_1 = self._batchnorm_1(_aten__squeeze107)
        _linear_1 = self._linear_1(_batchnorm_1)
        _relu_1 = self._relu_1(_linear_1)
        _dropout_1 = self._dropout_1(_relu_1)
        _linear_2 = self._linear_2(_dropout_1)
        _dropout_2 = self._dropout_2(_linear_2)
        _aten__add113 = _aten__squeeze105.add(other=_dropout_2, alpha=_Constant104)
        return _aten__add113



class _model__resnetblocks(nn.Module):
    def __init__(self):
        super().__init__()
        self._blocks__0 = _model__resnetblocks__blocks__0()
        self._blocks__1 = _model__resnetblocks__blocks__1()
        self._blocks__2 = _model__resnetblocks__blocks__2()
        self._blocks__3 = _model__resnetblocks__blocks__3()
        self._blocks__4 = _model__resnetblocks__blocks__4()
        self._blocks__5 = _model__resnetblocks__blocks__5()
        self._mapping_ = {'_blocks__0': 'resnetblocks.blocks.0', '_blocks__1': 'resnetblocks.blocks.1', '_blocks__2': 'resnetblocks.blocks.2', '_blocks__3': 'resnetblocks.blocks.3', '_blocks__4': 'resnetblocks.blocks.4', '_blocks__5': 'resnetblocks.blocks.5'}

    def forward(self, x__1):
        _blocks__0 = self._blocks__0(x__1)
        _blocks__1 = self._blocks__1(_blocks__0)
        _blocks__2 = self._blocks__2(_blocks__1)
        _blocks__3 = self._blocks__3(_blocks__2)
        _blocks__4 = self._blocks__4(_blocks__3)
        _blocks__5 = self._blocks__5(_blocks__4)
        return _blocks__5



class _model__head__batchnorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_resnet_head_1 = torch.nn.modules.linear.Identity()
        self._mapping_ = {'layerchoice_resnet_head_1': 'head.batchnorm.1'}

    def forward(self, *_inputs):
        layerchoice_resnet_head_1 = self.layerchoice_resnet_head_1(_inputs[0])
        return layerchoice_resnet_head_1



class _model__head(nn.Module):
    def __init__(self):
        super().__init__()
        self._batchnorm = _model__head__batchnorm()
        self._relu = torch.nn.modules.activation.ReLU()
        self._linear = torch.nn.modules.linear.Linear(in_features=64, out_features=1)
        self._mapping_ = {'_batchnorm': 'head.batchnorm', '_relu': 'head.relu', '_linear': 'head.linear'}

    def forward(self, x__1):
        _Constant115 = 1
        _aten__squeeze117 = x__1.squeeze(dim=_Constant115)
        _batchnorm = self._batchnorm(_aten__squeeze117)
        _relu = self._relu(_batchnorm)
        _linear = self._linear(_relu)
        return _linear



class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._first_layer = torch.nn.modules.linear.Linear(in_features=33, out_features=64)
        self._resnetblocks = _model__resnetblocks()
        self._head = _model__head()
        self._mapping_ = {'_first_layer': 'first_layer', '_resnetblocks': 'resnetblocks', '_head': 'head'}

    def forward(self, x__1):
        _first_layer = self._first_layer(x__1)
        _resnetblocks = self._resnetblocks(_first_layer)
        _head = self._head(_resnetblocks)
        return _head