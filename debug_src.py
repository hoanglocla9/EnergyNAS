import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.nas.nn.pytorch

import torch
import nas


class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._tokenizer = nas.model.Tokenizer(d_numerical=33, d_token=192)
        self._encoder = torch.nn.modules.transformer.TransformerEncoderLayer(d_model=192, nhead=4, dim_feedforward=192, dropout=0.3, norm_first=False)
        self._last_normalization = torch.nn.modules.linear.Identity()
        self._head = torch.nn.modules.linear.Linear(in_features=192, out_features=1)
        self._mapping_ = {'_tokenizer': 'tokenizer', '_encoder': 'encoder', '_last_normalization': 'last_normalization', '_head': 'head'}

    def forward(self, x__1):
        _Constant1 = -1
        _Constant3 = False
        _Constant4 = None
        _tokenizer = self._tokenizer(x__1)
        _encoder = self._encoder(_tokenizer, _Constant4, _Constant4, _Constant3)
        _last_normalization = self._last_normalization(_encoder)
        _relu8 = F.relu(_last_normalization, _Constant3)
        _head = self._head(_relu8)
        _aten__squeeze10 = _head.squeeze(dim=_Constant1)
        return _aten__squeeze10

x = torch.rand((1, 33))
model = _model()
model(x)