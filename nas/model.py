import nni
import nni.retiarii
import nni.retiarii.nn.pytorch as nn
import torch
import math
from torch import Tensor

import torch.optim
import torch.nn.functional as F
import typing as ty
import copy
import warnings
from typing import Optional, Union, Callable
from .transformer import FTTransformerSpace_OneShot, FTTransformerSpace

def get_activation_name_options():
    return [
        # "__torch__.nni.retiarii.nn.pytorch.Identity",
        "None",  # replace for Identity
        "__torch__.nni.retiarii.nn.pytorch.ReLU",
        "__torch__.nni.retiarii.nn.pytorch.Tanh",
        "__torch__.nni.retiarii.nn.pytorch.LeakyReLU",
        "__torch__.nni.retiarii.nn.pytorch.Hardshrink",
        "__torch__.nni.retiarii.nn.pytorch.Hardsigmoid",
        "__torch__.nni.retiarii.nn.pytorch.Hardtanh",
        "__torch__.nni.retiarii.nn.pytorch.Hardswish",
        "__torch__.nni.retiarii.nn.pytorch.LogSigmoid",
        "__torch__.nni.retiarii.nn.pytorch.PReLU",
        "__torch__.nni.retiarii.nn.pytorch.ELU",
        "__torch__.nni.retiarii.nn.pytorch.ReLU6",
        "__torch__.nni.retiarii.nn.pytorch.RReLU",
        "__torch__.nni.retiarii.nn.pytorch.CELU",
        "__torch__.nni.retiarii.nn.pytorch.SiLU",
        # "__torch__.nni.retiarii.nn.pytorch.Mish",
        "__torch__.nni.retiarii.nn.pytorch.Softplus",
        "__torch__.nni.retiarii.nn.pytorch.Softshrink",
        "__torch__.nni.retiarii.nn.pytorch.Softsign",
        "__torch__.nni.retiarii.nn.pytorch.Tanhshrink"
    ]


@nni.retiarii.model_wrapper
class CalibrationModelSpace(nn.Module):
    def __init__(self, n_features=33):  # ,
        super().__init__()
        layers = []
        self.n_features = n_features

        _layers = nn.Placeholder(
            label='mutable_all',
            conv_kernel_size_options=[1, 3, 5, 7],
            conv_n_layer_options=[1, 2, 3],
            conv_n_filter_options=range(8, 1025, 8),
            conv_op_type='__torch__.nni.retiarii.nn.pytorch.Conv1d',
            in_ch=n_features,
            conv_activation_fn_options=get_activation_name_options(),
            pooling_op_type_options=[
                "__torch__.nni.retiarii.nn.pytorch.MaxPool1d", "__torch__.nni.retiarii.nn.pytorch.AvgPool1d"],
            pooling_kernel_size_options=[3, 5, 7],
            flatten_options=[True, False],
            n_neurons_options=range(8, 1025, 8),
            fc_n_layer_options=list(range(1, 19)),
            fc_op_type='__torch__.nni.retiarii.nn.pytorch.Linear',
            fc_activation_fn_options=get_activation_name_options()
        )
        layers.append(_layers)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


@nni.retiarii.model_wrapper
class MLPSpace(nn.Module):
    class MLPBlock(nn.Module):
        def __init__(self, d_in, d_hidden, idx=-1):
            super().__init__()
            self.linear = nn.Linear(d_in, d_hidden)
            self.activation = nn.ReLU()
            if idx == -1:
                idx = "final"
            self.dropout = nn.Dropout(p=nn.ValueChoice(
                [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], label="mlpblock_" + str(idx) + "_droprate"))

        def forward(self, x):
            x = self.linear(x)
            x = self.activation(x)
            x = self.dropout(x)
            return x

    def __init__(self, n_features=33):  # ,
        super().__init__()
        layers = []
        self.n_features = n_features
        d_hidden = nn.ValueChoice(range(4, 1025, 4), label="d_hidden")
        self.blocks = nn.Repeat(lambda idx: MLPSpace.MLPBlock(n_features, d_hidden, idx) if idx == 0
                                else MLPSpace.MLPBlock(d_hidden, d_hidden, idx), depth=(1, 10), label="n_mlpblocks")

        self.final_block = MLPSpace.MLPBlock(d_hidden, 1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.final_block(x)
        return x


@nni.retiarii.model_wrapper
class ResNetSpace(nn.Module):
    class ResNetBlock(nn.Module):
        def __init__(self, idx, d_in=33):
            super().__init__()
            self.batchnorm_1 = nn.LayerChoice([
                nn.BatchNorm1d(d_in),
                nn.Identity()
            ], key="resnetblock_isbn_"+str(idx))
            hidden_dim = nn.ValueChoice(
                range(4, 513, 4), label="resnetblock_" + str(idx) + "_d_hidden")
            self.linear_1 = nn.Linear(d_in, hidden_dim)
            self.relu_1 = nn.ReLU()
            self.dropout_1 = nn.Dropout(p=nn.ValueChoice(
                [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], label="resnetblock_" + str(idx) + "_droprate_0"))
            self.linear_2 = nn.Linear(hidden_dim, d_in)
            self.dropout_2 = nn.Dropout(p=nn.ValueChoice(
                [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], label="resnetblock_" + str(idx) + "_droprate_1"))

        def forward(self, x):
            x_input = x.squeeze(1)
            x = self.batchnorm_1(x.squeeze(dim=1))
            x = self.linear_1(x)
            x = self.relu_1(x)
            x = self.dropout_1(x)
            x = self.linear_2(x)
            x = self.dropout_2(x)
            return x_input + x

    class ResNetHead(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.batchnorm = nn.LayerChoice([
                nn.BatchNorm1d(d_in),
                nn.Identity()
            ], key="resnethead_isbn")
            self.relu = nn.ReLU()
            self.linear = nn.Linear(d_in, d_out)

        def forward(self, x):
            x = self.batchnorm(x.squeeze(dim=1))
            x = self.relu(x)
            x = self.linear(x)
            return x

    def __init__(self, n_features=33):  # ,
        super().__init__()
        d_main = nn.ValueChoice(range(4, 257, 4), label="resnet_d_main")
        self.first_layer = nn.Linear(n_features, d_main)
        self.resnetblocks = nn.Repeat(lambda idx: ResNetSpace.ResNetBlock(
            idx, d_main), (1, 10), label="n_resnetblocks")
        self.head = ResNetSpace.ResNetHead(d_main, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.resnetblocks(x)
        x = self.head(x)
        return x


@nni.retiarii.basic_unit
class Tokenizer(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        d_token: int
    ) -> None:
        super().__init__()
        d_bias = d_numerical

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token))
        # The initialization is inspired by nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight)

    def forward(self, x_num: Tensor) -> Tensor:
        x_num = x_num.squeeze(1)
        x = self.weight[:-1][None] * x_num[:, :, None]

        x = torch.cat(
            [x, self.weight[-1][None, None].repeat(len(x), 1, 1)],
            dim=1,
        )
        bias = torch.cat(
            [
                self.bias,
                torch.zeros(
                    1, self.bias.shape[1], device=x_num.device),
            ]
        )
        x = x + bias[None]
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
):

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               bias=bias, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,
                                 bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model,
                                 bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps,
                                  bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps,
                                  bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            is_last_layer: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ''
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = [
                "cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(
                    src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask, is_causal=is_causal)
            if is_last_layer:
                x = x[:, -1:]
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask,
                           src_key_padding_mask, is_causal=is_causal))
            if is_last_layer:
                x = x[:, -1:]
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


@nni.retiarii.basic_unit
class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(
        self,
        d_token: int,
        num_layers: int,
        n_heads: int,
        d_hidden: int,
        layer_dropout_rate: float,
        activation_fn: str,
        prenormalization: bool,
        enable_nested_tensor: bool = True,
        mask_check: bool = True
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(
            f"torch.nn.modules.{self.__class__.__name__}")

        norm = nn.LayerNorm(d_token)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_hidden, dropout=layer_dropout_rate,
            activation=activation_fn, norm_first=prenormalization)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first:
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps):
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"
        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(
                f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )
        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")
            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None
        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        layer_idx = 0
        for mod in self.layers:
            if layer_idx == self.num_layers-1:
                output = mod(output, src_mask=mask, is_causal=is_causal,
                             src_key_padding_mask=src_key_padding_mask_for_layers, is_last_layer=True)
            else:
                output = mod(output, src_mask=mask, is_causal=is_causal,
                             src_key_padding_mask=src_key_padding_mask_for_layers)
            layer_idx += 1
        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())
        if self.norm is not None:
            output = self.norm(output)

        return output


@nni.retiarii.model_wrapper
class ConventionalTransformerSpace(nn.Module):
    def __init__(self, n_features,  d_out=1):
        super().__init__()
        d_token = nn.ValueChoice([64, 128, 192, 256], label="d_token")
        n_heads = nn.ValueChoice([2, 4, 8, 16, 32], label="n_heads")
        d_hidden = nn.ValueChoice([64, 128, 192, 256], label="d_hidden")
        layer_dropout_rate = nn.ValueChoice(
            [0.1, 0.2, 0.3, 0.4, 0.0], label="layer_dropout_rate")
        activation_fn = 'relu'
        prenormalization = nn.ValueChoice(
            [True, False], label="prenormalization")

        n_layers = nn.ValueChoice(range(2, 9), label="n_layers")
        self.tokenizer = Tokenizer(
            n_features, d_token
        )
        self.encoder = TransformerEncoder(
            d_token=d_token, num_layers=n_layers, n_heads=n_heads, d_hidden=d_hidden,
            layer_dropout_rate=layer_dropout_rate, activation_fn=activation_fn,
            prenormalization=prenormalization)
        self.last_activation = F.relu
        self.last_normalization = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.encoder(x)
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

    
    


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

