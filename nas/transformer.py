import torch
import torch.nn.functional as F
# from torch import nn  # , einsum

import nni.retiarii.nn.pytorch as nn
import nni, math
from typing import cast, Any
from nni.nas.oneshot.pytorch.supermodule.operation import MixedOperation
from nni.nas.oneshot.pytorch.supermodule._valuechoice_utils import traverse_all_options
from nni.nas.nn.pytorch.choice import ValueChoiceX
from nni.nas.oneshot.pytorch.supermodule._operation_utils import Slicable as _S, MaybeWeighted as _W

from nni.nas.hub.pytorch.utils.fixed import FixedFactory
@ nni.retiarii.basic_unit
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)



@ nni.retiarii.basic_unit
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.layers= nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mult * 2)), # cast(int, nn.ValueChoice.to_int(dim * mult * 2))
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), dim) # cast(int, nn.ValueChoice.to_int(dim * mult))
        )
    def forward(self,x):
        return self.layers(x)
    
class FeedForwardOneShot(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.layers= nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, cast(int, nn.ValueChoice.to_int(dim * mult * 2))), # 
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(cast(int, nn.ValueChoice.to_int(dim * mult)), dim) # 
        )
    def forward(self,x):
        return self.layers(x)

# @ nni.retiarii.basic_unit
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h =  -1 # cast(int, nn.ValueChoice.to_int(self.heads))
        d = self.dim_head
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        b = q.shape[0]
        n = q.shape[1] # cast(int, q.shape[2]/h)
        q = q.view(b, n, h, d).permute(0, 2, 1, 3)

        b = k.shape[0]
        n = k.shape[1]
        
        k = k.view(b, n, h, d).permute(0, 2, 1, 3)

        b = v.shape[0]
        n = v.shape[1]
        v = v.view(b, n, h, d).permute(0, 2, 1, 3)
        # q, k, v = map(lambda t: self.rearrange1(
        #     t), (q, k, v))
        q = q * self.scale

        # einsum('b h i d, b h j d -> b h i j', q, k)
        sim = torch.matmul(q, k.transpose(-1, -2))

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        # einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = torch.matmul(dropped_attn, v)
        b = out.shape[0]
        h = out.shape[1]
        n = out.shape[2]
        d = out.shape[3]
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)

        out = self.to_out(out)
        return out #, attn


@ nni.retiarii.basic_unit
class NumericalEmbedder(nn.Module):
    def __init__(self, embed_dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, embed_dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, embed_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)

        return x * self.weights + self.biases


class MixedNumericalEmbed(MixedOperation, NumericalEmbedder):
    bound_type = NumericalEmbedder
    argument_list = ['embed_dim']

    def super_init_argument(self, name: str, value_choice: ValueChoiceX):
        return max(traverse_all_options(value_choice))

    def slice_param(self, embed_dim, **kwargs) -> Any:
        embed_dim_ = _W(embed_dim)
        weights = _S(self.weights)[..., :embed_dim_]
        biases = _S(self.biases)[..., :embed_dim_]

        return {'weights': weights, 'biases': biases}

    def forward_with_args(self,  embed_dim,
                        inputs: torch.Tensor) -> torch.Tensor:
        weights = self.slice_param(embed_dim)['weights']
        biases = self.slice_param(embed_dim)['biases']
        assert isinstance(weights, torch.Tensor)
        assert isinstance(biases, torch.Tensor)

        return inputs.unsqueeze(-1) * weights + biases
    
# main class

def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == "kaiming_uniform":
        nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == "kaiming_normal":
        nn.init.normal_(x, std=d_sqrt_inv)
    elif initialization is None:
        pass
    else:
        raise NotImplementedError("initialization should be either of `kaiming_normal`, `kaiming_uniform`," " `None`")


@nni.retiarii.basic_unit
class CLSToken(nn.Module):
    """Appends the [CLS] token for BERT-like inference."""

    def __init__(self, embed_dim: int, initialization: str) -> None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(embed_dim))
        d_sqrt_inv = 1 / math.sqrt(embed_dim)
        _initialize_kaiming(self.weight, initialization, d_sqrt_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)
    

class MixedClsToken(MixedOperation, CLSToken):
    """ Mixed class token concat operation.

    Supported arguments are:

    - ``embed_dim``

    Prefix of cls_token will be sliced.
    """
    bound_type = CLSToken
    argument_list = ['embed_dim']

    def super_init_argument(self, name: str, value_choice: ValueChoiceX):
        return max(traverse_all_options(value_choice))

    def slice_param(self, embed_dim, **kwargs) -> Any:
        embed_dim_ = _W(embed_dim)
        weight = _S(self.weight)[..., :embed_dim_]

        return {'weight': weight}

    def forward_with_args(self, embed_dim,
                        inputs: torch.Tensor) -> torch.Tensor:
        weight = self.slice_param(embed_dim)['weight']
        assert isinstance(weight, torch.Tensor)
        return torch.cat([inputs, weight.view(1, 1, -1).repeat(len(inputs), 1, 1)], dim=1)
    

@nni.retiarii.model_wrapper
class FTTransformerSpace_OneShot(nn.Module):
    class TransformerBlock(nn.Module):
        def __init__(self, n_heads, dim_head, attn_dropout, ff_dropout, d_ffn_factor, dim_hidden):
            super().__init__()
            # dim_hidden = nn.ValueChoice([8, 16, 32, 64, 128, 192, 256], label=f"dim_hidden_{index}")
            
            self.norm0 = nn.LayerNorm(dim_hidden)
            self.attention = Attention(dim_hidden, heads=n_heads, dim_head=dim_head,
                            dropout=attn_dropout)
            # print(d_ffn_factor)
            
            self.ff = FeedForwardOneShot(dim_hidden , mult= d_ffn_factor, dropout=ff_dropout)
            # self.residual_dropout = nn.Dropout(p=residual_dropout)
            self.norm1 = nn.LayerNorm(dim_hidden)
            
            
        def forward(self, x):
            x = self.norm0(x)
            attn_out = self.attention(x)
            x = attn_out + x
            x = self.ff(x) + x
            x = self.norm1(x)
            
            return x
    
    class TransformerStage(nn.Module):
        def __init__(self, search_embed_dim, search_mlp_ratio, search_num_heads, search_depth, dim_hidden):
            super().__init__()
            candidate_depths = search_depth
            depth = nn.ValueChoice(candidate_depths, label="depth")
            n_heads = [nn.ValueChoice(search_num_heads, label=f"depth_{i}") for i in range(max(candidate_depths))]
            # dim_head = [nn.ValueChoice(search_embed_dim, label=f"dim_head_{index}") for index in range(max(candidate_depths))] 
            attn_dropout = 0.1 # [nn.ValueChoice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], label=f"attn_dropout_{index}")  for index in range(max(candidate_depths))]
            ff_dropout =  0.1 #[nn.ValueChoice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], label=f"ff_dropout_{index}") for index in range(max(candidate_depths))]
            d_ffn_factor = [nn.ValueChoice(search_mlp_ratio, label=f"d_ffn_factor_{index}") for index in range(max(candidate_depths))]
            
            self.blocks = nn.Repeat(lambda idx: FTTransformerSpace_OneShot.TransformerBlock(
                                    n_heads[idx], 64, attn_dropout, ff_dropout,  #  dim_head[idx]
                                    d_ffn_factor[idx],dim_hidden=dim_hidden), 
                                    depth, label="n_blocks")

        def forward(self, x):
            x = self.blocks(x)
            return x
    
    
    def __init__(self, 
                 n_features, 
                 search_embed_dim=[64, 128, 192, 256], 
                 search_mlp_ratio=[2.0, 2.5, 3.0, 3.5, 4.0], 
                 search_num_heads=[3, 4, 5, 6, 7, 8], 
                 search_hidden_dim=[8, 16, 32], 
                 search_depth=[2, 3, 4, 5]):
        super().__init__()
        num_continuous = n_features
        dim_out=1
        dim_hidden = nn.ValueChoice(search_hidden_dim, label=f"dim_hidden")
            
        self.numerical_embedder = NumericalEmbedder(cast(int, dim_hidden), num_continuous)
        self.cls_token = CLSToken(cast(int, dim_hidden), 'kaiming_normal')
        self.transformer = FTTransformerSpace_OneShot.TransformerStage(search_embed_dim, search_mlp_ratio, search_num_heads, search_depth, dim_hidden)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out)
        )
        
        self.drop_path_prob=0.0

    def forward(self, x_numer): 
        x = self.numerical_embedder(x_numer)
        cls_tokens = self.cls_token(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        logits = self.to_logits(x)
        return logits 
    
    
    def set_drop_path_prob(self, prob):
        pass

    @classmethod
    def get_extra_mutation_hooks(cls):
        return [MixedNumericalEmbed.mutate, MixedClsToken.mutate]
    
    @classmethod
    def preset(cls, name: str):
        """Get the model space config proposed in paper."""
        name = name.lower()
        assert name in ['tiny', 'small', 'base']
        init_kwargs = { } # , 'drop_path_rate': 0.1
        if name == 'tiny':
            init_kwargs.update({
                'search_embed_dim': (64, 128, 192, 256),
                'search_mlp_ratio': (2.0, 2.5, 3.0, 3.5, 4.0),
                'search_num_heads': (3, 4, 5, 6, 7, 8),
                'search_depth': (2, 3, 4, 5),
                'search_hidden_dim': (8, 16, 32)
            })
        elif name == 'small':
            init_kwargs.update({
                'search_embed_dim': (192, 256, 320),
                'search_mlp_ratio': (3.0, 3.5, 4.0),
                'search_num_heads': (8, 9, 10),
                'search_depth': (4, 5, 6, 7, 8),
                'search_hidden_dim': (32, 64, 128)
            })
        elif name == 'base':
            init_kwargs.update({
                'search_embed_dim': (320, 384, 448),
                'search_mlp_ratio': (3.0, 3.5, 4.0),
                'search_num_heads': (10, 12, 14),
                'search_depth': (6, 7, 8, 9, 10),
                'search_hidden_dim': (64, 128, 256, 512)
            })
        else:
            raise ValueError(f'Unsupported architecture with name: {name}')
        return init_kwargs
    

    @classmethod
    def load_searched_model(cls, arch_config):
        """
        Load the searched subnet model.

        Parameters
        ----------
        name : str
            Search space size, must be one of {'ftformer-tiny', 'ftformer-small', 'ftformer-base'}.
        Returns
        -------
        nn.Module
            The subnet model.
        """
        legal = ['ftformer-tiny', 'ftformer-small', 'ftformer-base']
        if name not in legal:
            raise ValueError(f'Unsupported name: {name}. It should be one of {legal}.')
        name = name.split("-")[-1]
        init_kwargs = cls.preset(name)
        
        # if name == 'tiny':
        #     mlp_ratio = [3.5, 3.5, 3.0, 3.5, 3.0, 3.0, 4.0, 4.0, 3.5, 4.0, 3.5, 4.0, 3.5] + [3.0]
        #     num_head = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3] + [3]
        #     arch: Dict[str, Any] = {
        #         'embed_dim': 192,
        #         'depth': 13
        #     }
        #     for i in range(14):
        #         arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
        #         arch[f'num_head_{i}'] = num_head[i]
        # elif name == 'small':
        #     mlp_ratio = [3.0, 3.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 4.0] + [3.0]
        #     num_head = [6, 6, 5, 7, 5, 5, 5, 6, 6, 7, 7, 6, 7] + [5]
        #     arch: Dict[str, Any] = {
        #         'embed_dim': 384,
        #         'depth': 13
        #     }
        #     for i in range(14):
        #         arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
        #         arch[f'num_head_{i}'] = num_head[i]
        # elif name == 'base':
        #     mlp_ratio = [3.5, 3.5, 4.0, 3.5, 4.0, 3.5, 3.5, 3.0, 4.0, 4.0, 3.0, 4.0, 3.0, 3.5] + [3.0, 3.0]
        #     num_head = [9, 9, 9, 9, 9, 10, 9, 9, 10, 9, 10, 9, 9, 10] + [8, 8]
        #     arch: Dict[str, Any] = {
        #         'embed_dim': 576,
        #         'depth': 14
        #     }
        #     for i in range(16):
        #         arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
        #         arch[f'num_head_{i}'] = num_head[i]
        # else:
        #     raise ValueError(f'Unsupported architecture with name: {name}')

        model_factory = FixedFactory(cls, arch_config)
        model = model_factory(**init_kwargs)


        return model
    
    

@nni.retiarii.model_wrapper
class FTTransformerSpace(nn.Module):
    class TransformerBlock(nn.Module):
        def __init__(self, index, dim_hidden):
            super().__init__()
            n_heads = nn.ValueChoice(range(1,10), label=f"n_heads_{index}")
            dim_head = nn.ValueChoice([16, 32, 64, 128, 192, 256], label=f"dim_head_{index}")
            attn_dropout = nn.ValueChoice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], label=f"attn_dropout_{index}")
            ff_dropout = nn.ValueChoice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], label=f"ff_dropout_{index}")
            d_ffn_factor = nn.ValueChoice([1, 1.5, 2, 2.5, 3, 3.5, 4], label=f"d_ffn_factor_{index}")
            
            self.norm0 = nn.LayerNorm(dim_hidden)
            self.attention = nni.retiarii.basic_unit(Attention)(dim_hidden, heads=n_heads, dim_head=dim_head,
                            dropout=attn_dropout)
            # print(d_ffn_factor)
            self.ff = nni.retiarii.basic_unit(FeedForward)(dim_hidden, mult=d_ffn_factor, dropout=ff_dropout)
            # self.residual_dropout = nn.Dropout(p=residual_dropout)
            self.norm1 = nn.LayerNorm(dim_hidden)
            
            
        def forward(self, x):
            x = self.norm0(x)
            attn_out = self.attention(x)
            x = attn_out + x
            x = self.ff(x) + x
            x = self.norm1(x)
            
            return x
    
    class TransformerStage(nn.Module):
        def __init__(self, dim_hidden):
            super().__init__()
            self.blocks = nn.Repeat(lambda idx: FTTransformerSpace.TransformerBlock(index=idx, dim_hidden=dim_hidden), 
                                        (1, 8), label="n_blocks")

        def forward(self, x):
            x = self.blocks(x)
            return x
    
        
    def __init__(
        self,
        n_features
        
    ):
        super().__init__()
        num_continuous = n_features
        dim_out=1
        dim_hidden = nn.ValueChoice([8, 16, 32, 64, 128, 192, 256], label=f"dim_hidden")

        self.numerical_embedder = NumericalEmbedder(dim_hidden, num_continuous)
        self.cls_token = CLSToken(dim_hidden, 'kaiming_normal')
        self.transformer = FTTransformerSpace.TransformerStage(dim_hidden)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x_numer):
        x = self.numerical_embedder(x_numer)
        cls_tokens = self.cls_token(x)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        x = x[:, 0]

        logits = self.to_logits(x)

        return logits 