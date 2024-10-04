import torch
import torch.nn.functional as F
from torch import nn  # , einsum
import nni, math
# from einops.layers.torch import Rearrange
# from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
# allow_ops_in_compiled_graph()
# feedforward and attention



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
            nn.Linear(dim, int(dim * mult * 2)),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), dim)
        )
    def forward(self,x):
        return self.layers(x)


@ nni.retiarii.basic_unit
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

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        # self.rearrange1 = Rearrange('b n (h d) -> b h n d', h=self.heads)
        # self.rearrange2 = Rearrange('b h n d -> b n (h d)', h=self.heads)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        b = q.shape[0]
        n = q.shape[1]
        d = int(q.shape[2]/h)
        q = q.view(b, n, h, d).permute(0, 2, 1, 3)

        b = k.shape[0]
        n = k.shape[1]
        d = int(k.shape[2]/h)
        k = k.view(b, n, h, d).permute(0, 2, 1, 3)

        b = v.shape[0]
        n = v.shape[1]
        d = int(v.shape[2]/h)
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
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = x.unsqueeze(-1)

        return x * self.weights + self.biases

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

    def __init__(self, d_token: int, initialization: str) -> None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        _initialize_kaiming(self.weight, initialization, d_sqrt_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)