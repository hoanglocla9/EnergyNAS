import torch
import torch.nn.functional as F
from torch import nn  # , einsum
import nni
# from einops.layers.torch import Rearrange
# from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
# allow_ops_in_compiled_graph()
# feedforward and attention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )


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

        q_ = q
        b = q.shape[0]
        n = q.shape[1]
        d = int(q.shape[2]/h)
        q = q.view(b, n, h, d).permute(0, 2, 1, 3)

        k_ = k
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

        return out, attn

# transformer


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head,
                          dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout),
            ]))

    def forward(self, x):
        post_softmax_attns = []

        for layers in self.layers:
            attn = layers[0]
            ff = layers[1]
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        return x, torch.stack(post_softmax_attns)

# numerical embedder


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = x.unsqueeze(-1)

        return x * self.weights + self.biases

# main class


@ nni.retiarii.basic_unit
class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        num_special_tokens=2,
        attn_dropout=0.,
        ff_dropout=0.,
        batch_size=32
    ):
        super().__init__()
        # assert all(map(lambda n: n > 0, categories)
        #            ), 'number of each category must be positive'
        assert len(categories) + \
            num_continuous > 0, 'input shape must not be null'

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(
                dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_numer):  # , return_attn=False

        xs = []
        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        x = torch.cat(xs, dim=1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # attend

        x, attns = self.transformer(x)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        return logits  # , attns
