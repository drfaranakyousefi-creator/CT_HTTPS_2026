"""
Server core of CT-HTTPS: Transformer encoder with [CLS] token + Capsule Network (dynamic routing).
Ablation switches (all default to the full model):
  use_transformer : False -> masked mean pooling of the latent series (no temporal modelling)
  use_cls         : False -> masked mean pooling of the encoder output instead of the [CLS] token
  use_pos         : False -> no positional encoding
  use_capsule     : False -> parameter-matched MLP head (2 x Linear(64,64)+LeakyReLU) instead of
                    primary capsules + dynamic routing
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x, dim=-1):
    n2 = (x ** 2).sum(dim=dim, keepdim=True)
    return (n2 / (1.0 + n2)) * x / (n2.sqrt() + 1e-8)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerCell(nn.Module):
    def __init__(self, d, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d), nn.Dropout(dropout))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, kpm):
        a, _ = self.attn(x, x, x, key_padding_mask=kpm, need_weights=False)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class PrimaryCaps(nn.Module):
    def __init__(self, d_in, n_caps, caps_dim):
        super().__init__()
        self.n_caps, self.caps_dim = n_caps, caps_dim
        self.lin = nn.Linear(d_in, n_caps * caps_dim)

    def forward(self, x):
        return squash(self.lin(x).view(x.size(0), self.n_caps, self.caps_dim))


class Routing(nn.Module):
    def __init__(self, n_in, d_in, n_out, d_out, iters=3):
        super().__init__()
        self.iters, self.n_out = iters, n_out
        self.W = nn.Parameter(torch.randn(1, n_in, n_out, d_out, d_in) * 0.01)

    def forward(self, x):                                   # x: (B, n_in, d_in)
        u_hat = torch.matmul(self.W, x.unsqueeze(2).unsqueeze(4)).squeeze(-1)   # (B,n_in,n_out,d_out)
        b = torch.zeros(x.size(0), x.size(1), self.n_out, device=x.device)
        for _ in range(self.iters):
            c = F.softmax(b, dim=2)
            v = squash((c.unsqueeze(-1) * u_hat).sum(dim=1))
            b = b + (u_hat * v.unsqueeze(1)).sum(-1)
        return v                                            # (B, n_out, d_out)


class ServerCore(nn.Module):
    def __init__(self, D, w, nhead=4, d_ff=32, num_cells=3, dropout=0.1,
                 fc_hidden1=128, fc_hidden2=64, num_primary_caps=8, primary_dim=8,
                 num_output_caps=4, output_dim=16, num_routing=3,
                 use_transformer=True, use_cls=True, use_pos=True, use_capsule=True):
        super().__init__()
        self.use_transformer, self.use_cls = use_transformer, use_cls
        self.use_pos, self.use_capsule = use_pos, use_capsule
        self.out_dim = num_output_caps * output_dim

        if use_transformer and use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, D))
        self.pos = PositionalEncoding(D, max_len=w + 2, dropout=dropout) if use_pos else None
        self.cells = nn.ModuleList([TransformerCell(D, nhead, d_ff, dropout) for _ in range(num_cells)]) \
            if use_transformer else nn.ModuleList()

        self.fc_in = nn.Sequential(nn.Linear(D, fc_hidden1), nn.LeakyReLU(),
                                   nn.Linear(fc_hidden1, fc_hidden2), nn.LeakyReLU())
        if use_capsule:
            self.primary = PrimaryCaps(fc_hidden2, num_primary_caps, primary_dim)
            self.routing = Routing(num_primary_caps, primary_dim, num_output_caps, output_dim, num_routing)
        else:
            self.mlp_head = nn.Sequential(nn.Linear(fc_hidden2, self.out_dim), nn.LeakyReLU(),
                                          nn.Linear(self.out_dim, self.out_dim), nn.LeakyReLU())

    @staticmethod
    def _masked_mean(x, m):                                  # x (B,T,D), m (B,T)
        return (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)

    def forward(self, z, pad_mask):
        """z: (B,W,D) latent series received from the client, pad_mask: (B,W)."""
        B = z.size(0)
        if self.use_transformer:
            if self.use_cls:
                x = torch.cat([self.cls_token.expand(B, -1, -1), z], dim=1)
                m_full = torch.cat([torch.ones(B, 1, device=z.device, dtype=pad_mask.dtype), pad_mask], 1)
            else:
                x, m_full = z, pad_mask
            if self.pos is not None:
                x = self.pos(x)
            x = x * m_full.unsqueeze(-1)
            kpm = m_full < 0.5
            for cell in self.cells:
                x = cell(x, kpm) * m_full.unsqueeze(-1)
            pooled = x[:, 0] if self.use_cls else self._masked_mean(x, m_full)
        else:
            x = self.pos(z) if self.pos is not None else z
            pooled = self._masked_mean(x, pad_mask)

        h = self.fc_in(pooled)
        if self.use_capsule:
            return self.routing(self.primary(h)).reshape(B, -1)
        return self.mlp_head(h)
