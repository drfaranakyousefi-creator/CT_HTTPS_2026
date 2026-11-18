"""
Client-side segments of CT-HTTPS (matches Fig. 1 of the paper):
  * ClientFront : per-feature AutoEncoders  (raw values never leave this module)
  * ClientBack  : final regression layer    (labels never leave the client)
"""
import torch
import torch.nn as nn


class FeatureAutoEncoder(nn.Module):
    """1 -> 4 -> latent -> 4 -> 1  (9*latent + 17 parameters)."""
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x):                      # x: (B, W, 1)
        z = self.encoder(x)
        return z, self.decoder(z)


class ClientFront(nn.Module):
    def __init__(self, n_features, d_model=16):
        super().__init__()
        self.n_features, self.d_model = n_features, d_model
        self.feature_aes = nn.ModuleList([FeatureAutoEncoder(d_model) for _ in range(n_features)])

    def forward(self, x, pad_mask):
        """x: (B,W,N), pad_mask: (B,W) with 1 = valid.
        Returns z (B,W,N*d_model) and the reconstruction loss (valid positions only)."""
        zs, rec = [], 0.0
        m = pad_mask.unsqueeze(-1)
        denom = m.sum().clamp(min=1.0)
        for i, ae in enumerate(self.feature_aes):
            xi = x[:, :, i:i + 1]
            z, xh = ae(xi)
            zs.append(z)
            rec = rec + (((xh - xi) * m) ** 2).sum() / denom
        return torch.cat(zs, dim=-1), rec / self.n_features


class ClientBack(nn.Module):
    """Final regression on the client: capsule output (d_in) -> scalar prediction."""
    def __init__(self, d_in):
        super().__init__()
        self.fc = nn.Linear(d_in, 1)

    def forward(self, h):
        return self.fc(h).squeeze(-1)
