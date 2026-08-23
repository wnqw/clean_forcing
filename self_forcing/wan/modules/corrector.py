"""
Latent-space velocity-residual corrector r_phi for the Wan2.1-1.3B DF base (idea #6).

Port of the toy `algorithms/diffusion_forcing/corrector.py` (VelocityResidualCorrector)
to the Wan latent DiT: a lightweight, history-conditioned network that outputs a residual
on the base velocity (flow_pred). Injected in CausalDiffusionInferencePipeline as
    flow_pred_rect = flow_pred + alpha(t) * r_phi(z_t, history, t)
alpha=0 (or zero-init output) == baseline.

Input/output operate on Wan latents: (B, F, C=16, H=60, W=104); timestep (B, F).
"""

import math
import torch
from torch import nn


def _timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[..., None] * freqs
    return torch.cat([a.cos(), a.sin()], dim=-1)


class WanVelocityResidual(nn.Module):
    def __init__(self, channels=16, hidden=128, t_dim=128, n_blocks=4):
        super().__init__()
        self.t_dim = t_dim
        self.in_conv = nn.Conv3d(channels, hidden, 3, padding=1)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden))  # FiLM
        self.hist_pool = nn.Sequential(nn.Conv3d(channels, hidden, 1), nn.SiLU())  # committed history -> cond vec
        self.blocks = nn.ModuleList([nn.Conv3d(hidden, hidden, 3, padding=1) for _ in range(n_blocks)])
        self.act = nn.SiLU()
        self.out_conv = nn.Conv3d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)  # zero-init -> r_phi starts as the identity correction (== baseline)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, timestep, history=None):
        # x:(B,F,C,H,W)  timestep:(B,F)  history:(B,Fh,C,H,W) or None
        feat = self.in_conv(x.permute(0, 2, 1, 3, 4).float())              # (B,hidden,F,H,W)
        scale, shift = self.t_mlp(_timestep_embedding(timestep.float().mean(1), self.t_dim)).chunk(2, -1)
        feat = feat * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        if history is not None and history.shape[1] > 0:
            hp = self.hist_pool(history.permute(0, 2, 1, 3, 4).float()).mean(dim=(2, 3, 4))  # (B,hidden)
            feat = feat + hp[:, :, None, None, None]
        for blk in self.blocks:
            feat = feat + self.act(blk(feat))
        out = self.out_conv(feat)                                          # (B,C,F,H,W)
        return out.permute(0, 2, 1, 3, 4).to(x.dtype)                      # (B,F,C,H,W)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
