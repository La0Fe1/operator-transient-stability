"""Physics-informed neural operator (DeepONet) for transient-stability trajectories.

Architecture: a branch network encodes the fault scenario (fault bus, clearing
time, topology) into per-machine basis coefficients, and a *Fourier-feature*
trunk maps time to a shared basis. Fourier features (cos/sin harmonics) are
used as the trunk input because the swing dynamics are oscillatory; this lets
the trunk represent the natural modes far more efficiently than a plain MLP.

The physics loss imposes the post-fault swing-equation residual in the
center-of-inertia (COI) reference frame.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mlp(sizes: list[int], act=nn.Tanh) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class FourierTrunk(nn.Module):
    """Trunk mapping normalized time t in [0,1] to a basis of p functions.

    The input is augmented with K cosine/sine harmonics (Fourier features), then
    passed through a small tanh MLP. Output is smooth in t, so its first/second
    time derivatives are well-defined via autodiff.
    """

    def __init__(self, K: int, hidden: list[int], p: int):
        super().__init__()
        self.K = K
        self.mlp = _mlp([1 + 2 * K] + hidden + [p], act=nn.Tanh)

    def forward(self, tn: torch.Tensor) -> torch.Tensor:
        feats = [tn]
        for k in range(1, self.K + 1):
            w = 2.0 * np.pi * k
            feats.append(torch.cos(w * tn))
            feats.append(torch.sin(w * tn))
        return self.mlp(torch.cat(feats, dim=-1))


class DeepONet(nn.Module):
    """DeepONet with a branch (scenario -> n x p) and a Fourier trunk (time -> p)."""

    def __init__(
        self,
        s_dim: int,
        n_machines: int,
        p: int = 128,
        K: int = 16,
        branch_hidden: list[int] | None = None,
        trunk_hidden: list[int] | None = None,
        t_scale: float = 1.0,
    ):
        super().__init__()
        branch_hidden = branch_hidden or [512, 512, 512]
        trunk_hidden = trunk_hidden or [128, 128, 128]
        self.n_machines = n_machines
        self.p = p
        self.t_scale = t_scale

        self.branch = _mlp([s_dim] + branch_hidden + [n_machines * p], act=nn.Tanh)
        self.trunk = FourierTrunk(K, trunk_hidden, p)

    def branch_coeff(self, s: torch.Tensor) -> torch.Tensor:
        """Scenario s (B, s_dim) -> basis coefficients B (B, n, p)."""
        return self.branch(s).view(-1, self.n_machines, self.p)

    def trunk_basis(self, t: torch.Tensor) -> torch.Tensor:
        """Time t (T, 1) -> basis tau (T, p)."""
        return self.trunk(t / self.t_scale)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = self.branch_coeff(s)          # (B, n, p)
        tau = self.trunk_basis(t)         # (T, p)
        return torch.einsum("bnp,tp->bnt", B, tau)


class AttentionBranch(nn.Module):
    """Branch with self-attention over the n machines before the MLP.

    The severity vector is treated as n tokens (one per machine); multi-head
    self-attention captures which machines are jointly affected by a fault,
    then an MLP maps the context-aware features to basis coefficients.
    """

    def __init__(self, n, p, d_model=64, n_heads=4, hidden=(256, 256, 256)):
        super().__init__()
        self.n = n
        self.proj = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = _mlp([n * d_model + 1] + list(hidden) + [n * p], act=nn.Tanh)

    def forward(self, s):
        sev = s[:, : self.n].unsqueeze(-1)          # (B, n, 1)
        x = self.proj(sev)                          # (B, n, d_model)
        x, _ = self.attn(x, x, x)                   # self-attention over machines
        x = x.flatten(1)                            # (B, n*d_model)
        x = torch.cat([x, s[:, self.n:]], dim=1)    # append t_clear
        return self.mlp(x)


class AttentionDeepONet(DeepONet):
    """DeepONet with a self-attention branch (same Fourier trunk)."""

    def __init__(self, s_dim, n_machines, p=128, K=16, t_scale=1.0,
                 d_model=64, n_heads=4):
        super().__init__(s_dim, n_machines, p=p, K=K, t_scale=t_scale)
        self.branch = AttentionBranch(n_machines, p, d_model=d_model, n_heads=n_heads)


def trunk_derivs(trunk: FourierTrunk, t: torch.Tensor, t_scale: float):
    """Trunk basis and its first/second time derivatives.

    Uses forward-mode autodiff (torch.func.jvp), which is exact and much faster
    than a loop of reverse-mode autograd calls for a 1-D input.
    """
    from torch.func import jvp

    t_ = t / t_scale
    ones = torch.ones_like(t_)

    tau, dtau = jvp(lambda x: trunk(x), (t_,), (ones,))          # dtau = d(tau)/dt_norm

    def _first_deriv(x):
        _, g = jvp(lambda z: trunk(z), (x,), (torch.ones_like(x),))
        return g

    _, d2tau = jvp(_first_deriv, (t_,), (ones,))                 # d2tau = d2(tau)/dt_norm2

    dtau = dtau / t_scale
    d2tau = d2tau / (t_scale * t_scale)
    return tau, dtau, d2tau


def physics_residual(
    delta: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    E: torch.Tensor,
    Pm: torch.Tensor,
    M: torch.Tensor,
    G: torch.Tensor,
    Bsus: torch.Tensor,
    Mtot: float,
    ws: float = 2.0 * np.pi * 60.0,
) -> torch.Tensor:
    """Post-fault swing-equation residual in the COI frame (undamped).

    The COI-frame swing equation reads M_i d2delta_i = ws*(Pmi - Pei) - ws*M_i*c
    with c = sum_j(Pmj-Pej)/Mtot, so the residual is
    M_i*d2 - ws*(Pmi-Pei) + ws*M_i*c (all terms in rad/s).
    """
    d_ij = delta[:, :, None, :] - delta[:, None, :, :]            # (B, n, n, T)
    cos = torch.cos(d_ij)
    sin = torch.sin(d_ij)
    Eij = (E[:, None] * E[None, :])                               # (n, n)
    Pe = (Eij[None, :, :, None] * (G[None, :, :, None] * cos + Bsus[None, :, :, None] * sin)).sum(dim=2)
    sum_Pm = Pm.sum()
    sum_Pe = Pe.sum(dim=1)                                        # (B, T)
    cm = (sum_Pm - sum_Pe) / Mtot
    # COI-frame swing residual: M_i d2delta_i - ws*(Pmi - Pei) + ws*M_i*cm = 0
    resid = M[None, :, None] * d2 - ws * (Pm[None, :, None] - Pe) + ws * M[None, :, None] * cm[:, None, :]
    return resid
