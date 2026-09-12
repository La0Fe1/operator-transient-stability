"""Sequence-to-sequence (GRU) trajectory baseline for TSA.

Contrasts with the DeepONet operator: a recurrent decoder maps the same
physics-informed severity encoding + Fourier time features to the full COI
rotor-angle trajectory, but through a sequential GRU rather than an
operator/basis decomposition. Same data, same balanced loss, same evaluation.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from evaluate import severity_enc, predict

CLIP = 3.0 * np.pi


class GRUTrajectory(nn.Module):
    """Scenario encoding -> initial GRU state; GRU decodes the trajectory over time.

    The decoder input at each step is the Fourier-feature embedding of time (the
    same K=16 harmonics used by the DeepONet trunk), so the comparison isolates
    the recurrent vs. operator structure.
    """

    def __init__(self, s_dim, n, hidden=256, K=16, t_scale=1.0):
        super().__init__()
        self.n = n
        self.K = K
        self.t_scale = t_scale
        self.s_proj = nn.Linear(s_dim, hidden)
        self.gru = nn.GRU(1 + 2 * K, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n)

    def time_feats(self, t):
        tn = t / self.t_scale
        feats = [tn]
        for k in range(1, self.K + 1):
            w = 2.0 * np.pi * k
            feats.append(torch.cos(w * tn))
            feats.append(torch.sin(w * tn))
        return torch.cat(feats, dim=-1)          # (T, 1+2K)

    def forward(self, s, t):
        B = s.shape[0]; T = t.shape[0]
        h0 = self.s_proj(s).unsqueeze(0)         # (1, B, hidden)
        x = self.time_feats(t).unsqueeze(0).expand(B, T, -1)   # (B, T, 1+2K)
        out, _ = self.gru(x, h0)                 # (B, T, hidden)
        return self.head(out).permute(0, 2, 1)   # (B, n, T)


def eval_pooled(model, s_te, y_te, st_te, t_out, dev):
    with torch.no_grad():
        pred = model(s_te, t_out).cpu().numpy()
    y = y_te.cpu().numpy(); st = st_te.astype(bool)
    err = np.abs(pred - y)
    pooled = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
    fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
    acc = ((fs < np.pi) == st).mean()
    return pooled, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load("data39.npz")
    n = int(d["n_machines"]); t_end = float(d["t_end"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]

    s_train = severity_enc(d["fb_train"], d["tc_train"], d, dev)
    s_test = severity_enc(d["fb_test"], d["tc_test"], d, dev)
    y_train = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_test = torch.as_tensor(d["y_test"], dtype=torch.float32, device=dev)
    st_train = torch.as_tensor(d["s_train"].astype(bool), device=dev)
    st_test = d["s_test"].astype(bool)

    model = GRUTrajectory(s_train.shape[1], n, hidden=args.hidden, t_scale=t_end).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    B_tr = y_train.shape[0]
    n_batches = max(1, B_tr // args.batch_size)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(B_tr, device=dev)
        for it in range(n_batches):
            idx = perm[it * args.batch_size:(it + 1) * args.batch_size]
            s = s_train[idx]; y = y_train[idx]; st = st_train[idx]
            pred = model(s, t_out)
            target = torch.clamp(y, -CLIP, CLIP)
            se = ((pred - target) ** 2).mean(dim=(1, 2))
            n_st = st.sum().clamp(min=1); n_un = (~st).sum().clamp(min=1)
            loss = 0.5 * (se[st].sum() / n_st + se[~st].sum() / n_un)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        if ep % 100 == 0:
            print(f"ep {ep}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), "model_rnn.pt")
    model.eval()
    pooled, acc = eval_pooled(model, s_test, y_test, st_test, t_out, dev)
    print(f"GRU baseline: pooled stable RMSE={pooled:.2f} deg | clf acc={acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
