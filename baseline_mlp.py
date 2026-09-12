"""Vanilla MLP baseline (pointwise map scenario x time -> trajectory).

Contrasts with the DeepONet operator: here the network is a plain MLP with no
operator / basis structure. Same data, same loss, same evaluation.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from evaluate import severity_enc, cross_conformal, predict
from model import _mlp

CLIP = 3.0 * np.pi


class PointMLP(nn.Module):
    def __init__(self, s_dim, n, hidden=(256, 256, 256, 256), t_scale=1.0):
        super().__init__()
        self.n = n
        self.t_scale = t_scale
        self.mlp = _mlp([s_dim + 1] + list(hidden) + [n], act=nn.Tanh)

    def forward(self, s, t):
        B = s.shape[0]; T = t.shape[0]
        s_b = s[:, None, :].expand(B, T, -1)
        t_b = (t[None, :, :] / self.t_scale).expand(B, T, -1)
        x = torch.cat([s_b, t_b], dim=-1)
        return self.mlp(x).permute(0, 2, 1)          # (B, n, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load("data39.npz")
    n = int(d["n_machines"]); t_end = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]

    s_train = severity_enc(d["fb_train"], d["tc_train"], d, dev)
    s_val = severity_enc(d["fb_val"], d["tc_val"], d, dev)
    y_train = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_val = torch.as_tensor(d["y_val"], dtype=torch.float32, device=dev)
    st_train = torch.as_tensor(d["s_train"].astype(bool), device=dev)

    model = PointMLP(s_train.shape[1], n, t_scale=t_end).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    B_tr = y_train.shape[0]
    n_batches = max(1, B_tr // args.batch_size)
    best = float("inf"); t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(B_tr, device=dev)
        for it in range(n_batches):
            idx = perm[it * args.batch_size:(it + 1) * args.batch_size]
            s = s_train[idx]; y = y_train[idx]; st = st_train[idx]
            pred = model(s, t)
            target = torch.clamp(y, -CLIP, CLIP)
            se = ((pred - target) ** 2).mean(dim=(1, 2))
            n_st = st.sum().clamp(min=1); n_un = (~st).sum().clamp(min=1)
            loss = 0.5 * (se[st].sum() / n_st + se[~st].sum() / n_un)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        with torch.no_grad():
            mv = ((model(s_val, t) - torch.clamp(y_val, -CLIP, CLIP)) ** 2).mean().item()
        if mv < best:
            best = mv
            torch.save(model.state_dict(), "model_mlp.pt")
        if ep % 100 == 0:
            print(f"ep {ep}  val_mse {mv:.4e}  ({time.time()-t0:.0f}s)", flush=True)

    # evaluate (reuse the DeepONet eval helpers)
    model.load_state_dict(torch.load("model_mlp.pt"))
    model.eval()
    fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]; st = d["s_test"].astype(bool)
    s = severity_enc(fb, tc, d, dev)
    pred = predict(model, s, t, dev)
    err = np.abs(pred - y)
    rmse_stable = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
    final_spread = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
    acc = ((final_spread < np.pi) == st).mean()
    scores = err[st].max(axis=(1, 2))
    _, cov = cross_conformal(scores, 0.1)
    print(f"\nMLP baseline: stable_rmse={rmse_stable:.2f} deg | clf_acc={acc:.3f} | "
          f"conformal_coverage={cov:.3f}")


if __name__ == "__main__":
    main()
