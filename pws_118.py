# -*- coding: utf-8 -*-
"""Q5: pairwise-difference conformal score on the 118-bus system (real data)."""
import numpy as np
import torch

from model import DeepONet
from evaluate import severity_enc

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = np.load("data118.npz")
n = int(d["n_machines"]); te = float(d["t_end"]); tm = float(d["t_max"])
t = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
m.load_state_dict(torch.load("model_118.pt")); m.eval()

fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]; st = d["s_test"].astype(bool)
with torch.no_grad():
    pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
signed = pred - y
R_pw = np.abs(signed[st][:, None, :, :] - signed[st][:, :, None, :]).max(axis=(1, 2, 3))
spread_pred = (pred.max(axis=1) - pred.min(axis=1)).max(axis=1)
spread_true = (y.max(axis=1) - y.min(axis=1)).max(axis=1)

rng = np.random.default_rng(0)
idx = rng.permutation(len(R_pw)); nc = len(R_pw) // 2
print(f"n_stable = {int(st.sum())}, n_cal = {nc}, n_eva = {len(R_pw) - nc}")
print(f"n_stable / n_test = {st.mean():.3f}")

for alpha in [0.1, 0.3, 0.5]:
    k = int(np.ceil((nc + 1) * (1 - alpha)))
    q = np.sort(R_pw[idx[:nc]])[k - 1]
    cov = (R_pw[idx[nc:]] <= q).mean()
    # resolvable fraction on the evaluation half (predicted-stable scenarios)
    ps = spread_pred[st][idx[nc:]] < np.pi
    dec = (ps & (spread_pred[st][idx[nc:]] + np.rad2deg(q) < np.pi)).sum()
    n_ps = int(ps.sum())
    frac = dec / max(n_ps, 1)
    print(f"alpha={alpha}: q_pw={np.rad2deg(q):.1f} deg, coverage={cov:.3f}, "
          f"resolvable={frac:.3f} ({dec}/{n_ps} pred-stable)")
