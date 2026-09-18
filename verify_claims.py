# -*- coding: utf-8 -*-
"""Verify: (1) exact first-crossing CCT mean on 39-bus (paper says 25.0, checker says 25.22);
(2) damped-model split-conformal coverage (paper says 0.90+-0.02 over 5 splits, checker says 0.801)."""
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch
from evaluate import severity_enc, predict

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def order_quantile(scores, alpha=0.1):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

# ---------------- (1) exact first-crossing CCT, 39-bus ----------------
d = np.load("data39.npz")
n = int(d["n_machines"]); t_end = float(d["t_end"]); t_out = d["t_out"]
t = torch.as_tensor(t_out, dtype=torch.float32, device=dev)[:, None]
m = DeepONet(n + 1, n, p=512, t_scale=t_end).to(dev)
m.load_state_dict(torch.load("model_p512.pt")); m.eval()
sys_model = ClassicalModel(case="39")

def tds_stable(bus, tc):
    r = simulate_batch(sys_model, np.array([bus]), np.array([tc]), np.array([-1]), 4.0, dt=0.0005)
    return bool(r["stable"][0])

def op_stable(bus, tc):
    with torch.no_grad():
        tr = predict(m, severity_enc(np.array([bus]), np.array([tc]), d, dev), t, dev)[0]
    return bool((tr.max(axis=0) - tr.min(axis=0)).max() < np.pi)

def first_crossing(fn, lo=0.01, hi=0.8, sweep=0.005, tol=0.0001):
    grid = np.arange(lo, hi + 1e-12, sweep)
    labels = [fn(grid[0])]
    for tc in grid[1:]:
        labels.append(fn(tc))
        if labels[-2] and not labels[-1]:
            a, b = tc - sweep, tc
            for _ in range(int(np.ceil(np.log2(sweep / tol)))):
                mid = (a + b) / 2
                if fn(mid):
                    a = mid
                else:
                    b = mid
            return b
    return None

buses = sorted(np.unique(d["fb_test"]))
refs, ops = [], []
for b in buses:
    refs.append(first_crossing(lambda tc: tds_stable(b, tc)) * 1000)
    ops.append(first_crossing(lambda tc: op_stable(b, tc)) * 1000)
errs = np.array(ops) - np.array(refs)
print("=== CCT first-crossing (exact) ===")
for b, r, o, e in zip(buses, refs, ops, errs):
    print(f"bus {int(b):2d}: ref={r:8.3f}  op={o:8.3f}  signed={e:+8.3f}")
print(f"mean |err| (exact)      = {np.abs(errs).mean():.4f} ms")
print(f"mean |err| (rounded 0.1)= {np.round(np.abs(errs), 1).mean():.4f} ms")
print(f"mean signed             = {errs.mean():.4f} ms")
print(f"max |err|               = {np.abs(errs).max():.4f} ms")

# ---------------- (2) damped coverage, 1 split and 5 splits ----------------
d2 = np.load("data39_damped.npz")
n2 = int(d2["n_machines"]); te2 = float(d2["t_end"])
t2 = torch.as_tensor(d2["t_out"], dtype=torch.float32, device=dev)[:, None]
m2 = DeepONet(n2 + 1, n2, p=512, t_scale=te2).to(dev)
m2.load_state_dict(torch.load("model_damped_p512.pt")); m2.eval()
fb2, tc2, y2, st2 = d2["fb_test"], d2["tc_test"], d2["y_test"], d2["s_test"].astype(bool)
with torch.no_grad():
    pred2 = m2(severity_enc(fb2, tc2, d2, dev), t2).cpu().numpy()
err2 = np.abs(pred2 - y2)
scores2 = err2[st2].max(axis=(1, 2))
print(f"\n=== damped coverage ===  n_stable={int(st2.sum())}, n_scores={len(scores2)}")
for seed in range(5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(scores2)); nc = len(scores2) // 2
    q = order_quantile(scores2[idx[:nc]])
    cov = (scores2[idx[nc:]] <= q).mean()
    print(f"seed {seed}: n_cal={nc}, n_eva={len(scores2)-nc}, q={np.rad2deg(q):.1f} deg, cov={cov:.3f}")
c5 = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(scores2)); nc = len(scores2) // 2
    q = order_quantile(scores2[idx[:nc]])
    c5.append((scores2[idx[nc:]] <= q).mean())
print(f"mean={np.mean(c5):.3f}  std={np.std(c5):.3f}  (paper: 0.90+-0.02)")
