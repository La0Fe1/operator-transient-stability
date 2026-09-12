"""Full evaluation: trajectory accuracy, stability classification, conformal
prediction (cross-conformal / jackknife+ certified coverage), critical clearing
time (CCT) estimation, and the speedup versus time-domain simulation.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch


def load_model(path, s_dim, n, p=128):
    d = np.load("data39.npz")
    te = float(d["t_end"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DeepONet(s_dim, n, p=p, t_scale=te).to(dev)
    m.load_state_dict(torch.load(path))
    m.eval()
    return m, dev, d


def severity_enc(fb, tc, d, dev):
    Pm = d["Pm"]; M = d["M"]; pe = d["pe_fault"]; t_max = float(d["t_max"])
    sev = (Pm[None, :] - pe[fb - 1]) / M[None, :] * 10.0
    tcn = np.asarray(tc, dtype=np.float32)[:, None] / t_max
    return torch.cat([torch.as_tensor(sev, dtype=torch.float32, device=dev),
                      torch.as_tensor(tcn, device=dev)], dim=1)


def predict(model, s, t, dev):
    with torch.no_grad():
        return model(s, t).cpu().numpy()          # (B, n, T)


def cross_conformal(scores: np.ndarray, alpha: float):
    """Leave-one-out (jackknife+) conformal quantiles and coverage.

    Returns (q_loo, coverage): q_loo is the per-point calibrated half-width and
    coverage is the fraction of points whose score is within its own interval.
    """
    N = len(scores)
    q_loo = np.empty(N)
    for i in range(N):
        others = np.delete(scores, i)
        q_loo[i] = np.quantile(others, min(1.0, (1 - alpha) * (1 + 1.0 / (N - 1))))
    coverage = (scores <= q_loo).mean()
    return q_loo, coverage


def main(path="model_dataonly.pt", alpha=0.1, p=128):
    d = np.load("data39.npz")
    n = int(d["n_machines"])
    t_out = d["t_out"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.as_tensor(t_out, dtype=torch.float32, device=dev)[:, None]
    s_dim = n + 1
    model, dev, _ = load_model(path, s_dim, n, p=p)

    # ---- predictions on the full held-out test set (12 fault buses) ----
    fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]; st = d["s_test"].astype(bool)
    s = severity_enc(fb, tc, d, dev)
    pred = predict(model, s, t, dev)
    err = np.abs(pred - y)

    # ---- 1. trajectory accuracy (stable scenarios) ----
    rmse_stable = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
    # fault-on vs post-fault pooled RMSE (Table 1: pooled within each region)
    masks = t_out[None, :] < tc[st, None]          # (n_stable, T)
    fault_on_pooled = np.rad2deg(np.sqrt((err[st][masks] ** 2).mean()))
    post_fault_pooled = np.rad2deg(np.sqrt((err[st][~masks] ** 2).mean()))
    # per-scenario mean error (alternative view)
    fault_on_err, post_fault_err = [], []
    for i in range(len(fb)):
        if not st[i]:
            continue
        mask = t_out < tc[i]
        fault_on_err.append(np.rad2deg(err[i][:, mask].mean()))
        post_fault_err.append(np.rad2deg(err[i][:, ~mask].mean()))

    # ---- 2. stability classification ----
    final_spread = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
    pred_stable = final_spread < np.pi
    acc = (pred_stable == st).mean()
    tp = (pred_stable & st).sum(); fp = (pred_stable & ~st).sum()
    fn = (~pred_stable & st).sum(); tn = (~pred_stable & ~st).sum()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)

    # ---- 3. split conformal certified trajectory interval (stable only) ----
    scores = err[st].max(axis=(1, 2))             # worst-case error per stable scenario
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(scores))
    n_cal = len(scores) // 2
    q = np.quantile(scores[idx[:n_cal]], min(1.0, (1 - alpha) * (1 + 1.0 / n_cal)))
    coverage = (scores[idx[n_cal:]] <= q).mean()
    q_loo = np.full_like(scores, q)

    # ---- 4. CCT estimation (binary search with the operator) ----
    model_cct = ClassicalModel(case="39")

    def true_cct(bus):
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            r = simulate_batch(model_cct, np.array([bus]), np.array([mid]), np.array([-1]), 4.0)
            lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
        return lo

    def pred_cct(bus):
        def stable(tc):
            tr = predict(model, severity_enc(np.array([bus]), np.array([tc]), d, dev), t, dev)[0]
            return (tr[:, -1].max() - tr[:, -1].min()) < np.pi
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if stable(mid) else (lo, mid)
        return lo

    test_buses = np.unique(fb)
    cct_errs = np.array([abs(pred_cct(b) - true_cct(b)) for b in test_buses]) * 1000.0

    # ---- 5. speedup vs time-domain simulation ----
    B = 64
    s_r = severity_enc(fb[:B], tc[:B], d, dev)
    t0 = time.time()
    for _ in range(20):
        predict(model, s_r, t, dev)
    t_op = (time.time() - t0) / 20 / B
    t0 = time.time()
    simulate_batch(model_cct, fb[:B], tc[:B], np.full(B, -1), 4.0, dt=0.0005)
    t_rk = (time.time() - t0) / B
    speedup = t_rk / t_op

    print("=" * 70)
    print(f"MODEL {path}  (test: {len(test_buses)} held-out fault buses)")
    print("=" * 70)
    print(f"stable trajectory RMSE   : {rmse_stable:.2f} deg   (n_stable={st.sum()})")
    print(f"  - fault-on region (pooled RMSE) : {fault_on_pooled:.2f} deg")
    print(f"  - post-fault region (pooled RMSE): {post_fault_pooled:.2f} deg")
    print(f"  - fault-on region (mean per-scen): {np.mean(fault_on_err):.2f} deg")
    print(f"  - post-fault region (mean per-scen): {np.mean(post_fault_err):.2f} deg")
    print(f"classification acc       : {acc:.3f}  (P={prec:.3f} R={rec:.3f} F1={2*tp/(2*tp+fp+fn):.3f})")
    print(f"conformal (alpha={alpha}) : q={np.rad2deg(q_loo.mean()):.2f} deg, "
          f"coverage={coverage:.3f} (target {1-alpha:.2f}, split cal/eval)")
    print(f"CCT error                : mean={cct_errs.mean():.1f} ms, max={cct_errs.max():.1f} ms")
    print(f"speedup vs TDS           : {speedup:.0f}x")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_dataonly.pt")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--p", type=int, default=128)
    args = ap.parse_args()
    main(args.model, args.alpha, args.p)
