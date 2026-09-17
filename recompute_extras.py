# -*- coding: utf-8 -*-
"""Remaining paper numbers under the corrected protocol (max-over-time criterion,
order-statistic quantile, regenerated reference): noise robustness, boundary
stratification, Bonferroni/Sidak, alternative scores."""
import numpy as np
import torch

from model import DeepONet
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)
ALPHA = 0.1


def order_quantile(scores, alpha=ALPHA):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()
    fb, tc, y, st = d["fb_test"], d["tc_test"], d["y_test"], d["s_test"].astype(bool)
    rng = np.random.default_rng(0)

    def spread_max(p):
        return (p.max(axis=1) - p.min(axis=1)).max(axis=1)

    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
    err = np.abs(pred - y)
    sm = spread_max(pred)
    acc0 = ((sm < np.pi) == st).mean()
    print(f"clean: acc={acc0:.4f}", flush=True)

    # ---- 1. noise robustness (new criterion) ----
    sev0 = severity_enc(fb, tc, d, dev).cpu().numpy()
    for pct in (0.05, 0.10, 0.20):
        rng_n = np.random.default_rng(7)
        scale = np.abs(sev0).max(axis=1, keepdims=True)
        noisy = sev0 + rng_n.normal(0, pct, size=sev0.shape) * scale
        s_n = torch.as_tensor(noisy, **F32, device=dev)
        with torch.no_grad():
            p_n = m(s_n, t).cpu().numpy()
        acc_n = ((spread_max(p_n) < np.pi) == st).mean()
        rmse_n = float(np.sqrt(((p_n - y)[st] ** 2).mean()))
        print(f"noise {int(pct*100)}%: acc={acc_n:.4f}  pooled RMSE={np.rad2deg(rmse_n):.1f} deg",
              flush=True)

    # ---- 2. boundary stratification by distance to TDS CCT (new labels) ----
    from power_system import ClassicalModel
    from dataset import simulate_batch
    model39 = ClassicalModel(case="39")
    def true_cct(bus):
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            r = simulate_batch(model39, np.array([bus]), np.array([mid]), np.array([-1]), 4.0,
                               dt=0.0005)
            lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
        return lo
    cct_map = {int(b): true_cct(int(b)) for b in np.unique(fb)}
    dist = np.array([cct_map[int(b)] - c for b, c in zip(fb, tc)]) * 1000   # ms inside boundary
    for thr in (100, 30, 10):
        sel = st & (dist > thr)
        rmse_s = float(np.sqrt((err[sel] ** 2).mean()))
        mis = 1 - ((sm[sel] < np.pi) == st[sel]).mean()
        sel_near = (dist > 0) & (dist <= thr)
        mis_near = 1 - ((sm[sel_near] < np.pi) == st[sel_near]).mean() if sel_near.sum() else np.nan
        print(f"dist>{thr} ms: stable RMSE={np.rad2deg(rmse_s):.1f} deg, misclass={mis:.3f}; "
              f"0<dist<={thr} ms: misclass={mis_near:.3f} (n={int(sel_near.sum())})", flush=True)

    # ---- 3. Bonferroni / Sidak per-machine (new) ----
    Rm = err[st].max(axis=2)
    idx = rng.permutation(len(Rm)); nc = len(Rm) // 2
    q_b = np.array([order_quantile(Rm[idx[:nc], j], ALPHA / n) for j in range(n)])
    cov_b = (Rm[idx[nc:]] <= q_b[None, :]).all(axis=1).mean()
    pw_b = (q_b[:, None] + q_b[None, :]).max()
    a_s = 1 - (1 - ALPHA) ** (1 / n)
    q_s = np.array([order_quantile(Rm[idx[:nc], j], a_s) for j in range(n)])
    pw_s = (q_s[:, None] + q_s[None, :]).max()
    print(f"Bonferroni: pairwise={np.rad2deg(pw_b):.1f} deg, joint cov={cov_b:.3f}; "
          f"Sidak pairwise={np.rad2deg(pw_s):.1f} deg", flush=True)

    # ---- 4. alternative scores (order-stat) ----
    y_s, err_s = y[st], err[st]
    amp = np.rad2deg(np.abs(y_s).max(axis=(1, 2)))
    rel = err_s.max(axis=(1, 2)) / amp
    Rg = err_s.max(axis=(1, 2))
    idx2 = rng.permutation(len(Rg)); nc2 = len(Rg) // 2

    def qc(scores):
        q_ = order_quantile(scores[idx2[:nc2]])
        return q_, (scores[idx2[nc2:]] <= q_).mean()

    q_r, c_r = qc(rel)
    print(f"relative score: band={q_r*100:.1f}% of amplitude, cov={c_r:.3f}", flush=True)

    fs = []
    for k in range(y_s.shape[0]):
        ends = []
        for j in range(n):
            w = np.where(d["t_out"] > tc[st][k])[0]
            jj = np.where(np.diff(np.sign(np.diff(y_s[k, j, w]))) < 0)[0]
            ends.append(w[0] + jj[0] + 1 if len(jj) else len(d["t_out"]) - 1)
        fs.append(err_s[k, :, :max(ends) + 1].max())
    fs = np.array(fs)
    q_f, c_f = qc(fs)
    print(f"first-swing score: q={np.rad2deg(q_f):.1f} deg, cov={c_f:.3f}", flush=True)

    ta = err_s.mean(axis=2).max(axis=1)
    q_t, c_t = qc(ta)
    print(f"time-averaged score: q={np.rad2deg(q_t):.1f} deg, cov={c_t:.3f}", flush=True)


if __name__ == "__main__":
    main()
