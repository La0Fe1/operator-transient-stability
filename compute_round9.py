"""Round-9 computations.

(1) Alternative nonconformity scores on existing predictions: first-swing-window
    score and time-averaged score, both with split calibration.
(2) Stratified error analysis vs distance to the TDS CCT.
"""
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()

    fb, tc, y, st = d["fb_test"], d["tc_test"], d["y_test"], d["s_test"].astype(bool)
    t_out = d["t_out"]
    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
    err = np.abs(pred - y)

    def split_q_cov(scores, alpha=0.1):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(scores)); nc = len(scores) // 2
        q = np.quantile(scores[idx[:nc]], min(1.0, (1 - alpha) * (1 + 1.0 / nc)))
        return q, (scores[idx[nc:]] <= q).mean()

    # ---- (1a) first-swing-window score ----
    R_fs = []
    for i in range(len(fb)):
        if not st[i]:
            continue
        mask = t_out >= tc[i]
        tt = t_out[mask]
        ei = err[i][:, mask]
        yi = np.abs(y[i][:, mask])
        t_peak = tt[0]
        for j in range(n):
            for k in range(1, len(tt) - 1):
                if yi[j, k - 1] <= yi[j, k] and yi[j, k] >= yi[j, k + 1]:
                    t_peak = max(t_peak, tt[k])
                    break
        w1 = tt <= t_peak
        R_fs.append(ei[:, w1].max() if w1.any() else ei.max())
    R_fs = np.array(R_fs)
    q_fs, cov_fs = split_q_cov(R_fs)
    print(f"(1a) first-swing score: q={np.rad2deg(q_fs):.1f} deg, coverage={cov_fs:.3f}", flush=True)

    # ---- (1b) time-averaged score (mean over time, max over machines) ----
    R_avg = err[st].mean(axis=2).max(axis=1)
    q_avg, cov_avg = split_q_cov(R_avg)
    print(f"(1b) time-averaged score: q={np.rad2deg(q_avg):.1f} deg, coverage={cov_avg:.3f}", flush=True)

    # ---- (2) stratified error vs distance to CCT ----
    model39 = ClassicalModel(case="39")
    cct_cache = {}

    def true_cct(bus):
        if bus not in cct_cache:
            lo, hi = 0.01, 0.8
            for _ in range(16):
                mid = (lo + hi) / 2
                r = simulate_batch(model39, np.array([bus]), np.array([mid]), np.array([-1]), 4.0)
                lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
            cct_cache[bus] = lo
        return cct_cache[bus]

    dist = tc - np.array([true_cct(int(b)) for b in fb])      # negative = inside the boundary
    pred_stable = (pred[:, :, -1].max(1) - pred[:, :, -1].min(1)) < np.pi
    print("(2) stratified by distance to CCT (negative = stable side):", flush=True)
    bins = [(-1.0, -0.1), (-0.1, -0.03), (-0.03, -0.01), (-0.01, 0.0), (0.0, 0.05)]
    for lo, hi in bins:
        sel = (dist >= lo) & (dist < hi)
        if sel.sum() == 0:
            continue
        mis = (pred_stable[sel] != st[sel]).mean()
        rmse_st = np.rad2deg(np.sqrt((err[sel][st[sel]] ** 2).mean())) if st[sel].any() else np.nan
        print(f"  dist [{lo:+.2f},{hi:+.2f}] s: n={sel.sum()}, "
              f"misclassification={mis:.3f}, stable pooled RMSE={rmse_st:.1f} deg", flush=True)


if __name__ == "__main__":
    main()
