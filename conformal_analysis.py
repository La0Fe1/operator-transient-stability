"""Split conformal certification of the operator.

Calibration and evaluation use DISJOINT halves of the held-out test set's stable
scenarios (random 50/50 split, seed 0), so the calibration set is independent of
the set on which coverage is reported. Produces: (1) coverage vs. miscoverage
level alpha; (2) Mondrian (grouped) split-conformal intervals by distance-to-
boundary, showing the certified interval widens near the stability boundary.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import DeepONet
from evaluate import severity_enc

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

SEED = 0


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()

    fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]; st = d["s_test"].astype(bool)
    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
    err = np.abs(pred - y)
    scores = err[st].max(axis=(1, 2))          # worst-case error per stable scenario
    peaks = np.rad2deg(y[st].max(axis=(1, 2)))  # post-fault peak angle per stable scenario

    # ---- disjoint calibration / evaluation split ----
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(scores))
    n_cal = len(scores) // 2
    cal, eva = idx[:n_cal], idx[n_cal:]
    scores_cal, scores_eva = scores[cal], scores[eva]
    peaks_cal, peaks_eva = peaks[cal], peaks[eva]
    print(f"split conformal: n_cal={len(cal)}, n_eva={len(eva)}")

    # ---- 1. coverage vs alpha (split conformal) ----
    def split_q_cov(alpha):
        q = np.quantile(scores_cal, min(1.0, (1 - alpha) * (1 + 1.0 / n_cal)))
        cov = (scores_eva <= q).mean()
        return q, cov

    alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    covs, qs = [], []
    for a in alphas:
        q, c = split_q_cov(a); covs.append(c); qs.append(q)
        print(f"alpha={a:.2f}: coverage={c:.3f} (target {1-a:.2f}), q={np.rad2deg(q):.1f} deg")

    # ---- 2. Mondrian split conformal by peak angle ----
    bins = [(0, 45), (45, 90), (90, 135), (135, 180)]
    bin_q, bin_cov = [], []
    for lo, hi in bins:
        mc = (peaks_cal >= lo) & (peaks_cal < hi)
        me = (peaks_eva >= lo) & (peaks_eva < hi)
        if mc.sum() < 2 or me.sum() == 0:
            bin_q.append(np.nan); bin_cov.append(np.nan); continue
        lev = min(1.0, 0.9 * (1 + 1.0 / mc.sum()))
        q = np.quantile(scores_cal[mc], lev)
        cov = (scores_eva[me] <= q).mean()
        bin_q.append(np.rad2deg(q)); bin_cov.append(cov)
        print(f"peak {lo:3d}-{hi:3d} deg: n_cal={mc.sum():3d}, n_eva={me.sum():3d}, "
              f"q={np.rad2deg(q):6.1f} deg, cov={cov:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].plot([1 - a for a in alphas], covs, "o-", color="#1f77b4")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("target coverage $1-\\alpha$"); axes[0].set_ylabel("empirical coverage")
    axes[0].set_title("coverage validity (split conformal)")
    axes[1].bar(np.arange(len(bins)), bin_q, 0.6, color="#4C72B0")
    axes[1].set_xticks(np.arange(len(bins))); axes[1].set_xticklabels(["0-45", "45-90", "90-135", "135-180"])
    axes[1].set_xlabel("post-fault peak (deg)"); axes[1].set_ylabel("interval half-width (deg)")
    axes[1].set_title("certified interval widens near boundary")
    fig.tight_layout(); fig.savefig("fig_conformal.pdf"); plt.close(fig)
    print("saved fig_conformal.pdf")


if __name__ == "__main__":
    main()
