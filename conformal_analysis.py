"""Split conformal calibration of the operator (order-statistic quantile).

Calibration and evaluation use DISJOINT halves of the held-out test set's stable
scenarios (random 50/50 split, seed 0). Produces: (1) coverage vs. miscoverage
level alpha at the four reported levels; (2) Mondrian (grouped) split-conformal
intervals by post-fault peak of the pairwise spread -- an OFFLINE diagnostic,
since the grouping uses the ground-truth trajectory.

Peak definition (fixed per audit A5): max over the post-fault window (t >= t_c)
of the pairwise spread max_i delta_i - min_i delta_i of the true COI angles.
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
ALPHA = 0.1


def order_quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t_out = d["t_out"]
    t = torch.as_tensor(t_out, dtype=torch.float32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()

    fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]; st = d["s_test"].astype(bool)
    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
    err = np.abs(pred - y)
    scores = err[st].max(axis=(1, 2))          # worst-case error per stable scenario
    # post-fault peak of the pairwise spread (true trajectory; offline diagnostic)
    y_s = y[st]; tc_s = tc[st]
    peaks = []
    for k in range(y_s.shape[0]):
        w = t_out >= tc_s[k]
        peaks.append(np.rad2deg((y_s[k, :, w].max(axis=0) - y_s[k, :, w].min(axis=0)).max()))
    peaks = np.array(peaks)

    # ---- disjoint calibration / evaluation split ----
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(scores))
    n_cal = len(scores) // 2
    cal, eva = idx[:n_cal], idx[n_cal:]
    scores_cal, scores_eva = scores[cal], scores[eva]
    peaks_cal, peaks_eva = peaks[cal], peaks[eva]
    print(f"split conformal: n_cal={len(cal)}, n_eva={len(eva)}")

    # ---- 1. coverage vs alpha at the four reported levels ----
    def split_q_cov(alpha):
        q = order_quantile(scores_cal, alpha)
        cov = (scores_eva <= q).mean()
        return q, cov

    levels = [0.05, 0.10, 0.30, 0.50]
    covs, qs = [], []
    for a in levels:
        q, c = split_q_cov(a); covs.append(c); qs.append(q)
        k = int((scores_eva <= q).sum())
        print(f"alpha={a:.2f}: coverage={c:.3f} ({k}/{len(scores_eva)}) "
              f"(target {1-a:.2f}), q={np.rad2deg(q):.1f} deg")

    # ---- 2. Mondrian split conformal by post-fault peak (3 groups) ----
    bins = [(0, 45), (45, 90), (90, 180)]
    bin_q, bin_cov, bin_annot = [], [], []
    for lo, hi in bins:
        mc = (peaks_cal >= lo) & (peaks_cal < hi)
        me = (peaks_eva >= lo) & (peaks_eva < hi)
        if mc.sum() < 2 or me.sum() == 0:
            bin_q.append(np.nan); bin_cov.append(np.nan)
            bin_annot.append(f"n={mc.sum()}/{me.sum()}")
            continue
        q = order_quantile(scores_cal[mc], ALPHA)
        cov = (scores_eva[me] <= q).mean()
        bin_q.append(np.rad2deg(q)); bin_cov.append(cov)
        bin_annot.append(f"n={mc.sum()}/{me.sum()}\ncov={cov:.2f}")
        print(f"peak {lo:3d}-{hi:3d} deg: n_cal={mc.sum():3d}, n_eva={me.sum():3d}, "
              f"q={np.rad2deg(q):6.1f} deg, cov={cov:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].errorbar([1 - a for a in levels], covs,
                     yerr=[1.96 * np.sqrt(c * (1 - c) / len(scores_eva)) for c in covs],
                     fmt="o-", color="#1f77b4", capsize=3)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("target coverage $1-\\alpha$"); axes[0].set_ylabel("empirical coverage")
    axes[0].set_title("coverage validity (split conformal)")
    bars = axes[1].bar(np.arange(len(bins)), bin_q, 0.55, color="#4C72B0")
    for x0, b, a in zip(np.arange(len(bins)), bars, bin_annot):
        axes[1].text(x0, b.get_height() + 4, a, ha="center", fontsize=6.5)
    axes[1].set_xticks(np.arange(len(bins)))
    axes[1].set_xticklabels(["<45", "45-90", ">90"])
    axes[1].set_xlabel("post-fault peak of pairwise spread (deg)")
    axes[1].set_ylabel("interval half-width (deg)")
    axes[1].set_title("band widens toward the boundary (offline)")
    fig.tight_layout(); fig.savefig("fig_conformal.pdf"); plt.close(fig)
    print("saved fig_conformal.pdf")


if __name__ == "__main__":
    main()
