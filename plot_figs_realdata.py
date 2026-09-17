# -*- coding: utf-8 -*-
"""Fig2 / Fig3 regenerated from REAL simulation data (data39.npz + model_p512.pt).

Replaces the hand-reproduced curves (figures/plot_fig2.py) and normal-approx
error bars (figures/plot_fig3.py) with the actual TDS reference trajectories,
model predictions and split-conformal outputs used in the manuscript.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import DeepONet
from evaluate import severity_enc, predict

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.linewidth": 0.7, "svg.fonttype": "none",
})
BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#999999"
OUT = r"C:\Users\佬肥\Desktop\算子\figures"

SEED = 0
ALPHA = 0.1

def order_quantile(scores, alpha):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

def wilson_ci(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

# ---------------- load model + data ----------------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = np.load("data39.npz")
n = int(d["n_machines"]); t_end = float(d["t_end"]); t_out = d["t_out"]
t = torch.as_tensor(t_out, dtype=torch.float32, device=dev)[:, None]
m = DeepONet(n + 1, n, p=512, t_scale=t_end).to(dev)
m.load_state_dict(torch.load("model_p512.pt")); m.eval()

fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]; st = d["s_test"].astype(bool)
with torch.no_grad():
    pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
err = np.abs(pred - y)
scores = err[st].max(axis=(1, 2))
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(scores)); nc = len(scores) // 2
cal, eva = idx[:nc], idx[nc:]

# first-swing score (offline diagnostic: window from TRUE trajectory)
fs_scores = []
for k in np.where(st)[0]:
    peaks = []
    for j in range(n):
        w = np.where(t_out > tc[k])[0]
        jj = np.where(np.diff(np.sign(np.diff(y[k, j, w]))) < 0)[0]
        peaks.append(w[0] + jj[0] + 1 if len(jj) else len(t_out) - 1)
    end = max(peaks)
    fs_scores.append(err[k, :, :end + 1].max())
fs_scores = np.array(fs_scores)
q_fs = order_quantile(fs_scores[cal], ALPHA)
q_fs_deg = float(np.rad2deg(q_fs))
print(f"[Fig2] first-swing band q_fs = {q_fs_deg:.1f} deg (paper: 70.8)")

# ================= Fig 2: real scenario (bus 11, tc ~ 0.1700) =================
i = int(np.where(fb == 11)[0][np.argmin(np.abs(tc[fb == 11] - 0.1700))])
assert st[i], "scenario not stable"
print(f"[Fig2] scenario idx={i}, fault bus={int(fb[i])}, t_c={tc[i]:.4f} s, stable={st[i]}")
s = severity_enc(fb[i:i+1], tc[i:i+1], d, dev)
p_i = predict(m, s, t, dev)[0]
y_i = y[i]
amp = np.abs(y_i).max(axis=1)
jj = np.argsort(amp)[-2:]          # two most affected machines
jj = jj[np.argsort(-amp[jj])]      # most affected first
print(f"[Fig2] most-affected machines (1-indexed): {[int(j)+1 for j in jj]} "
      f"(max|COI| = {[round(float(amp[j]),2) for j in jj]} rad)")
# first-swing window of the TRUE trajectory (offline diagnostic)
peak_ends = []
for j in range(n):
    w = np.where(t_out > tc[i])[0]
    jm = np.where(np.diff(np.sign(np.diff(y_i[j, w]))) < 0)[0]
    peak_ends.append(w[0] + jm[0] + 1 if len(jm) else len(t_out) - 1)
fs_end = max(peak_ends)
print(f"[Fig2] first-swing window end t_fs = {t_out[fs_end]:.2f} s")

t_deg = t_out
ydeg = np.rad2deg(y_i); pdeg = np.rad2deg(p_i)
tc_plot = float(tc[i])
fig, ax = plt.subplots(figsize=(3.4, 2.35), dpi=600)
ax.axvspan(0, tc_plot, color=GRAY, alpha=0.18, lw=0, zorder=0)
mask = t_deg <= t_deg[fs_end]
for c, j in enumerate(jj):
    ax.fill_between(t_deg, pdeg[j] - q_fs_deg, pdeg[j] + q_fs_deg, where=mask,
                    color=[BLUE, ORANGE][c], alpha=0.13, lw=0, zorder=1)
for c, j in enumerate(jj):
    col = [BLUE, ORANGE][c]
    ax.plot(t_deg, ydeg[j], color=col, lw=1.3, ls="-",  label=f"true, m{int(j)+1}",  zorder=3)
    ax.plot(t_deg, pdeg[j], color=col, lw=1.1, ls="--", label=f"pred, m{int(j)+1}", zorder=3)
ax.axvline(tc_plot, color="#555555", ls=":", lw=0.9, zorder=2)
ax.text(tc_plot + 0.05, 150, "$t_c$", fontsize=8, color="#222222")
ax.set_xlim(-0.15, 4.1); ax.set_ylim(-115, 175)
ax.set_xlabel("time (s)"); ax.set_ylabel("COI angle (deg)")
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([-100, -50, 0, 50, 100, 150])
ax.yaxis.grid(True, color="#D9D9D9", alpha=0.5, lw=0.6)
ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.legend(loc="upper right", fontsize=6.8, frameon=False, ncol=2,
          handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
plt.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.14)
fig.savefig(OUT + r"\fig2_trajectory.png", dpi=600)
fig.savefig(OUT + r"\fig2_trajectory.pdf")
fig.savefig(OUT + r"\fig2_trajectory.svg")
plt.close(fig)
print("[Fig2] saved (real data)")

# ================= Fig 3: real conformal outputs + Wilson CIs =================
levels = [0.05, 0.10, 0.30, 0.50]   # alpha -> targets 0.95/0.90/0.70/0.50
covs, counts = [], []
for a in levels:
    q = order_quantile(scores[cal], a)
    k = int((scores[eva] <= q).sum())
    covs.append(k / len(scores[eva])); counts.append(k)
    print(f"[Fig3] alpha={a:.2f}: cov={k}/{len(scores[eva])} = {k/len(scores[eva]):.3f} "
          f"(paper target {1-a:.2f})")

# Mondrian groups from real scores
y_s = y[st]; tc_s = tc[st]
peaks = []
for k in range(y_s.shape[0]):
    w = t_out >= tc_s[k]
    peaks.append(np.rad2deg((y_s[k, :, w].max(axis=0) - y_s[k, :, w].min(axis=0)).max()))
peaks = np.array(peaks)
bins = [(0, 45), (45, 90), (90, 180)]
bin_q, bin_cov, bin_annot = [], [], []
for lo, hi in bins:
    mc = (peaks[cal] >= lo) & (peaks[cal] < hi)
    me = (peaks[eva] >= lo) & (peaks[eva] < hi)
    q = order_quantile(scores[cal][mc], ALPHA)
    cov = (scores[eva][me] <= q).mean()
    bin_q.append(np.rad2deg(q)); bin_cov.append(cov)
    bin_annot.append(f"n={int(mc.sum())}/{int(me.sum())}\ncov={cov:.3f}")
    print(f"[Fig3] Mondrian {lo}-{hi}: q={np.rad2deg(q):.1f} deg, "
          f"n={int(mc.sum())}/{int(me.sum())}, cov={cov:.3f}")

n_eva = len(scores[eva])
target = np.array([1 - a for a in levels])
empir = np.array(covs)
cis = [wilson_ci(k, n_eva) for k in counts]
print("[Fig3] Wilson 95% CIs:",
      [f"{k}/{n_eva}: [{lo:.3f},{hi:.3f}]" for k, (lo, hi) in zip(counts, cis)])
err_lo = empir - np.array([c[0] for c in cis])
err_hi = np.array([c[1] for c in cis]) - empir

groups = ["$<$45", "45-90", "$>$90"]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(3.4, 1.75), dpi=600,
                               gridspec_kw={"width_ratios": [1.15, 1.0]})
axL.plot([0, 1], [0, 1], ls="--", color="#333333", lw=0.8, zorder=1)
axL.errorbar(target, empir, yerr=[err_lo, err_hi], fmt="o", color=BLUE, ecolor=BLUE,
             elinewidth=0.8, capsize=2.5, ms=4, lw=0.8, zorder=3)
axL.set_xlim(-0.02, 1.02); axL.set_ylim(-0.02, 1.05)
axL.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); axL.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axL.set_xlabel("target coverage $1-\\alpha$", fontsize=7.2)
axL.set_ylabel("empirical coverage", fontsize=7.2)
axL.set_title("(a) coverage validity", fontsize=7.4)
axL.tick_params(labelsize=6.6)
axL.yaxis.grid(True, color="#D9D9D9", alpha=0.5, lw=0.6); axL.set_axisbelow(True)
for sp in ["top", "right"]: axL.spines[sp].set_visible(False)

bars = axR.bar(groups, bin_q, width=0.62, color=BLUE, alpha=0.85, zorder=2)
for b, txt in zip(bars, bin_annot):
    axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, txt,
             ha="center", va="bottom", fontsize=5.6, color="#111111")
axR.set_ylim(0, 155)
axR.set_yticks([0, 40, 80, 120])
axR.set_ylabel("half-width (deg)", fontsize=7.0)
axR.set_xlabel("post-fault peak of spread (deg)", fontsize=6.2)
axR.set_title("(b) Mondrian (offline)", fontsize=7.4)
axR.tick_params(labelsize=6.4)
axR.yaxis.grid(True, color="#D9D9D9", alpha=0.5, lw=0.6); axR.set_axisbelow(True)
for sp in ["top", "right"]: axR.spines[sp].set_visible(False)

plt.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.24, wspace=0.42)
fig.savefig(OUT + r"\fig3_conformal.png", dpi=600)
fig.savefig(OUT + r"\fig3_conformal.pdf")
fig.savefig(OUT + r"\fig3_conformal.svg")
plt.close(fig)
print("[Fig3] saved (real data, Wilson CIs)")
print("DONE")
