"""Error-structure analysis of the operator.

(1) First-swing vs later-oscillation decomposition of the post-fault error;
(2) a relative (amplitude-normalized) conformal score;
(3) validation-set calibration transferred to the held-out test set (a
    cross-distribution calibration check).
"""
import numpy as np
import torch

from model import DeepONet
from evaluate import severity_enc


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()

    fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]; st = d["s_test"].astype(bool)
    t_out = d["t_out"]
    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
    err = np.abs(pred - y)

    # ---- (1) first-swing vs later-oscillation (post-fault, pooled, stable) ----
    fs_err, later_err = [], []
    for i in range(len(fb)):
        if not st[i]:
            continue
        mask = t_out >= tc[i]
        tt = t_out[mask]
        ei = err[i][:, mask]
        yi = np.abs(y[i][:, mask])
        t_peak = tt[0]                   # end of the first swing: latest first local max
        any_lmax = False
        for j in range(n):
            for k in range(1, len(tt) - 1):
                if yi[j, k - 1] <= yi[j, k] and yi[j, k] >= yi[j, k + 1]:
                    t_peak = max(t_peak, tt[k])
                    any_lmax = True
                    break
        if not any_lmax:
            t_peak = tt[-1]
        w1 = tt <= t_peak
        if w1.sum() > 0 and (~w1).sum() > 0:
            fs_err.append((ei[:, w1] ** 2).mean())
            later_err.append((ei[:, ~w1] ** 2).mean())
        else:
            fs_err.append((ei ** 2).mean())
    fs_pooled = np.rad2deg(np.sqrt(np.mean(fs_err)))
    later_pooled = np.rad2deg(np.sqrt(np.mean(later_err)))
    print(f"(1) post-fault error structure: first-swing pooled={fs_pooled:.1f} deg, "
          f"later-oscillation pooled={later_pooled:.1f} deg (n={len(fs_err)}/{len(later_err)})")

    # ---- (2) relative (scenario-amplitude-normalized) conformal score ----
    A_s = np.abs(y[st]).max(axis=(1, 2))                # per-scenario amplitude (S,)
    R_rel = (err[st] / (A_s[:, None, None] + 1e-6)).max(axis=(1, 2))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(R_rel)); n_cal = len(R_rel) // 2
    q_rel = np.quantile(R_rel[idx[:n_cal]], min(1.0, 0.9 * (1 + 1.0 / n_cal)))
    cov_rel = (R_rel[idx[n_cal:]] <= q_rel).mean()
    print(f"(2) relative score: q_rel={q_rel:.2f} (band = {q_rel*100:.0f}% of the scenario "
          f"amplitude), coverage={cov_rel:.3f}")

    # ---- (3) validation-set calibration -> held-out test coverage ----
    fb_v = d["fb_val"]; tc_v = d["tc_val"]; y_v = d["y_val"]; s_v = d["s_val"].astype(bool)
    with torch.no_grad():
        pred_v = m(severity_enc(fb_v, tc_v, d, dev), t).cpu().numpy()
    scores_v = np.abs(pred_v - y_v)[s_v].max(axis=(1, 2))
    q_v = np.quantile(scores_v, min(1.0, 0.9 * (1 + 1.0 / len(scores_v))))
    scores_t = err[st].max(axis=(1, 2))
    cov_t = (scores_t <= q_v).mean()
    print(f"(3) val-set calibration: n_val_stable={s_v.sum()}, q={np.rad2deg(q_v):.1f} deg, "
          f"held-out test coverage={cov_t:.3f} (target 0.90)")


if __name__ == "__main__":
    main()
