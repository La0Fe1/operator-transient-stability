"""Tighter conformal calibration: per-machine scores with Bonferroni / Sidak correction.

The paper's score is the worst-case error over machines AND time (q=111 deg at
alpha=0.1), whose pairwise bound 2q=222 deg exceeds pi. Here we calibrate each
machine separately (worst case over time only) with a multiplicity correction, and
derive the certified pairwise bound max_{i,j}(q_i + q_j).
"""
import numpy as np
import torch

from model import DeepONet
from evaluate import severity_enc

ALPHA = 0.1


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

    # per-machine worst-over-time scores: (n_stable, n)
    R = err[st].max(axis=2)
    # global score (paper)
    Rg = err[st].max(axis=(1, 2))

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(R))
    n_cal = len(R) // 2
    R_cal, R_eva = R[idx[:n_cal]], R[idx[n_cal:]]
    Rg_cal, Rg_eva = Rg[idx[:n_cal]], Rg[idx[n_cal:]]

    # global score (paper's setting)
    qg = np.quantile(Rg_cal, min(1.0, (1 - ALPHA) * (1 + 1.0 / n_cal)))
    covg = (Rg_eva <= qg).mean()
    print(f"global max score       : q={np.rad2deg(qg):6.1f} deg, coverage={covg:.3f}, pairwise 2q={np.rad2deg(2*qg):.0f} deg")

    # Bonferroni: per-machine level alpha/n
    a_bonf = ALPHA / n
    q_bonf = np.array([np.quantile(R_cal[:, i], min(1.0, (1 - a_bonf) * (1 + 1.0 / n_cal)))
                       for i in range(n)])
    cov_bonf = (R_eva <= q_bonf[None, :]).all(axis=1).mean()
    pw_bonf = (q_bonf[:, None] + q_bonf[None, :]).max()
    print(f"Bonferroni per-machine : q mean={np.rad2deg(q_bonf.mean()):6.1f} deg, joint coverage={cov_bonf:.3f}, pairwise bound={np.rad2deg(pw_bonf):.0f} deg")
    print(f"  per-machine q: " + " ".join(f"{np.rad2deg(q):.0f}" for q in q_bonf))

    # Sidak: per-machine level 1-(1-alpha)^(1/n)
    a_sidak = 1 - (1 - ALPHA) ** (1 / n)
    q_sidak = np.array([np.quantile(R_cal[:, i], min(1.0, (1 - a_sidak) * (1 + 1.0 / n_cal)))
                        for i in range(n)])
    cov_sidak = (R_eva <= q_sidak[None, :]).all(axis=1).mean()
    pw_sidak = (q_sidak[:, None] + q_sidak[None, :]).max()
    print(f"Sidak per-machine      : q mean={np.rad2deg(q_sidak.mean()):6.1f} deg, joint coverage={cov_sidak:.3f}, pairwise bound={np.rad2deg(pw_sidak):.0f} deg")

    print(f"\npi = {np.rad2deg(np.pi):.0f} deg: pairwise bound below pi? "
          f"global={2*qg < np.pi}, Bonferroni={pw_bonf < np.pi}, Sidak={pw_sidak < np.pi}")

    # ---- direct pairwise-difference score (the quantity the pi-criterion needs) ----
    # R_pw(s) = max_{i,j,t} |e_i(t) - e_j(t)|
    e = err[st]                                     # (n_stable, n, T)
    diffs = np.abs(e[:, None, :, :] - e[:, :, None, :])   # (n_stable, n, n, T)
    R_pw = diffs.max(axis=(1, 2, 3))                # worst pairwise-difference error per scenario
    R_pw_cal, R_pw_eva = R_pw[idx[:n_cal]], R_pw[idx[n_cal:]]
    q_pw = np.quantile(R_pw_cal, min(1.0, (1 - ALPHA) * (1 + 1.0 / n_cal)))
    cov_pw = (R_pw_eva <= q_pw).mean()
    print(f"direct pairwise score : q={np.rad2deg(q_pw):6.1f} deg, coverage={cov_pw:.3f} "
          f"-> certified pairwise bound={np.rad2deg(q_pw):.0f} deg (below pi: {q_pw < np.pi})")


if __name__ == "__main__":
    main()
