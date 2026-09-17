# -*- coding: utf-8 -*-
"""B4: decidable fraction of the screening rule spread + q_pw(alpha) < pi,
and the q_pw-vs-alpha trade-off, on the disjoint evaluation half."""
import numpy as np
import torch

from model import DeepONet
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)


def order_quantile(scores, alpha):
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
    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
    sm = (pred.max(axis=1) - pred.min(axis=1)).max(axis=1)
    signed = (pred - y)[st]
    R_pw = np.abs(signed[:, None, :, :] - signed[:, :, None, :]).max(axis=(1, 2, 3))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(R_pw)); nc = len(R_pw) // 2
    eva = idx[nc:]

    print("alpha | q_pw (deg) | coverage | decidable fraction (pred-stable, eval half)")
    for alpha in (0.1, 0.2, 0.3, 0.4, 0.5):
        q = order_quantile(R_pw[idx[:nc]], alpha)
        cov = (R_pw[eva] <= q).mean()
        gate = sm[eva] < np.pi
        dec = (sm[eva] + q < np.pi) & gate
        frac = dec.sum() / gate.sum() if gate.sum() else np.nan
        print(f"{alpha:.1f}  | {np.rad2deg(q):6.1f} | {cov:.3f} | {frac:.3f} "
              f"({int(dec.sum())}/{int(gate.sum())})")


if __name__ == "__main__":
    main()
