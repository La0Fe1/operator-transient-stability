"""Round-10 computations.

(1) Operating-point calibration transfer: coverage of the N-0-calibrated interval
    on load-scaled (0.8) scenarios.
(2) Severity-encoding cost on the 118-bus system (Kron reduction per fault bus).
"""
import time

import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- (1) coverage transfer to 0.8 load ----------
    d0 = np.load("data39.npz")
    n = int(d0["n_machines"]); te = float(d0["t_end"])
    t = torch.as_tensor(d0["t_out"], **F32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()

    # N-0 calibration q (split, seed 0)
    with torch.no_grad():
        pred0 = m(severity_enc(d0["fb_test"], d0["tc_test"], d0, dev), t).cpu().numpy()
    st0 = d0["s_test"].astype(bool)
    scores0 = np.abs(pred0 - d0["y_test"])[st0].max(axis=(1, 2))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(scores0)); nc = len(scores0) // 2
    q0 = np.quantile(scores0[idx[:nc]], min(1.0, 0.9 * (1 + 1.0 / nc)))

    dl = np.load("data39_loadscale.npz")
    tm = float(dl["t_max"])
    for lv in sorted(np.unique(dl["ls_train"])):
        sel = dl["ls_train"] == lv
        s_l = torch.cat([torch.as_tensor(dl["sev_train"][sel], **F32, device=dev),
                         torch.as_tensor(dl["tc_train"][sel, None] / tm, **F32, device=dev)], dim=1)
        with torch.no_grad():
            pred_l = m(s_l, t).cpu().numpy()
        st_l = dl["s_train"][sel].astype(bool)
        scores_l = np.abs(pred_l - dl["y_train"][sel])[st_l].max(axis=(1, 2))
        cov = (scores_l <= q0).mean()
        print(f"(1) load={1+lv:.1f}: n_stable={st_l.sum()}, N-0 q={np.rad2deg(q0):.1f} deg, "
              f"coverage on load-shifted scenarios={cov:.3f}", flush=True)

    # ---------- (2) 118-bus encoding cost ----------
    import generate_118_authoritative as g
    model118 = ClassicalModel(case="118", gendyn=g.build_gendyn_118())
    t0 = time.time()
    for _ in range(5):
        for b in range(1, 119):
            model118.fault_yint(b)
    per_bus = (time.time() - t0) / 5 / 118
    print(f"(2) 118-bus fault_yint (Kron reduction): {per_bus*1e3:.2f} ms per fault bus "
          f"(39-bus: 0.13 ms)", flush=True)


if __name__ == "__main__":
    main()
