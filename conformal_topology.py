"""Split conformal calibration of the N-1 and N-2 operators.

Checks whether the certified interval, calibrated on a random half of each
held-out split's stable scenarios and evaluated on the disjoint other half,
achieves valid coverage when the operator generalizes to unseen topologies
(N-1 line trips, N-2 line pairs). Mirrors the N-0 split calibration.
"""
import numpy as np
import torch

from model import DeepONet

ALPHA = 0.1
SEED = 0


def split_conformal(scores, alpha=ALPHA, seed=SEED):
    """Split conformal: calibrate on a random half, evaluate coverage on the other."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(scores))
    n_cal = len(scores) // 2
    scores_cal, scores_eva = scores[idx[:n_cal]], scores[idx[n_cal:]]
    q = np.quantile(scores_cal, min(1.0, (1 - alpha) * (1 + 1.0 / n_cal)))
    cov = (scores_eva <= q).mean()
    return np.rad2deg(q), cov, len(scores_cal), len(scores_eva)


def run(tag, datafile, model_path, p, enc_fn, splits):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(datafile)
    n = int(d["n_machines"]); t_end = float(d["t_end"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    model = DeepONet(2 * n + 1, n, p=p, t_scale=t_end).to(dev)
    model.load_state_dict(torch.load(model_path)); model.eval()

    print(f"=== {tag} ({model_path}) ===")
    for name, (fb, tl_or_sev, tc, y, st) in splits.items():
        s = enc_fn(fb, tl_or_sev, tc, d, dev)
        with torch.no_grad():
            pred = model(s, t_out).cpu().numpy()
        err = np.abs(pred - y)
        st = st.astype(bool)
        scores = err[st].max(axis=(1, 2))
        q, cov, n_c, n_e = split_conformal(scores)
        print(f"  {name:12s}: n_stable={st.sum():4d} (cal {n_c} / eva {n_e}), "
              f"interval half-width={q:6.1f} deg, coverage={cov:.3f} (target {1-ALPHA:.2f})")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- N-1: enc = [sev_f (n), sev_p (n), tcn (1)] ----
    def enc_n1(fb, tl, tc, d, dev):
        n = int(d["n_machines"]); Pm = d["Pm"]; M = d["M"]
        pf = d["pe_fault"]; pp = d["pe_post"]; t_max = float(d["t_max"])
        sev_f = (Pm[None, :] - pf[fb - 1]) / M[None, :] * 10.0
        sev_p = np.zeros((len(fb), n))
        valid = tl >= 0
        if valid.any():
            sev_p[valid] = (Pm[None, :] - pp[tl[valid]]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / t_max
        return torch.cat([torch.as_tensor(sev_f, dtype=torch.float32, device=dev),
                          torch.as_tensor(sev_p, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    d1 = np.load("data39_n1.npz")
    run("N-1", "data39_n1.npz", "model_n1_p512.pt", 512, enc_n1, {
        "heldout_bus":  (d1["fb_testb"], d1["tl_testb"], d1["tc_testb"], d1["y_testb"], d1["s_testb"]),
        "heldout_line": (d1["fb_testl"], d1["tl_testl"], d1["tc_testl"], d1["y_testl"], d1["s_testl"]),
    })

    # ---- N-2: enc = [sf (n), sev (n), tcn (1)] with precomputed sev ----
    def enc_n2(fb, sev, tc, d, dev):
        n = int(d["n_machines"]); Pm = d["Pm"]; M = d["M"]
        pf = d["pe_fault"]; t_max = float(d["t_max"])
        sf = (Pm[None, :] - pf[fb - 1]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / t_max
        return torch.cat([torch.as_tensor(sf, dtype=torch.float32, device=dev),
                          torch.as_tensor(sev, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    d2 = np.load("data39_n2.npz")
    run("N-2", "data39_n2.npz", "model_n2_p512.pt", 512, enc_n2, {
        "heldout_bus":  (d2["fb_testb"], d2["sev_testb"], d2["tc_testb"], d2["y_testb"], d2["s_testb"]),
        "heldout_pair": (d2["fb_testn2"], d2["sev_testn2"], d2["tc_testn2"], d2["y_testn2"], d2["s_testn2"]),
    })


if __name__ == "__main__":
    main()
