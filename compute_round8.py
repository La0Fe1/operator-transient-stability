"""Round-8 computations.

(1) Gated-regime conformal coverage: apply the interval only to scenarios the
    operator PREDICTS stable; report gate purity and coverage within the gate.
(2) N-2 held-out line-pair CCT error of the multi-topology operator.
"""
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from generate_data_n2 import simulate_batch_multi

F32 = dict(dtype=torch.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- (1) gated-regime coverage (N-0, model_p512) ----------
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()

    from evaluate import severity_enc
    fb, tc, y, st = d["fb_test"], d["tc_test"], d["y_test"], d["s_test"].astype(bool)
    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()
    pred_stable = (pred.max(axis=1) - pred.min(axis=1)).max(axis=1) < np.pi   # A2
    gate_purity = st[pred_stable].mean()
    print(f"(1) gate: predicted-stable={pred_stable.sum()}/960, "
          f"purity (P(true stable | predicted stable))={gate_purity:.3f}", flush=True)

    # conformal q calibrated on true-stable (split), coverage within the gate
    err = np.abs(pred - y)
    scores = err[st].max(axis=(1, 2))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(scores)); nc = len(scores) // 2
    k = int(np.ceil((nc + 1) * 0.9))               # A4: order-statistic quantile
    q = np.sort(scores[idx[:nc]])[k - 1] if k <= nc else np.inf
    # A3: gated coverage on the DISJOINT EVALUATION HALF only
    eva = idx[nc:]
    st_e, ps_e = st[eva], pred_stable[eva]
    gate = ps_e
    purity = st_e[gate].mean() if gate.sum() else np.nan
    wr = (~st_e & gate).sum() / (~st_e).sum() if (~st_e).sum() else np.nan
    cov_all_ps = (err[eva][gate].max(axis=(1, 2)) <= q).mean() if gate.sum() else np.nan
    in_gate = st_e & gate                        # oracle-conditioned diagnostic
    cov_ts_ps = (err[eva][in_gate].max(axis=(1, 2)) <= q).mean() if in_gate.sum() else np.nan
    print(f"(1) gate (eval half, n={len(eva)}): purity={purity:.3f}, "
          f"wrong-release={wr:.3f}, cov(pred-stable)={cov_all_ps:.3f} (n={int(gate.sum())}), "
          f"cov(true-stable&pred-stable, oracle)={cov_ts_ps:.3f} (n={int(in_gate.sum())}), "
          f"q={np.rad2deg(q):.1f} deg", flush=True)

    # ---------- (2) N-2 CCT error ----------
    dn = np.load("data39_n2.npz")
    n2 = int(dn["n_machines"]); te2 = float(dn["t_end"]); tm2 = float(dn["t_max"])
    t2 = torch.as_tensor(dn["t_out"], **F32, device=dev)[:, None]
    Pm = dn["Pm"]; M = dn["M"]; pf = dn["pe_fault"]
    m2 = DeepONet(2 * n2 + 1, n2, p=512, t_scale=te2).to(dev)
    m2.load_state_dict(torch.load("model_n2_p512.pt")); m2.eval()

    fb2 = dn["fb_testn2"]; sev2 = dn["sev_testn2"]; tc2 = dn["tc_testn2"]
    combos = list(dict.fromkeys(zip(fb2.tolist(), [tuple(s) for s in sev2.tolist()])))

    def enc2(bus, sev, tcc):
        sf = (Pm[None, :] - pf[bus - 1]) / M[None, :] * 10.0
        tcn = np.asarray(tcc, dtype=np.float32)[:, None] / tm2
        return torch.cat([torch.as_tensor(sf, **F32, device=dev),
                          torch.as_tensor(sev, **F32, device=dev),
                          torch.as_tensor(tcn, **F32, device=dev)], dim=1)

    model39 = ClassicalModel(case="39")
    # reconstruct the held-out test line pairs exactly as generate_data_n2 does
    import generate_data_n2 as g
    rng2 = np.random.default_rng(g.SEED)
    all_buses = np.arange(1, 40); rng2.shuffle(all_buses)
    all_lines = np.arange(len(model39.net.line)); rng2.shuffle(all_lines)
    pairs = [(a, b) for i, a in enumerate(all_lines) for b in all_lines[i + 1:]]
    rng2.shuffle(pairs)
    test_pairs = pairs[:150]
    sev_to_pair = {}
    for (a, b) in test_pairs:
        pe = model39._pe(model39.delta0, model39.post_yint_multi([a, b]))
        sv = (model39.Pm - pe) / model39.M * 10.0          # float64, as saved in the npz
        sev_to_pair[tuple(np.round(sv, 3))] = (a, b)

    def true_cct(bus, sev):
        trips_list = sev_to_pair[tuple(np.round(np.array(sev), 3))]
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            r = simulate_batch_multi(model39, np.array([bus]), [list(trips_list)], np.array([mid]),
                                     te, dt=0.001, out_dt=0.01)
            lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
        return lo

    def pred_cct(bus, sev):
        def stable(tcc):
            with torch.no_grad():
                tr = m2(enc2(np.array([bus]), np.array([sev]), np.array([tcc])), t2).cpu().numpy()
            return (tr[:, :, -1].max() - tr[:, :, -1].min()) < np.pi
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if stable(mid) else (lo, mid)
        return lo

    errs = []
    for bus, sev in combos:
        e = abs(pred_cct(int(bus), np.array(sev)) - true_cct(int(bus), sev)) * 1000
        errs.append(e)
    print(f"(2) N-2 CCT: mean={np.mean(errs):.1f} ms, max={np.max(errs):.1f} ms "
          f"(n_combos={len(combos)})", flush=True)


if __name__ == "__main__":
    main()
