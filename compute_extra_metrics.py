"""Extra metrics requested in the 7th review round.

(1) IEEE 118-bus: split-conformal coverage and CCT error;
(2) N-1 / N-2 held-out splits: CCT error of the multi-topology operators.
"""
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch

F32 = dict(dtype=torch.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ (1) 118-bus: conformal + CCT ============
    d = np.load("data118.npz")
    n = int(d["n_machines"]); te = float(d["t_end"]); tm = float(d["t_max"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    Pm = d["Pm"]; M = d["M"]; pf = d["pe_fault"]

    def enc(fb, tc):
        s = (Pm[None, :] - pf[fb - 1]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / tm
        return torch.cat([torch.as_tensor(s, **F32, device=dev),
                          torch.as_tensor(tcn, **F32, device=dev)], dim=1)

    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_118.pt")); m.eval()
    fb, tc, y, st = d["fb_test"], d["tc_test"], d["y_test"], d["s_test"].astype(bool)
    with torch.no_grad():
        pred = m(enc(fb, tc), t).cpu().numpy()
    err = np.abs(pred - y)
    scores = err[st].max(axis=(1, 2))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(scores)); nc = len(scores) // 2
    q = np.quantile(scores[idx[:nc]], min(1.0, 0.9 * (1 + 1.0 / nc)))
    cov = (scores[idx[nc:]] <= q).mean()
    print(f"118-bus conformal: n_stable={st.sum()}, q={np.rad2deg(q):.1f} deg, coverage={cov:.3f}",
          flush=True)

    # 118 CCT (unique held-out buses; true vs operator binary search)
    # reconstruct the authoritative model as in generate_118_authoritative.py
    import generate_118_authoritative as g
    model_ref = ClassicalModel(case="118", gendyn=g.build_gendyn_118())

    def true_cct(bus):
        lo, hi = 0.005, 0.2
        for _ in range(16):
            mid = (lo + hi) / 2
            r = simulate_batch(model_ref, np.array([bus]), np.array([mid]), np.array([-1]), 4.0,
                               dt=0.001, out_dt=0.01)
            lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
        return lo

    def pred_cct(bus):
        def stable(tc):
            with torch.no_grad():
                tr = m(enc(np.array([bus]), np.array([tc])), t).cpu().numpy()
            return (tr[:, :, -1].max() - tr[:, :, -1].min()) < np.pi
        lo, hi = 0.005, 0.2
        for _ in range(16):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if stable(mid) else (lo, mid)
        return lo

    buses118 = np.unique(fb)
    cct_errs = [abs(pred_cct(int(b)) - true_cct(int(b))) * 1000 for b in buses118]
    print(f"118-bus CCT error: mean={np.mean(cct_errs):.1f} ms, max={np.max(cct_errs):.1f} ms "
          f"(n_buses={len(buses118)})", flush=True)

    # ============ (2) N-1 / N-2 CCT ============
    def cct_n1(tag, split):
        dn = np.load("data39_n1.npz")
        n1 = int(dn["n_machines"]); te1 = float(dn["t_end"]); tm1 = float(dn["t_max"])
        t1 = torch.as_tensor(dn["t_out"], **F32, device=dev)[:, None]
        Pm1 = dn["Pm"]; M1 = dn["M"]; pf1 = dn["pe_fault"]; pp1 = dn["pe_post"]
        m1 = DeepONet(2 * n1 + 1, n1, p=512, t_scale=te1).to(dev)
        m1.load_state_dict(torch.load(f"model_{tag}.pt")); m1.eval()

        def enc1(fb, tl, tc):
            sf = (Pm1[None, :] - pf1[fb - 1]) / M1[None, :] * 10.0
            sp = np.zeros((len(fb), n1)); v = tl >= 0
            if v.any():
                sp[v] = (Pm1[None, :] - pp1[tl[v]]) / M1[None, :] * 10.0
            tcn = np.asarray(tc, dtype=np.float32)[:, None] / tm1
            return torch.cat([torch.as_tensor(sf, **F32, device=dev),
                              torch.as_tensor(sp, **F32, device=dev),
                              torch.as_tensor(tcn, **F32, device=dev)], dim=1)

        if split == "bus":
            fb, tl = dn["fb_testb"], dn["tl_testb"]
        else:
            fb, tl = dn["fb_testl"], dn["tl_testl"]
        combos = list(dict.fromkeys(zip(fb.tolist(), tl.tolist())))
        model39 = ClassicalModel(case="39")

        def true(bus, line):
            lo, hi = 0.01, 0.8
            for _ in range(16):
                mid = (lo + hi) / 2
                r = simulate_batch(model39, np.array([bus]), np.array([mid]), np.array([line]), 4.0)
                lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
            return lo

        def pred(bus, line):
            def stable(tc):
                with torch.no_grad():
                    tr = m1(enc1(np.array([bus]), np.array([line]), np.array([tc])), t1).cpu().numpy()
                return (tr[:, :, -1].max() - tr[:, :, -1].min()) < np.pi
            lo, hi = 0.01, 0.8
            for _ in range(16):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if stable(mid) else (lo, mid)
            return lo

        errs = [abs(pred(b, l) - true(b, l)) * 1000 for b, l in combos]
        print(f"{tag} {split}: CCT error mean={np.mean(errs):.1f} ms, max={np.max(errs):.1f} ms "
              f"(n_combos={len(combos)})", flush=True)

    cct_n1("n1_p512", "line")
    cct_n1("n1_p512", "bus")


if __name__ == "__main__":
    main()
