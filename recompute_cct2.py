# -*- coding: utf-8 -*-
"""CCT + multi-seed numbers under the corrected protocol:
  1. four-seed classification accuracy (max-over-time)
  2. 118-bus CCT (24 held-out buses)
  3. one-axis CCT (12 held-out buses, split one-axis reference)
  4. damped CCT (12 held-out buses, D=1 reference)
  5. N-1 CCT (held-out trip lines and held-out fault buses)
  6. N-2 CCT (sampled held-out line pairs)
"""
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from power_system_oneaxis import OneAxisModel
from dataset import simulate_batch
from evaluate import severity_enc
from regen_all_refs import sim_oneaxis

F32 = dict(dtype=torch.float32)


def spread_max(p):
    return (p.max(axis=1) - p.min(axis=1)).max(axis=1)


def cct_search(stable_fn, lo=0.01, hi=0.8, iters=16):
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if stable_fn(mid) else (lo, mid)
    return lo


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. four-seed accuracy ----
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    fb, tc, st = d["fb_test"], d["tc_test"], d["s_test"].astype(bool)
    accs = []
    for path in ("model_p512.pt", "model_p512_s1.pt", "model_p512_s2.pt", "model_p512_s3.pt"):
        mm = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
        mm.load_state_dict(torch.load(path)); mm.eval()
        with torch.no_grad():
            pr = mm(severity_enc(fb, tc, d, dev), t).cpu().numpy()
        accs.append(((spread_max(pr) < np.pi) == st).mean())
    accs = np.array(accs)
    print(f"4-seed acc (max-over-time): {[round(a*100,2) for a in accs]}  mean={accs.mean()*100:.2f} "
          f"std={accs.std()*100:.2f}", flush=True)

    # ---- 2. 118 CCT ----
    d118 = np.load("data118.npz")
    n118 = int(d118["n_machines"]); te118 = float(d118["t_end"])
    t118 = torch.as_tensor(d118["t_out"], **F32, device=dev)[:, None]
    m118 = DeepONet(n118 + 1, n118, p=512, t_scale=te118).to(dev)
    m118.load_state_dict(torch.load("model_118.pt")); m118.eval()
    from generate_118_authoritative import build_gendyn_118
    from power_system import ClassicalModel as CM
    model118 = CM(case="118", gendyn=build_gendyn_118())
    buses118 = np.unique(d118["fb_test"])
    errs = []
    for b in buses118:
        ct = cct_search(lambda mid: simulate_batch(model118, np.array([b]), np.array([mid]),
                                                   np.array([-1]), 4.0, dt=0.0005)["stable"][0],
                        lo=0.005, hi=0.2)
        def stab(mid):
            with torch.no_grad():
                tr = m118(severity_enc(np.array([b]), np.array([mid]), d118, dev),
                          t118).cpu().numpy()
            return spread_max(tr)[0] < np.pi
        cp = cct_search(stab, lo=0.005, hi=0.2)
        errs.append((cp - ct) * 1000)
    e = np.array(errs)
    print(f"118 CCT: mean={e.mean():.2f} |mean|={np.abs(e).mean():.2f} "
          f"max={np.abs(e).max():.1f} ms (n={len(e)})", flush=True)

    # ---- 3. one-axis CCT ----
    d1a = np.load("data39_oneaxis.npz")
    n1a = int(d1a["n_machines"]); te1a = float(d1a["t_end"])
    t1a = torch.as_tensor(d1a["t_out"], **F32, device=dev)[:, None]
    m1a = DeepONet(n1a + 1, n1a, p=512, t_scale=te1a).to(dev)
    m1a.load_state_dict(torch.load("model_oneaxis_p512.pt")); m1a.eval()
    oa = OneAxisModel(case="39")
    errs = []
    for b in np.unique(d1a["fb_test"]):
        ct = cct_search(lambda mid: sim_oneaxis(oa, np.array([b]), np.array([mid]))["stable"][0])
        def stab(mid):
            with torch.no_grad():
                tr = m1a(severity_enc(np.array([b]), np.array([mid]), d1a, dev),
                         t1a).cpu().numpy()
            return spread_max(tr)[0] < np.pi
        cp = cct_search(stab)
        errs.append((cp - ct) * 1000)
    e = np.array(errs)
    print(f"one-axis CCT: mean={e.mean():.2f} |mean|={np.abs(e).mean():.2f} "
          f"max={np.abs(e).max():.1f} ms", flush=True)

    # ---- 4. damped CCT ----
    dd = np.load("data39_damped.npz")
    nd = int(dd["n_machines"]); ted = float(dd["t_end"])
    td = torch.as_tensor(dd["t_out"], **F32, device=dev)[:, None]
    md = DeepONet(nd + 1, nd, p=512, t_scale=ted).to(dev)
    md.load_state_dict(torch.load("model_damped_p512.pt")); md.eval()
    modelD = ClassicalModel(case="39", damping=1.0)
    errs = []
    for b in np.unique(dd["fb_test"]):
        ct = cct_search(lambda mid: simulate_batch(modelD, np.array([b]), np.array([mid]),
                                                   np.array([-1]), 4.0, dt=0.0005)["stable"][0])
        def stab(mid):
            with torch.no_grad():
                tr = md(severity_enc(np.array([b]), np.array([mid]), dd, dev), td).cpu().numpy()
            return spread_max(tr)[0] < np.pi
        cp = cct_search(stab)
        errs.append((cp - ct) * 1000)
    e = np.array(errs)
    print(f"damped CCT: mean={e.mean():.2f} |mean|={np.abs(e).mean():.2f} "
          f"max={np.abs(e).max():.1f} ms", flush=True)

    # ---- 5. N-1 CCT ----
    dn = np.load("data39_n1.npz")
    n1 = int(dn["n_machines"]); te1 = float(dn["t_end"]); tm1 = float(dn["t_max"])
    t1 = torch.as_tensor(dn["t_out"], **F32, device=dev)[:, None]
    m1 = DeepONet(2 * n1 + 1, n1, p=512, t_scale=te1).to(dev)
    m1.load_state_dict(torch.load("model_n1_p512.pt")); m1.eval()
    Pm = dn["Pm"]; M = dn["M"]; pf = dn["pe_fault"]; pp = dn["pe_post"]
    model39 = ClassicalModel(case="39")

    def enc1(fb_, tl_, tc_):
        sf = (Pm[None, :] - pf[np.asarray(fb_) - 1]) / M[None, :] * 10.0
        sp_ = np.zeros((len(fb_), n1))
        v = np.asarray(tl_) >= 0
        if v.any():
            sp_[v] = (Pm[None, :] - pp[np.asarray(tl_)[v]]) / M[None, :] * 10.0
        tcn = np.asarray(tc_, dtype=np.float32)[:, None] / tm1
        return torch.cat([torch.as_tensor(sf, **F32, device=dev),
                          torch.as_tensor(sp_, **F32, device=dev),
                          torch.as_tensor(tcn, **F32, device=dev)], dim=1)

    held_lines = np.unique(dn["tl_testl"])
    tr_buses = np.unique(dn["fb_testb"])          # training buses used in N-1 test bus split
    errs_lines = []
    for l in held_lines:
        for b in tr_buses[:2]:
            ct = cct_search(lambda mid: simulate_batch(model39, np.array([b]), np.array([mid]),
                                                       np.array([l]), 4.0, dt=0.0005)["stable"][0])
            def stab(mid, b=b, l=l):
                with torch.no_grad():
                    tr = m1(enc1([b], [l], [mid]), t1).cpu().numpy()
                return spread_max(tr)[0] < np.pi
            cp = cct_search(stab)
            errs_lines.append((cp - ct) * 1000)
    e = np.array(errs_lines)
    print(f"N-1 held-out lines CCT (10 lines x 2 buses): mean={e.mean():.2f} "
          f"|mean|={np.abs(e).mean():.2f} max={np.abs(e).max():.1f} ms", flush=True)

    held_buses = np.unique(dn["fb_testb"])
    errs_bus = []
    for b in held_buses:
        ct = cct_search(lambda mid: simulate_batch(model39, np.array([b]), np.array([mid]),
                                                   np.array([-1]), 4.0, dt=0.0005)["stable"][0])
        def stab(mid, b=b):
            with torch.no_grad():
                tr = m1(enc1([b], [-1], [mid]), t1).cpu().numpy()
            return spread_max(tr)[0] < np.pi
        cp = cct_search(stab)
        errs_bus.append((cp - ct) * 1000)
    e = np.array(errs_bus)
    print(f"N-1 held-out buses CCT (12 buses): mean={e.mean():.2f} "
          f"|mean|={np.abs(e).mean():.2f} max={np.abs(e).max():.1f} ms", flush=True)

    # ---- 6. N-2 CCT (sampled held-out pairs) ----
    dn2 = np.load("data39_n2.npz")
    n2 = int(dn2["n_machines"]); te2 = float(dn2["t_end"]); tm2 = float(dn2["t_max"])
    t2 = torch.as_tensor(dn2["t_out"], **F32, device=dev)[:, None]
    m2 = DeepONet(2 * n2 + 1, n2, p=512, t_scale=te2).to(dev)
    m2.load_state_dict(torch.load("model_n2_p512.pt")); m2.eval()
    # reproduce the held-out pairs exactly (same seed logic as generate_data_n2)
    from generate_data_n2 import simulate_batch_multi
    n_line = len(model39.net.line)
    rng2 = np.random.default_rng(0)
    all_buses2 = np.arange(1, 40); rng2.shuffle(all_buses2)
    tr_buses2 = all_buses2[12:]
    all_lines2 = np.arange(n_line); rng2.shuffle(all_lines2)
    pairs2 = [(a, b) for i, a in enumerate(all_lines2) for b in all_lines2[i + 1:]]
    rng2.shuffle(pairs2)
    test_pairs2 = pairs2[:150]

    def sev_post(trip_list):
        pe = model39._pe(model39.delta0, model39.post_yint_multi(list(trip_list)))
        return (model39.Pm - pe) / model39.M * 10.0

    errs_p = []
    for trip in test_pairs2[:10]:
        b = int(tr_buses2[0])
        ct = cct_search(lambda mid: simulate_batch_multi(model39, np.array([b]), [tuple(trip)],
                                                         np.array([mid]), 4.0, 0.0005,
                                                         0.01)["stable"][0])
        def stab(mid, b=b, trip=trip):
            sf = (dn2["Pm"][None, :] - dn2["pe_fault"][int(b) - 1]) / dn2["M"][None, :] * 10.0
            tcn = np.asarray([mid], dtype=np.float32)[:, None] / tm2
            s2_ = torch.cat([torch.as_tensor(sf, **F32, device=dev),
                             torch.as_tensor(sev_post(trip)[None, :], **F32, device=dev),
                             torch.as_tensor(tcn, **F32, device=dev)], dim=1)
            with torch.no_grad():
                tr = m2(s2_, t2).cpu().numpy()
            return spread_max(tr)[0] < np.pi
        cp = cct_search(stab)
        errs_p.append((cp - ct) * 1000)
    e = np.array(errs_p)
    print(f"N-2 sampled pairs CCT (10 pairs x 1 bus): mean={e.mean():.2f} "
          f"|mean|={np.abs(e).mean():.2f} max={np.abs(e).max():.1f} ms", flush=True)


if __name__ == "__main__":
    main()
