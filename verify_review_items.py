# -*- coding: utf-8 -*-
"""Verify the reviewer's V1-V11 data items from the actual data and code."""
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)


def tds_cct(model, bus, lo=0.01, hi=0.8):
    for _ in range(16):
        mid = (lo + hi) / 2
        r = simulate_batch(model, np.array([bus]), np.array([mid]), np.array([-1]), 4.0)
        lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
    return lo


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model39 = ClassicalModel(case="39")

    # V1: CCT range over ALL 39 buses and over the 12 held-out buses
    cct_all = [tds_cct(model39, b) for b in range(1, 40)]
    print(f"V1: CCT range all 39 buses: {min(cct_all)*1000:.0f}-{max(cct_all)*1000:.0f} ms")
    d = np.load("data39.npz")
    hb = sorted(np.unique(d["fb_test"]))
    cct_ho = [tds_cct(model39, int(b)) for b in hb]
    print(f"V1: CCT range 12 held-out buses: {min(cct_ho)*1000:.0f}-{max(cct_ho)*1000:.0f} ms")

    # V2: clean accuracy of model_p512 (seed-0) — evaluate.py prints 0.971
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()
    with torch.no_grad():
        pred = m(severity_enc(d["fb_test"], d["tc_test"], d, dev), t).cpu().numpy()
    st = d["s_test"].astype(bool)
    fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
    acc = ((fs < np.pi) == st).mean()
    print(f"V2: seed-0 model clean accuracy = {acc:.4f}")

    # V3: calibration/evaluation sizes per split (split conformal, seed 0)
    for tag, datafile, model_path, keys in [
        ("N-1 bus", "data39_n1.npz", "model_n1_p512.pt", ("fb_testb", "tl_testb", "tc_testb", "y_testb", "s_testb")),
        ("N-1 line", "data39_n1.npz", "model_n1_p512.pt", ("fb_testl", "tl_testl", "tc_testl", "y_testl", "s_testl")),
        ("N-2 bus", "data39_n2.npz", "model_n2_p512.pt", ("fb_testb", None, "tc_testb", "y_testb", "s_testb")),
        ("N-2 pair", "data39_n2.npz", "model_n2_p512.pt", ("fb_testn2", None, "tc_testn2", "y_testn2", "s_testn2")),
    ]:
        dn = np.load(datafile)
        st_n = dn[keys[4]].astype(bool)
        n_st = int(st_n.sum())
        print(f"V3: {tag}: n_stable={n_st}, n_cal={n_st//2}, n_eva={n_st - n_st//2}")

    # V6: 1.2-load stable count
    dl = np.load("data39_loadscale.npz")
    for lv in sorted(np.unique(dl["ls_train"])):
        sel = dl["ls_train"] == lv
        n_st = int(dl["s_train"][sel].astype(bool).sum())
        print(f"V6: load={1+lv:.1f}: {n_st}/{int(sel.sum())} stable")

    # V7: multi-topology training composition
    d1 = np.load("data39_n1.npz")
    tl_tr = d1["tl_train"]
    n0 = int((tl_tr < 0).sum())
    n1 = int((tl_tr >= 0).sum())
    print(f"V7: N-1 training: total={len(tl_tr)}, N-0={n0}, N-1={n1}, "
          f"unique trip lines={len(np.unique(tl_tr[tl_tr>=0]))}")
    d2 = np.load("data39_n2.npz")
    sev_tr = d2["sev_train"]
    n0_2 = int((np.abs(sev_tr).sum(axis=1) == 0).sum())
    print(f"V7: N-2 training: total={len(sev_tr)}, N-0 scenarios={n0_2}, "
          f"testn2 scenarios={len(d2['fb_testn2'])}")
    fb2 = d2["fb_testn2"]; sev2 = d2["sev_testn2"]
    combos = list(dict.fromkeys(zip(fb2.tolist(), [tuple(s) for s in sev2.tolist()])))
    n_buses = len(np.unique(fb2)); n_pairs = len({tuple(np.round(s,3)) for s in sev2})
    print(f"V7: N-2 testn2: {len(fb2)} scenarios = {n_buses} buses x {n_pairs} unique pairs "
          f"-> {len(combos)} unique combos")

    # V8: Mondrian group sizes (from conformal_analysis split output)
    err = np.abs(pred - d["y_test"])
    scores = err[st].max(axis=(1, 2))
    peaks = np.rad2deg(d["y_test"][st].max(axis=(1, 2)))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(scores)); nc = len(scores) // 2
    for lo, hi in [(0, 45), (45, 90), (90, 135), (135, 180)]:
        mc = ((peaks[idx[:nc]] >= lo) & (peaks[idx[:nc]] < hi)).sum()
        me = ((peaks[idx[nc:]] >= lo) & (peaks[idx[nc:]] < hi)).sum()
        print(f"V8: Mondrian {lo}-{hi}: n_cal={mc}, n_eva={me}")

    # V11: damped coverage over 5 seeds (round-8 data)
    dd = np.load("data39_damped.npz")
    with torch.no_grad():
        pd_ = m(severity_enc(dd["fb_test"], dd["tc_test"], dd, dev), t).cpu().numpy()
    sd = dd["s_test"].astype(bool)
    sd_scores = np.abs(pd_ - dd["y_test"])[sd].max(axis=(1, 2))
    covs = []
    for seed in range(5):
        rr = np.random.default_rng(seed)
        ix = rr.permutation(len(sd_scores)); ncc = len(sd_scores) // 2
        q = np.quantile(sd_scores[ix[:ncc]], min(1.0, 0.9 * (1 + 1.0 / ncc)))
        covs.append((sd_scores[ix[ncc:]] <= q).mean())
    print(f"V11: damped coverage over 5 splits: {np.round(covs, 3)}, "
          f"mean={np.mean(covs):.3f} +/- {np.std(covs):.3f}")


if __name__ == "__main__":
    main()
