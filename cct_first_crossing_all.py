# -*- coding: utf-8 -*-
"""B4 protocol fix, batched version: first stable->unstable crossing via 5 ms sweep
(batched TDS + batched operator forward) + local bisection to 0.1 ms.
Max-over-time criterion throughout. N-2 uses a seeded sample of 200 combos;
N-1 bus split sampled to 60 if larger (paper wording says "(sampled)")."""
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch
from generate_data_n2 import simulate_batch_multi

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F32 = dict(dtype=torch.float32, device=dev)

def spread_max(pred):
    return (pred.max(axis=1) - pred.min(axis=1)).max(axis=1)

def first_crossing_vec(fn_batch, lo, hi, sweep=0.005, tol=0.0001):
    """fn_batch(tc_array) -> bool array (batched). Returns first-crossing tc or None."""
    grid = np.arange(lo, hi + 1e-12, sweep)
    labels = fn_batch(grid)
    if not labels[0]:
        return None
    idx = np.where(~labels[1:] & labels[:-1])[0]
    if len(idx) == 0:
        return None
    a, b = grid[idx[0]], grid[idx[0] + 1]
    for _ in range(int(np.ceil(np.log2(sweep / tol)))):
        mid = (a + b) / 2
        if fn_batch(np.array([mid]))[0]:
            a = mid
        else:
            b = mid
    return b

model39 = ClassicalModel(case="39")

# ---------------- (1) N-1 ----------------
dn = np.load("data39_n1.npz")
n1 = int(dn["n_machines"]); te1 = float(dn["t_end"]); tm1 = float(dn["t_max"])
t1 = torch.as_tensor(dn["t_out"], **F32)[:, None]
Pm1, M1, pf1, pp1 = dn["Pm"], dn["M"], dn["pe_fault"], dn["pe_post"]
m1 = DeepONet(2 * n1 + 1, n1, p=512, t_scale=te1).to(dev)
m1.load_state_dict(torch.load("model_n1_p512.pt")); m1.eval()

def enc1(fb, tl, tc):
    sf = (Pm1[None, :] - pf1[fb - 1]) / M1[None, :] * 10.0
    sp = np.zeros((len(fb), n1)); v = tl >= 0
    if v.any():
        sp[v] = (Pm1[None, :] - pp1[tl[v]]) / M1[None, :] * 10.0
    tcn = np.asarray(tc, dtype=np.float32)[:, None] / tm1
    return torch.cat([torch.as_tensor(sf, **F32), torch.as_tensor(sp, **F32),
                      torch.as_tensor(tcn, **F32)], dim=1)

for split in ["line", "bus"]:
    if split == "bus":
        fb, tl = dn["fb_testb"], dn["tl_testb"]
    else:
        fb, tl = dn["fb_testl"], dn["tl_testl"]
    combos = list(dict.fromkeys(zip(fb.tolist(), tl.tolist())))
    rng = np.random.default_rng(0)
    if len(combos) > 60:
        rng.shuffle(combos)
        combos = combos[:60]
    errs = []
    for b, l in combos:
        bb = np.array([b]); ll = np.array([l])
        def tds_batch(tc_arr):
            r = simulate_batch(model39, np.repeat(bb, len(tc_arr)),
                               np.asarray(tc_arr, dtype=float), np.repeat(ll, len(tc_arr)),
                               4.0, dt=0.0005)
            return np.asarray(r["stable"], dtype=bool)
        def op_batch(tc_arr):
            with torch.no_grad():
                tr = m1(enc1(np.repeat(bb, len(tc_arr)), np.repeat(ll, len(tc_arr)),
                             np.asarray(tc_arr, dtype=float)), t1).cpu().numpy()
            return spread_max(tr) < np.pi
        rc = first_crossing_vec(tds_batch, 0.01, 0.8)
        oc = first_crossing_vec(op_batch, 0.01, 0.8)
        if rc is None or oc is None:
            continue
        errs.append(abs(oc - rc) * 1000)
    print(f"N-1 {split}: CCT |err| mean={np.mean(errs):.1f} ms, max={np.max(errs):.1f} ms "
          f"(n={len(errs)})", flush=True)

# ---------------- (2) N-2 (200 sampled combos) ----------------
import generate_data_n2 as g
dn2 = np.load("data39_n2.npz")
n2 = int(dn2["n_machines"]); te2 = float(dn2["t_end"]); tm2 = float(dn2["t_max"])
t2 = torch.as_tensor(dn2["t_out"], **F32)[:, None]
Pm2, M2, pf2 = dn2["Pm"], dn2["M"], dn2["pe_fault"]
m2 = DeepONet(2 * n2 + 1, n2, p=512, t_scale=te2).to(dev)
m2.load_state_dict(torch.load("model_n2_p512.pt")); m2.eval()

def enc2(bus_arr, sev_arr, tc_arr):
    sf = (Pm2[None, :] - pf2[np.asarray(bus_arr) - 1]) / M2[None, :] * 10.0
    tcn = np.asarray(tc_arr, dtype=np.float32)[:, None] / tm2
    return torch.cat([torch.as_tensor(sf, **F32),
                      torch.as_tensor(np.asarray(sev_arr, dtype=np.float32), **F32),
                      torch.as_tensor(tcn, **F32)], dim=1)

rng2 = np.random.default_rng(g.SEED)
all_buses = np.arange(1, 40); rng2.shuffle(all_buses)
all_lines = np.arange(len(model39.net.line)); rng2.shuffle(all_lines)
pairs = [(a, b) for i, a in enumerate(all_lines) for b in all_lines[i + 1:]]
rng2.shuffle(pairs)
test_pairs = pairs[:150]
sev_to_pair = {}
for (a, b) in test_pairs:
    pe = model39._pe(model39.delta0, model39.post_yint_multi([a, b]))
    sv = (model39.Pm - pe) / model39.M * 10.0
    sev_to_pair[tuple(np.round(sv, 3))] = (a, b)

fb2 = dn2["fb_testn2"]; sev2 = dn2["sev_testn2"]; tc2 = dn2["tc_testn2"]
combos2 = list(dict.fromkeys(zip(fb2.tolist(), [tuple(s) for s in sev2.tolist()])))
rng_s = np.random.default_rng(123)
rng_s.shuffle(combos2)
combos2 = combos2[:200]

errs2 = []
for k, (bus, sev) in enumerate(combos2):
    trips = sev_to_pair[tuple(np.round(np.array(sev), 3))]
    bb = np.array([bus]); svv = np.array([sev])
    def tds_batch(tc_arr):
        r = simulate_batch_multi(model39, np.repeat(bb, len(tc_arr)),
                                 [list(trips)] * len(tc_arr),
                                 np.asarray(tc_arr, dtype=float), te2, dt=0.001, out_dt=0.01)
        return np.asarray(r["stable"], dtype=bool)
    def op_batch(tc_arr):
        with torch.no_grad():
            tr = m2(enc2(np.repeat(bb, len(tc_arr)),
                         np.repeat(svv, len(tc_arr), axis=0),
                         np.asarray(tc_arr, dtype=float)), t2).cpu().numpy()
        return spread_max(tr) < np.pi
    rc = first_crossing_vec(tds_batch, 0.01, 0.8)
    oc = first_crossing_vec(op_batch, 0.01, 0.8)
    if rc is None or oc is None:
        continue
    errs2.append(abs(oc - rc) * 1000)
    if (k + 1) % 50 == 0:
        print(f"N-2 progress {k+1}/{len(combos2)} ...", flush=True)
print(f"N-2 pairs: CCT |err| mean={np.mean(errs2):.1f} ms, max={np.max(errs2):.1f} ms "
      f"(n={len(errs2)})", flush=True)

# ---------------- (3) 118-bus ----------------
import generate_118_authoritative as ga
d118 = np.load("data118.npz")
n118 = int(d118["n_machines"]); te118 = float(d118["t_end"]); tm118 = float(d118["t_max"])
t118 = torch.as_tensor(d118["t_out"], **F32)[:, None]
m118 = DeepONet(n118 + 1, n118, p=512, t_scale=te118).to(dev)
m118.load_state_dict(torch.load("model_118.pt")); m118.eval()
model118 = ClassicalModel(case="118", gendyn=ga.build_gendyn_118())
Pm118, M118, pf118 = d118["Pm"], d118["M"], d118["pe_fault"]

buses118 = np.unique(d118["fb_test"])
errs118 = []
for k, b in enumerate(buses118):
    bb = np.array([b])
    def tds_batch(tc_arr):
        r = simulate_batch(model118, np.repeat(bb, len(tc_arr)),
                           np.asarray(tc_arr, dtype=float), np.repeat(-1, len(tc_arr)),
                           te118, dt=0.001, out_dt=0.01)
        return np.asarray(r["stable"], dtype=bool)
    def op_batch(tc_arr):
        sf = (Pm118[None, :] - pf118[bb - 1]) / M118[None, :] * 10.0
        tcn = np.asarray(tc_arr, dtype=np.float32)[:, None] / tm118
        s = torch.cat([torch.as_tensor(np.repeat(sf, len(tc_arr), axis=0), **F32),
                       torch.as_tensor(tcn, **F32)], dim=1)
        with torch.no_grad():
            tr = m118(s, t118).cpu().numpy()
        return spread_max(tr) < np.pi
    rc = first_crossing_vec(tds_batch, 0.005, 0.2)
    oc = first_crossing_vec(op_batch, 0.005, 0.2)
    if rc is None or oc is None:
        continue
    errs118.append(abs(oc - rc) * 1000)
    print(f"118 progress {k+1}/{len(buses118)} (bus {int(b)})", flush=True)
print(f"118-bus: CCT |err| mean={np.mean(errs118):.1f} ms, max={np.max(errs118):.1f} ms "
      f"(n={len(errs118)})", flush=True)
print("ALL DONE", flush=True)
