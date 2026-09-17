# -*- coding: utf-8 -*-
"""B4 protocol fix for N-1 / N-2 / 118-bus CCTs: first stable->unstable crossing
via 5 ms sweep + local bisection to 0.1 ms, max-over-time criterion throughout
(fixes the old last-timestep `tr[:, :, -1]` bug in compute_round8/compute_extra_metrics)."""
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

def first_crossing(fn, lo, hi, sweep=0.005, tol=0.0001):
    grid = np.arange(lo, hi + 1e-12, sweep)
    prev = fn(grid[0])
    if not prev:
        return None  # starts unstable; no defined first crossing
    for tc in grid[1:]:
        cur = fn(tc)
        if prev and not cur:
            a, b = tc - sweep, tc
            for _ in range(int(np.ceil(np.log2(sweep / tol)))):
                mid = (a + b) / 2
                if fn(mid):
                    a = mid
                else:
                    b = mid
            return b
        prev = cur
    return None  # never crosses

model39 = ClassicalModel(case="39")

def tds39(bus, line, tc, dt=0.0005):
    r = simulate_batch(model39, np.array([bus]), np.array([tc]), np.array([line]), 4.0, dt=dt)
    return bool(r["stable"][0])

# ---------------- (1) N-1: held-out trip lines + held-out buses ----------------
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
    return torch.cat([torch.as_tensor(sf, **F32),
                      torch.as_tensor(sp, **F32),
                      torch.as_tensor(tcn, **F32)], dim=1)

def op_n1(bus, line, tc):
    with torch.no_grad():
        tr = m1(enc1(np.array([bus]), np.array([line]), np.array([tc])), t1).cpu().numpy()
    return bool((tr[0].max(axis=0) - tr[0].min(axis=0)).max() < np.pi)

for split in ["line", "bus"]:
    if split == "bus":
        fb, tl = dn["fb_testb"], dn["tl_testb"]
    else:
        fb, tl = dn["fb_testl"], dn["tl_testl"]
    combos = list(dict.fromkeys(zip(fb.tolist(), tl.tolist())))
    errs = []
    for b, l in combos:
        rc = first_crossing(lambda tc: tds39(int(b), int(l), tc), 0.01, 0.8)
        oc = first_crossing(lambda tc: op_n1(int(b), int(l), tc), 0.01, 0.8)
        if rc is None or oc is None:
            continue
        errs.append(abs(oc - rc) * 1000)
    print(f"N-1 {split}: CCT |err| mean={np.mean(errs):.1f} ms, max={np.max(errs):.1f} ms "
          f"(n={len(errs)})", flush=True)

# ---------------- (2) N-2: held-out line pairs ----------------
import generate_data_n2 as g
dn2 = np.load("data39_n2.npz")
n2 = int(dn2["n_machines"]); te2 = float(dn2["t_end"]); tm2 = float(dn2["t_max"])
t2 = torch.as_tensor(dn2["t_out"], **F32)[:, None]
Pm2, M2, pf2 = dn2["Pm"], dn2["M"], dn2["pe_fault"]
m2 = DeepONet(2 * n2 + 1, n2, p=512, t_scale=te2).to(dev)
m2.load_state_dict(torch.load("model_n2_p512.pt")); m2.eval()

def enc2(bus, sev, tcc):
    sf = (Pm2[None, :] - pf2[bus - 1]) / M2[None, :] * 10.0
    tcn = np.asarray(tcc, dtype=np.float32)[:, None] / tm2
    return torch.cat([torch.as_tensor(sf, **F32),
                      torch.as_tensor(np.asarray(sev, dtype=np.float32), **F32),
                      torch.as_tensor(tcn, **F32)], dim=1)

def op_n2(bus, sev, tcc):
    with torch.no_grad():
        tr = m2(enc2(bus, sev, tcc), t2).cpu().numpy()
    return bool((tr[0].max(axis=0) - tr[0].min(axis=0)).max() < np.pi)

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

def tds_n2(bus, sev, tcc):
    trips = sev_to_pair[tuple(np.round(np.array(sev), 3))]
    r = simulate_batch_multi(model39, np.array([bus]), [list(trips)], np.array([tcc]),
                             te2, dt=0.001, out_dt=0.01)
    return bool(r["stable"][0])

errs2 = []
for k, (bus, sev) in enumerate(combos2):
    rc = first_crossing(lambda tc: tds_n2(int(bus), np.array(sev), tc), 0.01, 0.8)
    oc = first_crossing(lambda tc: op_n2(int(bus), np.array(sev), tc), 0.01, 0.8)
    if rc is None or oc is None:
        continue
    errs2.append(abs(oc - rc) * 1000)
    if (k + 1) % 100 == 0:
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

def enc118(bus, tcc):
    sf = (Pm118[None, :] - pf118[bus - 1]) / M118[None, :] * 10.0
    tcn = np.asarray(tcc, dtype=np.float32)[:, None] / tm118
    return torch.cat([torch.as_tensor(sf, **F32),
                      torch.as_tensor(tcn, **F32)], dim=1)

def tds118(bus, tcc):
    r = simulate_batch(model118, np.array([bus]), np.array([tcc]), np.array([-1]), te118,
                       dt=0.001, out_dt=0.01)
    return bool(r["stable"][0])

def op118(bus, tcc):
    with torch.no_grad():
        tr = m118(enc118(bus, tcc), t118).cpu().numpy()
    return bool((tr[0].max(axis=0) - tr[0].min(axis=0)).max() < np.pi)

buses118 = np.unique(d118["fb_test"])
errs118 = []
for k, b in enumerate(buses118):
    rc = first_crossing(lambda tc: tds118(int(b), tc), 0.005, 0.2)
    oc = first_crossing(lambda tc: op118(int(b), tc), 0.005, 0.2)
    if rc is None or oc is None:
        continue
    errs118.append(abs(oc - rc) * 1000)
    print(f"118 progress {k+1}/{len(buses118)} (bus {int(b)})", flush=True)
print(f"118-bus: CCT |err| mean={np.mean(errs118):.1f} ms, max={np.max(errs118):.1f} ms "
      f"(n={len(errs118)})", flush=True)
print("ALL DONE")
