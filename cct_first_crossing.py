# -*- coding: utf-8 -*-
"""B4 fix: locate the FIRST stable->unstable crossing (scan + local refine) for the
CCT of both the TDS reference and the operator on the 12 held-out fault buses.

Protocol: coarse sweep at 5 ms over [0.01, 0.8] s to find the first sign change,
then local bisection to 0.1 ms. Stability = max-over-time pairwise spread < pi on
the 0.01 s output grid (same criterion as the manuscript labels).
"""
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch
from evaluate import severity_enc, predict

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = np.load("data39.npz")
n = int(d["n_machines"]); t_end = float(d["t_end"]); t_out = d["t_out"]
t = torch.as_tensor(t_out, dtype=torch.float32, device=dev)[:, None]
m = DeepONet(n + 1, n, p=512, t_scale=t_end).to(dev)
m.load_state_dict(torch.load("model_p512.pt")); m.eval()

sys_model = ClassicalModel(case="39")

def tds_stable(bus, tc):
    r = simulate_batch(sys_model, np.array([bus]), np.array([tc]), np.array([-1]),
                       4.0, dt=0.0005)
    return bool(r["stable"][0])

def op_stable(bus, tc):
    with torch.no_grad():
        tr = predict(m, severity_enc(np.array([bus]), np.array([tc]), d, dev), t, dev)[0]
    return bool((tr.max(axis=0) - tr.min(axis=0)).max() < np.pi)

def first_crossing(fn, lo=0.01, hi=0.8, sweep=0.005, tol=0.0001):
    """First tc where fn(tc) goes stable->unstable, via sweep + local bisection."""
    grid = np.arange(lo, hi + 1e-12, sweep)
    labels = [fn(grid[0])]
    first_unstable = None
    for tc in grid[1:]:
        labels.append(fn(tc))
        if labels[-2] and not labels[-1]:
            first_unstable = tc
            break
    if first_unstable is None:
        return None, labels  # no crossing found (always stable or starts unstable)
    last_stable = first_unstable - sweep
    # local bisection between last_stable and first_unstable
    lo_, hi_ = last_stable, first_unstable
    for _ in range(int(np.ceil(np.log2(sweep / tol)))):
        mid = (lo_ + hi_) / 2
        if fn(mid):
            lo_ = mid
        else:
            hi_ = mid
    return hi_, labels

buses = sorted(np.unique(d["fb_test"]))
print("bus | ref 1st-crossing (ms) | op 1st-crossing (ms) | signed err (ms)")
refs, ops, errs = [], [], []
for b in buses:
    rc, _ = first_crossing(lambda tc: tds_stable(b, tc))
    oc, _ = first_crossing(lambda tc: op_stable(b, tc))
    rc = None if rc is None else rc * 1000
    oc = None if oc is None else oc * 1000
    refs.append(rc); ops.append(oc)
    e = None if (rc is None or oc is None) else oc - rc
    errs.append(e)
    print(f"{int(b):3d} | {rc if rc is None else round(rc,1):>8} | "
          f"{oc if oc is None else round(oc,1):>8} | {e if e is None else round(e,1):>10}")
valid = [abs(e) for e in errs if e is not None]
print()
print(f"mean |err| = {np.mean(valid):.1f} ms   max |err| = {np.max(valid):.1f} ms")
print(f"mean signed = {np.mean(errs):.1f} ms")
print("current paper macros: refs=(218,211,185,208,453,217,196,190,197,183,234,129)")
print("                       ops =(211,195,174,201,248,206,195,203,195,186,244,111)")
print("                       errs=(-7,-16,-12,-7,-205,-11,-1,+13,-2,+3,+10,-18), MAE=25.2")
