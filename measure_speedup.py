"""A7: speedup measurement with separated protocols (audit-corrected).

Throughput:  single-threaded vectorized batches of B=64, warm-up + repeats,
             median and P95 reported; torch/BLAS threads pinned to 1.
Latency B=1: single-scenario operator inference (sequential, no batching).
End-to-end:  one fault scenario including the severity encoding (Kron
             reduction) + the sequential 16-iteration CCT binary search, vs
             the same CCT search with TDS; also a 100-point sweep.

Run this with an idle CPU (no other heavy jobs).
"""
import time
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch
from evaluate import severity_enc, predict

torch.set_num_threads(1)          # pin BLAS/torch to a single thread
B = 64
REPS = 30
WARM = 5

d = np.load("data39.npz")
n = int(d["n_machines"])
t_end = float(d["t_end"])
fb = d["fb_test"][:B]
tc = d["tc_test"][:B]
t = torch.as_tensor(d["t_out"], dtype=torch.float32)[:, None]

m = DeepONet(n + 1, n, p=512, t_scale=t_end)
m.load_state_dict(torch.load("model_p512.pt")); m.eval()
s = severity_enc(fb, tc, d, "cpu")

model_cct = ClassicalModel(case="39")


def bench(fn, reps=REPS, warm=WARM):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3          # ms
    return float(np.median(ts)), float(np.percentile(ts, 95))


# ---- throughput, batched B=64 ----
with torch.no_grad():
    t_op_b64, _ = bench(lambda: predict(m, s, t, "cpu"))
t_rk_b64, _ = bench(lambda: simulate_batch(model_cct, fb, tc, np.full(B, -1), 4.0, dt=0.0005),
                    reps=7, warm=1)
t_op = t_op_b64 / B
t_rk = t_rk_b64 / B

# ---- latency, single scenario (B=1) ----
fb1, tc1 = fb[:1], tc[:1]
s1 = severity_enc(fb1, tc1, d, "cpu")
with torch.no_grad():
    t_op_b1, t_op_b1_p95 = bench(lambda: predict(m, s1, t, "cpu"))
t_rk_b1, t_rk_b1_p95 = bench(lambda: simulate_batch(model_cct, fb1, tc1, np.full(1, -1),
                                                    4.0, dt=0.0005), reps=7, warm=1)

# ---- end-to-end CCT search: operator (16 sequential inferences) + encoding ----
def cct_operator():
    enc = time.perf_counter()
    _ = severity_enc(fb1, tc1, d, "cpu")          # Kron reduction, per fault
    t_enc = (time.perf_counter() - enc) * 1e3
    lo, hi = 0.01, 0.8
    for _ in range(16):
        mid = (lo + hi) / 2
        s_mid = severity_enc(fb1, np.array([mid]), d, "cpu")
        with torch.no_grad():
            tr = predict(m, s_mid, t, "cpu")
        stab = (tr[0].max(axis=0) - tr[0].min(axis=0)).max() < np.pi
        lo, hi = (mid, hi) if stab else (lo, mid)
    return t_enc

t_cct_op, t_cct_op_p95 = bench(cct_operator, reps=15)

def cct_tds():
    lo, hi = 0.01, 0.8
    for _ in range(16):
        mid = (lo + hi) / 2
        r = simulate_batch(model_cct, fb1, np.array([mid]), np.full(1, -1), 4.0, dt=0.0005)
        lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)

t_cct_rk, _ = bench(cct_tds, reps=7, warm=1)

# ---- 100-point clearing-time sweep (encoding amortized) ----
def sweep100():
    tcs = np.linspace(0.01, 0.55, 100)
    fb_rep = np.repeat(fb1, 100)
    s_sw = severity_enc(fb_rep, tcs, d, "cpu")      # one encoding per distinct bus
    with torch.no_grad():
        predict(m, s_sw, t, "cpu")

t_sweep, _ = bench(sweep100, reps=10, warm=2)

print("=" * 78)
print(f"threads pinned: torch.set_num_threads(1)  |  repeats: {REPS} (+{WARM} warm-up), B={B}")
print(f"[throughput] operator  (B=64 batch, per scenario): {t_op:8.3f} ms")
print(f"[throughput] RK4 TDS   (B=64 batch, per scenario): {t_rk:8.3f} ms")
print(f"[throughput] speedup CPU operator vs CPU RK4     : {t_rk / t_op:8.1f}x")
print(f"[latency  ] operator B=1 : {t_op_b1:8.3f} ms (P95 {t_op_b1_p95:8.3f})")
print(f"[latency  ] RK4      B=1 : {t_rk_b1:8.3f} ms (P95 {t_rk_b1_p95:8.3f})")
print(f"[end-to-end] CCT search, operator+encoding (16 seq. inf): {t_cct_op:8.3f} ms (P95 {t_cct_op_p95:8.3f})")
print(f"[end-to-end] CCT search, TDS (16 seq. sims)             : {t_cct_rk:8.3f} ms")
print(f"[end-to-end] speedup (CCT, operator+encoding vs TDS)    : {t_cct_rk / t_cct_op:8.1f}x")
print(f"[sweep    ] 100-point tc sweep, operator+encoding       : {t_sweep:8.3f} ms")
print("=" * 78)
