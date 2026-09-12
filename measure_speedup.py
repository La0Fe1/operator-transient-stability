"""Like-for-like speedup measurement: operator vs RK4 on the same hardware.

The headline 578x compares GPU operator inference against CPU RK4, which mixes
hardware. Here we measure four quantities per scenario (batch of B=64):
  - t_op_gpu : operator inference on GPU (RTX 4070)
  - t_op_cpu : operator inference on CPU
  - t_rk_cpu : vectorized RK4 time-domain simulation on CPU (dt=0.5 ms, 4 s)
so that both a GPU-accelerated and a CPU-only (like-for-like) speedup are reported.
"""
import time
import numpy as np
import torch

from model import DeepONet
from power_system import ClassicalModel
from dataset import simulate_batch
from evaluate import severity_enc, predict

B = 64
REPS = 20

d = np.load("data39.npz")
n = int(d["n_machines"])
t_end = float(d["t_end"])
fb = d["fb_test"][:B]
tc = d["tc_test"][:B]

# ---- operator on GPU ----
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
m = DeepONet(n + 1, n, p=512, t_scale=t_end).to(dev)
m.load_state_dict(torch.load("model_p512.pt")); m.eval()
s = severity_enc(fb, tc, d, dev)
with torch.no_grad():
    for _ in range(3):                      # warmup
        predict(m, s, t, dev)
    t0 = time.time()
    for _ in range(REPS):
        predict(m, s, t, dev)
    t_op_gpu = (time.time() - t0) / REPS / B

# ---- operator on CPU ----
mcpu = DeepONet(n + 1, n, p=512, t_scale=t_end).to("cpu")
mcpu.load_state_dict(torch.load("model_p512.pt")); mcpu.eval()
tcpu = torch.as_tensor(d["t_out"], dtype=torch.float32)[:, None]
scpu = severity_enc(fb, tc, d, "cpu")
with torch.no_grad():
    for _ in range(3):                      # warmup
        predict(mcpu, scpu, tcpu, "cpu")
    t0 = time.time()
    for _ in range(REPS):
        predict(mcpu, scpu, tcpu, "cpu")
    t_op_cpu = (time.time() - t0) / REPS / B

# ---- RK4 on CPU (dt=0.5 ms, 4 s horizon) ----
model_cct = ClassicalModel(case="39")
simulate_batch(model_cct, fb[:4], tc[:4], np.full(4, -1), 4.0, dt=0.0005)  # warmup
t0 = time.time()
simulate_batch(model_cct, fb, tc, np.full(B, -1), 4.0, dt=0.0005)
t_rk_cpu = (time.time() - t0) / B

print("=" * 70)
print(f"device used for 'GPU' operator: {dev}")
print(f"per-scenario operator inference, GPU : {t_op_gpu*1e3:8.3f} ms")
print(f"per-scenario operator inference, CPU : {t_op_cpu*1e3:8.3f} ms")
print(f"per-scenario RK4 TDS, CPU (dt=0.5ms) : {t_rk_cpu*1e3:8.3f} ms")
print("-" * 70)
print(f"speedup (GPU operator vs CPU RK4)    : {t_rk_cpu/t_op_gpu:8.1f}x")
print(f"speedup (CPU operator vs CPU RK4)    : {t_rk_cpu/t_op_cpu:8.1f}x")
print("=" * 70)
