# -*- coding: utf-8 -*-
"""C3 baselines: a plain MLP that receives exactly the same severity encoding as the
DeepONet operator. Two heads: (a) stability classifier, (b) per-bus CCT regressor.
Both are trained on the training fault buses and evaluated on the held-out buses,
mirroring the operator's protocol."""
import numpy as np
import torch

from power_system import ClassicalModel
from dataset import simulate_batch
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)
T_END = 4.0
DT_REF = 0.0005


def true_cct(bus, model):
    lo, hi = 0.01, 0.8
    for _ in range(16):
        mid = (lo + hi) / 2
        r = simulate_batch(model, np.array([bus]), np.array([mid]), np.array([-1]), T_END,
                           dt=DT_REF)
        lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
    return lo


def train_severity_mlp(dev):
    d = np.load("data39.npz")
    n = int(d["n_machines"])
    fb_tr, tc_tr, st_tr = d["fb_train"], d["tc_train"], d["s_train"].astype(bool)
    fb_te, tc_te, st_te = d["fb_test"], d["tc_test"], d["s_test"].astype(bool)
    s_tr = severity_enc(fb_tr, tc_tr, d, dev).float()
    s_te = severity_enc(fb_te, tc_te, d, dev).float()

    # ---- (a) classifier ----
    torch.manual_seed(0)
    clf = torch.nn.Sequential(
        torch.nn.Linear(n + 1, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    y_tr = torch.as_tensor(st_tr, **F32, device=dev)[:, None]
    B = 256
    for ep in range(100):
        clf.train()
        perm = torch.randperm(len(s_tr))
        for i in range(0, len(s_tr), B):
            idx = perm[i:i+B]
            opt.zero_grad()
            loss = lossf(clf(s_tr[idx]), y_tr[idx])
            loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        p_te = (clf(s_te) > 0).cpu().numpy()[:, 0]
    acc = (p_te == st_te).mean()
    print(f"severity-MLP classifier: acc={acc:.4f}", flush=True)

    # ---- (b) per-bus CCT regressor ----
    model39 = ClassicalModel(case="39")
    all_buses = np.arange(1, 40)
    cct_map = {int(b): true_cct(int(b), model39) for b in all_buses}
    y_cct_tr = torch.as_tensor(np.array([cct_map[int(b)] for b in fb_tr])[:, None],
                               **F32, device=dev)
    y_cct_te = np.array([cct_map[int(b)] for b in fb_te])
    torch.manual_seed(0)
    reg = torch.nn.Sequential(
        torch.nn.Linear(n + 1, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(reg.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    for ep in range(100):
        reg.train()
        perm = torch.randperm(len(s_tr))
        for i in range(0, len(s_tr), B):
            idx = perm[i:i+B]
            opt.zero_grad()
            loss = lossf(reg(s_tr[idx]), y_cct_tr[idx])
            loss.backward(); opt.step()
    reg.eval()
    with torch.no_grad():
        pred_cct_te = reg(s_te).cpu().numpy()[:, 0]
    # per-bus mean prediction vs per-bus true CCT (same protocol as the GCN-CCT baseline)
    errs = []
    for b in np.unique(fb_te):
        m_ = pred_cct_te[fb_te == b].mean()
        errs.append(abs(m_ - cct_map[int(b)]) * 1000)
    print(f"severity-MLP CCT regressor: mean |err|={np.mean(errs):.2f} ms "
          f"(max {np.max(errs):.1f})", flush=True)
    return dict(acc=float(acc), cct_mean=float(np.mean(errs)), cct_max=float(np.max(errs)))


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_severity_mlp(dev)
