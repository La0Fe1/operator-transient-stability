"""GNN (graph convolutional network) baseline that regresses the critical clearing
time (CCT) directly from network topology + a fault flag, to compare against the
operator's CCT estimation (12.3 ms mean error on 12 held-out fault buses).

The CCT of a fault is a property of the faulted bus (for N-0). The GCN maps
(pre-fault node features + one-hot fault flag) -> CCT, trained on the 27 training
fault buses and evaluated on the 12 held-out fault buses---the same generalization
task the operator faces. Unlike the operator's physics-informed severity encoding,
the GCN sees only raw observables (voltage, load, generator/fault flags).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandapower.networks as pn
import pandapower as pp

from power_system import ClassicalModel
from dataset import simulate_batch


def build_graph(net):
    n = len(net.bus)
    A = np.eye(n)
    for _, ln in net.line.iterrows():
        i, j = int(ln["from_bus"]), int(ln["to_bus"])
        A[i, j] = 1; A[j, i] = 1
    for _, tr in net.trafo.iterrows():
        i, j = int(tr["hv_bus"]), int(tr["lv_bus"])
        A[i, j] = 1; A[j, i] = 1
    D = A.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D)
    A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
    return torch.as_tensor(A_norm, dtype=torch.float32)


def node_features(net, fault_bus):
    """Node features: [V_m, V_a, P_load, Q_load, gen_flag, fault_flag]."""
    n = len(net.bus)
    res = net.res_bus
    X = np.zeros((len(fault_bus), n, 6), dtype=np.float32)
    for b in range(n):
        X[:, b, 0] = res["vm_pu"].iloc[b]
        X[:, b, 1] = res["va_degree"].iloc[b] / 180.0
    for _, ld in net.load.iterrows():
        b = int(ld["bus"])
        X[:, b, 2] = ld["p_mw"] / 100.0
        X[:, b, 3] = ld["q_mvar"] / 100.0
    for _, g in net.gen.iterrows():
        X[:, int(g["bus"]), 4] = 1.0
    for i, fb in enumerate(fault_bus):
        X[i, fb - 1, 5] = 1.0
    return X


class GCNCCT(nn.Module):
    """Two-layer GCN -> mean pool -> CCT (seconds). No clearing-time input."""
    def __init__(self, in_dim, hidden=64, out_dim=32):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden)
        self.W2 = nn.Linear(hidden, out_dim)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, X, A):
        H = F.relu(A @ self.W1(X))
        H = F.relu(A @ self.W2(H))
        h = H.mean(dim=1)
        return self.head(h).squeeze(-1)


def true_cct(bus):
    """Largest stable clearing time via TDS binary search (dt=1 ms, 4 s)."""
    lo, hi = 0.01, 0.8
    for _ in range(16):
        mid = (lo + hi) / 2
        r = simulate_batch(model_cct, np.array([bus]), np.array([mid]), np.array([-1]), 4.0)
        lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
    return lo


model_cct = ClassicalModel(case="39")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = pn.case39()
    pp.runpp(net)
    A = build_graph(net).to(dev)

    d = np.load("data39.npz")
    train_buses = np.unique(d["fb_train"])      # 27 buses
    test_buses = np.unique(d["fb_test"])        # 12 buses

    # true CCT targets (seconds) for all 39 buses
    cct_true = {int(b): true_cct(int(b)) for b in np.concatenate([train_buses, test_buses])}

    X_tr = torch.as_tensor(node_features(net, train_buses), dtype=torch.float32, device=dev)
    y_tr = torch.as_tensor([cct_true[int(b)] for b in train_buses], dtype=torch.float32, device=dev)
    X_te = torch.as_tensor(node_features(net, test_buses), dtype=torch.float32, device=dev)
    y_te = torch.as_tensor([cct_true[int(b)] for b in test_buses], dtype=torch.float32, device=dev)

    n_seeds = 5
    results = []
    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = GCNCCT(6).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(400):
            model.train()
            pred = model(X_tr, A)
            loss = F.mse_loss(pred, y_tr)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred_te = model(X_te, A).cpu().numpy()
        err = np.abs(pred_te - y_te.cpu().numpy()) * 1000.0   # ms
        results.append((err.mean(), err.max()))
        print(f"seed {seed}: GCN CCT error mean={err.mean():.1f} ms, max={err.max():.1f} ms")

    means = np.array([r[0] for r in results])
    maxs = np.array([r[1] for r in results])
    print("=" * 70)
    print(f"GCN CCT baseline over {n_seeds} seeds (27 train buses -> 12 held-out buses)")
    print(f"  mean CCT error: {means.mean():.1f} +/- {means.std():.1f} ms")
    print(f"  max  CCT error: {maxs.mean():.1f} ms")
    print(f"(operator: mean 12.3 ms, max 48.8 ms)")
    print("=" * 70)


if __name__ == "__main__":
    main()
