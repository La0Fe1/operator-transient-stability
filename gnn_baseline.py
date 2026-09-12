"""GNN (graph convolutional network) classifier baseline for TSA.

A standard graph-based classifier: node features (voltage, load, fault flag) are
processed by graph convolution over the network topology, pooled, and mapped to a
stable/unstable label. This is a strong classification-only baseline against which
the operator's trajectory + CCT + certification is compared.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandapower.networks as pn


def build_graph():
    net = pn.case39()
    n = len(net.bus)
    A = np.eye(n)
    for _, ln in net.line.iterrows():
        i, j = int(ln["from_bus"]), int(ln["to_bus"])
        A[i, j] = 1; A[j, i] = 1
    for _, tr in net.trafo.iterrows():
        i, j = int(tr["hv_bus"]), int(tr["lv_bus"])
        A[i, j] = 1; A[j, i] = 1
    # normalized adjacency (with self-loops)
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


class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=32):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden)
        self.W2 = nn.Linear(hidden, out_dim)
        self.head = nn.Linear(out_dim + 1, 1)   # +1 for t_clear

    def forward(self, X, A, tc):
        H = F.relu(A @ self.W1(X))     # (B, n, hidden)
        H = A @ self.W2(H)             # (B, n, out_dim)
        h = H.mean(dim=1)              # (B, out_dim) mean pooling
        z = torch.cat([h, tc], dim=1)  # (B, out_dim+1)
        return self.head(z).squeeze(-1)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = pn.case39()
    import pandapower as pp
    pp.runpp(net)
    A = build_graph().to(dev)

    d = np.load("data39.npz")
    fb_tr, tc_tr, s_tr = d["fb_train"], d["tc_train"], d["s_train"].astype(np.float32)
    fb_te, tc_te, s_te = d["fb_test"], d["tc_test"], d["s_test"].astype(np.float32)

    X_tr = torch.as_tensor(node_features(net, fb_tr), dtype=torch.float32, device=dev)
    X_te = torch.as_tensor(node_features(net, fb_te), dtype=torch.float32, device=dev)
    tc_tr = torch.as_tensor(tc_tr / 0.6, dtype=torch.float32, device=dev)[:, None]
    tc_te = torch.as_tensor(tc_te / 0.6, dtype=torch.float32, device=dev)[:, None]
    y_tr = torch.as_tensor(s_tr, dtype=torch.float32, device=dev)
    y_te = torch.as_tensor(s_te, dtype=torch.float32, device=dev)

    torch.manual_seed(0); np.random.seed(0)
    model = GCNClassifier(6).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B = len(y_tr); nb = max(1, B // 256)
    for ep in range(300):
        model.train()
        perm = torch.randperm(B, device=dev)
        for it in range(nb):
            idx = perm[it * 256:(it + 1) * 256]
            logit = model(X_tr[idx], A, tc_tr[idx])
            loss = F.binary_cross_entropy_with_logits(logit, y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred = (model(X_te, A, tc_te) > 0).float().cpu().numpy()
    acc = (pred == s_te).mean()
    print(f"GNN classifier (classification only): accuracy = {acc:.3f}")
    print(f"(operator classification accuracy = 0.967, plus trajectory + CCT + certification)")


if __name__ == "__main__":
    main()
