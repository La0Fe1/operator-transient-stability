"""Stronger GNN baseline: three graph layers + concatenated mean/max pooling.

Addresses the concern that the paper's two-layer GCN with mean pooling (76.0%)
is a weak graph baseline. This variant uses three graph-convolution layers with
residual connections and concatenated mean/max/sum readouts on the same raw node
features, so the comparison remains feature-fair.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandapower.networks as pn
import pandapower as pp


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


class GCNStrong(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.W3 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(3 * hidden + 1, 1)   # mean+max+sum pooling + t_clear

    def forward(self, X, A, tc):
        H = F.relu(A @ self.W1(X))
        H = H + F.relu(A @ self.W2(H))             # residual
        H = H + F.relu(A @ self.W3(H))             # residual
        h = torch.cat([H.mean(dim=1), H.max(dim=1).values, H.sum(dim=1)], dim=1)
        z = torch.cat([h, tc], dim=1)
        return self.head(z).squeeze(-1)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = pn.case39()
    pp.runpp(net)
    A = build_graph(net).to(dev)

    d = np.load("data39.npz")
    fb_tr, tc_tr, s_tr = d["fb_train"], d["tc_train"], d["s_train"].astype(np.float32)
    fb_te, tc_te, s_te = d["fb_test"], d["tc_test"], d["s_test"].astype(np.float32)

    X_tr = torch.as_tensor(node_features(net, fb_tr), dtype=torch.float32, device=dev)
    X_te = torch.as_tensor(node_features(net, fb_te), dtype=torch.float32, device=dev)
    tc_tr = torch.as_tensor(tc_tr / 0.6, dtype=torch.float32, device=dev)[:, None]
    tc_te = torch.as_tensor(tc_te / 0.6, dtype=torch.float32, device=dev)[:, None]
    y_tr = torch.as_tensor(s_tr, dtype=torch.float32, device=dev)

    results = []
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        model = GCNStrong(6).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
        results.append(acc)
        print(f"seed {seed}: stronger GNN accuracy={acc:.3f}", flush=True)
    print(f"stronger GNN over 5 seeds: {np.mean(results):.3f} +/- {np.std(results):.3f}")
    print("(two-layer GCN: 76.0%; severity-encoded operator: 96.7%)")


if __name__ == "__main__":
    main()
