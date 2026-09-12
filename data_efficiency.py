"""Data-efficiency comparison: DeepONet vs pointwise MLP at varying training size.

The operator's low-rank (branch x trunk) structure should be more
sample-efficient than a pointwise MLP. We train both on 10/25/50/100% of the
training scenarios and compare held-out stable-trajectory RMSE.
"""
from __future__ import annotations

import numpy as np
import torch

from model import DeepONet
from baseline_mlp import PointMLP
from evaluate import severity_enc

CLIP = 3.0 * np.pi
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
EPOCHS = 400


def train_and_eval(ModelClass, s_full, y_full, st_full, s_test, y_test, st_test,
                   t_out, dev, frac, n, seed, **kw):
    torch.manual_seed(seed); np.random.seed(seed)
    B = len(s_full)
    rng = np.random.default_rng(seed)
    # stratified subsample: keep the stable/unstable ratio
    st_b = st_full.cpu().numpy() if isinstance(st_full, torch.Tensor) else st_full
    idx_st = np.where(st_b)[0]; idx_un = np.where(~st_b)[0]
    k_st = int(frac * len(idx_st)); k_un = int(frac * len(idx_un))
    idx = np.concatenate([rng.choice(idx_st, max(1, k_st), replace=False),
                          rng.choice(idx_un, max(1, k_un), replace=False)])

    s = s_full[idx]; y = y_full[idx]; st = st_full[idx]
    model = ModelClass(s_full.shape[1], n, t_scale=float(4.0), **kw).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    nb = max(1, len(idx) // 64)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(idx), device=dev)
        for it in range(nb):
            bi = perm[it * 64:(it + 1) * 64]
            pred = model(s[bi], t_out)
            target = torch.clamp(y[bi], -CLIP, CLIP)
            se = ((pred - target) ** 2).mean(dim=(1, 2))
            m = st[bi]
            n_st = m.sum().clamp(min=1); n_un = (~m).sum().clamp(min=1)
            loss = 0.5 * (se[m].sum() / n_st + se[~m].sum() / n_un)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(s_test, t_out).cpu().numpy()
    err = np.abs(pred - y_test.cpu().numpy())
    rmse = np.rad2deg(np.sqrt((err[st_test.cpu().numpy().astype(bool)] ** 2).mean()))
    return rmse


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); t_end = float(d["t_end"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]

    s_full = severity_enc(d["fb_train"], d["tc_train"], d, dev)
    s_test = severity_enc(d["fb_test"], d["tc_test"], d, dev)
    y_full = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_test = torch.as_tensor(d["y_test"], dtype=torch.float32, device=dev)
    st_full = torch.as_tensor(d["s_train"].astype(bool), device=dev)
    st_test = torch.as_tensor(d["s_test"].astype(bool), device=dev)

    print(f"{'frac':>6} | {'DeepONet (mean±std)':>20} | {'MLP (mean±std)':>20}")
    for frac in FRACTIONS:
        r_op = [train_and_eval(DeepONet, s_full, y_full, st_full, s_test, y_test, st_test,
                               t_out, dev, frac, n, seed=s, p=128) for s in (0, 1, 2)]
        r_mlp = [train_and_eval(PointMLP, s_full, y_full, st_full, s_test, y_test, st_test,
                                t_out, dev, frac, n, seed=s) for s in (0, 1, 2)]
        print(f"{frac*100:5.0f}% | {np.mean(r_op):6.2f}±{np.std(r_op):4.2f} deg | "
              f"{np.mean(r_mlp):6.2f}±{np.std(r_mlp):4.2f} deg")


if __name__ == "__main__":
    main()
