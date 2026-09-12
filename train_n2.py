"""Train the operator on N-0 + N-1 + N-2 data and evaluate topology generalization."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from model import DeepONet

CLIP = 3.0 * np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p", type=int, default=128)
    ap.add_argument("--tag", type=str, default="n2")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39_n2.npz")
    n = int(d["n_machines"]); t_end = float(d["t_end"]); t_max = float(d["t_max"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    Pm = d["Pm"]; M = d["M"]; pf = d["pe_fault"]

    def enc(fb, sev, tc):
        sf = (Pm[None, :] - pf[fb - 1]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / t_max
        return torch.cat([torch.as_tensor(sf, dtype=torch.float32, device=dev),
                          torch.as_tensor(sev, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    s_tr = enc(d["fb_train"], d["sev_train"], d["tc_train"])
    s_t2 = enc(d["fb_testn2"], d["sev_testn2"], d["tc_testn2"])
    s_tb = enc(d["fb_testb"], d["sev_testb"], d["tc_testb"])
    y_tr = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_t2 = torch.as_tensor(d["y_testn2"], dtype=torch.float32, device=dev)
    y_tb = torch.as_tensor(d["y_testb"], dtype=torch.float32, device=dev)
    st_tr = torch.as_tensor(d["s_train"].astype(bool), device=dev)

    model = DeepONet(s_tr.shape[1], n, p=args.p, t_scale=t_end).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    B = y_tr.shape[0]; nb = max(1, B // args.batch_size)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(B, device=dev)
        for it in range(nb):
            idx = perm[it * args.batch_size:(it + 1) * args.batch_size]
            s = s_tr[idx]; y = y_tr[idx]; st = st_tr[idx]
            delta = model(s, t_out)
            target = torch.clamp(y, -CLIP, CLIP)
            se = ((delta - target) ** 2).mean(dim=(1, 2))
            n_st = st.sum().clamp(min=1); n_un = (~st).sum().clamp(min=1)
            loss = 0.5 * (se[st].sum() / n_st + se[~st].sum() / n_un)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        if ep % 100 == 0:
            print(f"ep {ep}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), f"model_{args.tag}.pt")
    model.eval()
    def report(s, y, st, name):
        with torch.no_grad():
            pred = model(s, t_out).cpu().numpy()
        y = y.cpu().numpy(); st = st.astype(bool)
        err = np.abs(pred - y)
        rmse = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
        fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
        acc = ((fs < np.pi) == st).mean()
        print(f"{name:10s}: stable_rmse={rmse:6.2f} deg | clf_acc={acc:.3f} (n={st.sum()})")

    report(s_tb, y_tb, d["s_testb"], "test_bus_N0")
    report(s_t2, y_t2, d["s_testn2"], "test_N2_pairs")


if __name__ == "__main__":
    main()
