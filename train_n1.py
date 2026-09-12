"""Train the operator for N-0 + N-1 (line-trip) transient stability, with the
physics-informed encoding extended by the post-fault severity."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from model import DeepONet

CLIP = 3.0 * np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n1")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load("data39_n1.npz")
    n = int(d["n_machines"]); t_end = float(d["t_end"]); t_max = float(d["t_max"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    Pm = d["Pm"]; M = d["M"]; pe_fault = d["pe_fault"]; pe_post = d["pe_post"]

    def enc(fb, tl, tc):
        sev_f = (Pm[None, :] - pe_fault[fb - 1]) / M[None, :] * 10.0
        sev_p = np.zeros((len(fb), n))
        valid = tl >= 0
        if valid.any():
            sev_p[valid] = (Pm[None, :] - pe_post[tl[valid]]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / t_max
        return torch.cat([torch.as_tensor(sev_f, dtype=torch.float32, device=dev),
                          torch.as_tensor(sev_p, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    s_tr = enc(d["fb_train"], d["tl_train"], d["tc_train"])
    s_va = enc(d["fb_val"], d["tl_val"], d["tc_val"])
    s_tb = enc(d["fb_testb"], d["tl_testb"], d["tc_testb"])
    s_tl = enc(d["fb_testl"], d["tl_testl"], d["tc_testl"])
    y_tr = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_va = torch.as_tensor(d["y_val"], dtype=torch.float32, device=dev)
    y_tb = torch.as_tensor(d["y_testb"], dtype=torch.float32, device=dev)
    y_tl = torch.as_tensor(d["y_testl"], dtype=torch.float32, device=dev)
    st_tr = torch.as_tensor(d["s_train"].astype(bool), device=dev)

    model = DeepONet(s_tr.shape[1], n, p=args.p, t_scale=t_end).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    B = y_tr.shape[0]; nb = max(1, B // args.batch_size)
    best = float("inf"); t0 = time.time()
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
        with torch.no_grad():
            mv = ((model(s_va, t_out) - torch.clamp(y_va, -CLIP, CLIP)) ** 2).mean().item()
        if mv < best:
            best = mv; torch.save(model.state_dict(), f"model_{args.tag}.pt")
        if ep % 100 == 0:
            print(f"ep {ep}  val {mv:.4e}  ({time.time()-t0:.0f}s)", flush=True)

    # evaluation
    torch.save(model.state_dict(), f"model_{args.tag}.pt")
    model.eval()

    def report(s, y, st, name):
        with torch.no_grad():
            pred = model(s, t_out).cpu().numpy()
        y = y.cpu().numpy()
        err = np.abs(pred - y)
        st = st.astype(bool)
        rmse = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
        fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
        acc = ((fs < np.pi) == st).mean()
        print(f"{name:10s}: stable_rmse={rmse:6.2f} deg | clf_acc={acc:.3f} (n_stable={st.sum()})")
        return rmse, acc

    report(s_tb, y_tb, d["s_testb"], "test_bus")
    report(s_tl, y_tl, d["s_testl"], "test_line")


if __name__ == "__main__":
    main()
