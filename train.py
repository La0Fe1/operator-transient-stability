"""Train the physics-informed DeepONet for transient-stability trajectories."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from model import DeepONet, trunk_derivs, physics_residual

CLIP_DEFAULT = 3.0  # in units of pi; saturate the regression target beyond +/-540 deg


def to_torch(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


def evaluate(model, t, s, y_true, device, clip):
    model.eval()
    with torch.no_grad():
        pred = model(s, t)
        target = torch.clamp(y_true, -clip * np.pi, clip * np.pi)
        mse = ((pred - target) ** 2).mean().item()
        rmse = (pred - target).pow(2).mean(dim=(1, 2)).sqrt()
    return mse, rmse.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_phys", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p", type=int, default=128)
    ap.add_argument("--colloc_stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="dataonly")
    ap.add_argument("--data", type=str, default="data39.npz")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--clip", type=float, default=3.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  lambda_phys={args.lambda_phys}  p={args.p}  K={args.K}  clip={args.clip}", flush=True)

    d = np.load(args.data)
    t_out = to_torch(d["t_out"], device)[:, None]
    n_buses = int(d["n_buses"]); n = int(d["n_machines"])
    t_end = float(d["t_end"]); t_max = float(d["t_max"])

    E = to_torch(d["E"], device); Pm = to_torch(d["Pm"], device)
    M = to_torch(d["M"], device); G = to_torch(d["G"], device)
    Bsus = to_torch(d["B"], device); Mtot = float(d["Mtot"])

    # Physical, continuous fault encoding: the initial acceleration (severity)
    # of each machine, (Pm - Pe_fault)/M, which generalizes across fault buses.
    Pm_np = d["Pm"]; M_np = d["M"]; pe_fault = d["pe_fault"]

    def enc(fb, tc):
        pe = pe_fault[fb - 1]                                   # (B, n)
        sev = (Pm_np[None, :] - pe) / M_np[None, :] * 10.0      # (B, n) scaled
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / t_max  # (B, 1)
        return torch.cat([torch.as_tensor(sev, dtype=torch.float32, device=device),
                          torch.as_tensor(tcn, device=device)], dim=1)

    s_train = enc(d["fb_train"], d["tc_train"])
    s_val = enc(d["fb_val"], d["tc_val"])
    s_test = enc(d["fb_test"], d["tc_test"])
    y_train = to_torch(d["y_train"], device)
    y_val = to_torch(d["y_val"], device)
    y_test = to_torch(d["y_test"], device)
    tc_train = to_torch(d["tc_train"], device)
    st_train = torch.as_tensor(d["s_train"].astype(bool), device=device)  # (B,)

    model = DeepONet(s_train.shape[1], n, p=args.p, K=args.K, t_scale=t_end).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t_phys = t_out[:: args.colloc_stride]
    t_phys_col = t_phys[:, 0]

    B_tr = y_train.shape[0]
    n_batches = max(1, B_tr // args.batch_size)

    best_val = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(B_tr, device=device)
        total_data = 0.0; total_phys = 0.0
        for it in range(n_batches):
            idx = perm[it * args.batch_size:(it + 1) * args.batch_size]
            s = s_train[idx]; tc = tc_train[idx]; y = y_train[idx]
            st = st_train[idx]

            Bcoef = model.branch_coeff(s)
            tau = model.trunk_basis(t_out)
            delta = torch.einsum("bnp,tp->bnt", Bcoef, tau)

            # class-balanced MSE (stable vs unstable contribute equally)
            target = torch.clamp(y, -args.clip * np.pi, args.clip * np.pi)
            se = ((delta - target) ** 2).mean(dim=(1, 2))     # (b,)
            n_st = st.sum().clamp(min=1)
            n_un = (~st).sum().clamp(min=1)
            loss = 0.5 * (se[st].sum() / n_st + se[~st].sum() / n_un)
            total_data += loss.item()

            if args.lambda_phys > 0:
                tau_p, dtau_p, d2tau_p = trunk_derivs(model.trunk, t_phys, model.t_scale)
                dp = torch.einsum("bnp,tp->bnt", Bcoef, tau_p)
                d1 = torch.einsum("bnp,tp->bnt", Bcoef, dtau_p)
                d2 = torch.einsum("bnp,tp->bnt", Bcoef, d2tau_p)
                resid = physics_residual(dp, d1, d2, E, Pm, M, G, Bsus, Mtot) / M[None, :, None]
                mask = (t_phys_col[None, :] >= tc[:, None]).float().unsqueeze(1)
                lp = (resid.pow(2) * mask).sum() / (mask.sum() + 1e-8)
                loss = loss + args.lambda_phys * lp
                total_phys += lp.item()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()

        mse_val, _ = evaluate(model, t_out, s_val, y_val, device, args.clip)
        if mse_val < best_val:
            best_val = mse_val
            torch.save(model.state_dict(), f"model_{args.tag}.pt")

        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d}  data {total_data/n_batches:.4e}  phys {total_phys/n_batches:.4e}  "
                  f"val_mse {mse_val:.4e}  ({time.time()-t0:.0f}s)", flush=True)

    # final test metrics
    model.load_state_dict(torch.load(f"model_{args.tag}.pt"))
    with torch.no_grad():
        pred_test = model(s_test, t_out).cpu().numpy()
    yt = d["y_test"]
    stable_mask = d["s_test"].astype(bool)
    err = np.abs(pred_test - yt)
    rmse_all = np.sqrt((err ** 2).mean(axis=(1, 2)))
    rmse_stable = np.sqrt((err[stable_mask] ** 2).mean(axis=(1, 2)))
    final_spread = (pred_test.max(axis=1) - pred_test.min(axis=1)).max(axis=1)
    acc = ((final_spread < np.pi) == stable_mask).mean()
    print(f"\nTEST  rmse_all={np.rad2deg(rmse_all.mean()):.2f} deg | "
          f"stable_rmse={np.rad2deg(rmse_stable.mean()):.3f} deg (n={stable_mask.sum()}) | "
          f"clf_acc={acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
