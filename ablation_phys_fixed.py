"""Re-run the physics-residual ablation with the CORRECTLY scaled residual.

The residual previously omitted the synchronous speed ws on the power terms
(M d2delta - (Pm-Pe) + M c instead of M d2delta - ws*(Pm-Pe) + ws*M*c), which
made it a nearly-vacuous constraint (d2delta ~ 0). With the corrected residual
we re-evaluate the ablation at several loss weights.
"""
import numpy as np
import torch

from model import DeepONet, trunk_derivs, physics_residual

CLIP = 3.0 * np.pi


def train_phys(s_dim, n, te, dev, enc, y_train, st_train, t_out, lambda_phys,
               E, Pm, M, G, Bimag, Mtot, s_train, tc_train, stride=5):
    torch.manual_seed(0); np.random.seed(0)
    model = DeepONet(s_dim, n, p=512, t_scale=te).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
    nscen = len(y_train); nb = max(1, nscen // 128)
    t_phys = t_out[::stride]; tp_col = t_phys[:, 0]
    for ep in range(600):
        model.train()
        perm = torch.randperm(nscen, device=dev)
        for it in range(nb):
            idx = perm[it * 128:(it + 1) * 128]
            s = s_train[idx]; y = y_train[idx]; st = st_train[idx]
            delta = model(s, t_out)
            target = torch.clamp(y, -CLIP, CLIP)
            se = ((delta - target) ** 2).mean(dim=(1, 2))
            n_st = st.sum().clamp(min=1); n_un = (~st).sum().clamp(min=1)
            loss = 0.5 * (se[st].sum() / n_st + se[~st].sum() / n_un)
            Bc = model.branch_coeff(s)
            tp_, dtp_, d2tp_ = trunk_derivs(model.trunk, t_phys, model.t_scale)
            dp = torch.einsum("bnp,tp->bnt", Bc, tp_)
            d1 = torch.einsum("bnp,tp->bnt", Bc, dtp_)
            d2 = torch.einsum("bnp,tp->bnt", Bc, d2tp_)
            resid = physics_residual(dp, d1, d2, E, Pm, M, G, Bimag, Mtot) / M[None, :, None]
            mask = (tp_col[None, :] >= tc_train[idx][:, None]).float().unsqueeze(1)
            lp = (resid.pow(2) * mask).sum() / (mask.sum() + 1e-8)
            loss = loss + lambda_phys * lp
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_pooled(model, s_te, y_te, st_te, t_out, dev):
    with torch.no_grad():
        pred = model(s_te, t_out).cpu().numpy()
    y = y_te.cpu().numpy(); st = st_te.astype(bool)
    err = np.abs(pred - y)
    pooled = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
    fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
    acc = ((fs < np.pi) == st).mean()
    return pooled, acc


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); nb_ = int(d["n_buses"]); te = float(d["t_end"]); tm = float(d["t_max"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    y_train = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_test = torch.as_tensor(d["y_test"], dtype=torch.float32, device=dev)
    st_train = torch.as_tensor(d["s_train"].astype(bool), device=dev)
    st_test = d["s_test"].astype(bool)

    Pm = d["Pm"]; M = d["M"]; pf = d["pe_fault"]

    def sev_enc(fb, tc):
        s = (Pm[None, :] - pf[fb - 1]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / tm
        return torch.cat([torch.as_tensor(s, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    s_tr = sev_enc(d["fb_train"], d["tc_train"])
    s_te = sev_enc(d["fb_test"], d["tc_test"])

    E = torch.as_tensor(d["E"], dtype=torch.float32, device=dev)
    Pm_t = torch.as_tensor(d["Pm"], dtype=torch.float32, device=dev)
    M_t = torch.as_tensor(d["M"], dtype=torch.float32, device=dev)
    G_t = torch.as_tensor(d["G"], dtype=torch.float32, device=dev)
    B_t = torch.as_tensor(d["B"], dtype=torch.float32, device=dev)
    tc_train = torch.as_tensor(d["tc_train"], device=dev)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()

    m = train_phys(n + 1, n, te, dev, sev_enc, y_train, st_train, t_out, a.lam,
                   E, Pm_t, M_t, G_t, B_t, float(d["Mtot"]), s_tr, tc_train, stride=a.stride)
    p, acc = eval_pooled(m, s_te, y_test, st_test, t_out, dev)
    print(f"phys-loss (corrected) lambda={a.lam} stride={a.stride}: pooled={p:.2f} deg, clf={acc:.3f}",
          flush=True)
    torch.save(m.state_dict(), f"model_phys_fixed_{a.lam}_s{a.stride}.pt")


if __name__ == "__main__":
    main()
