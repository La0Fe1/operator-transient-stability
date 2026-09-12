"""Re-run one-hot and residual-loss ablations at p=512 for a consistent Table II."""
import numpy as np
import torch

from model import DeepONet, trunk_derivs, physics_residual

CLIP = 3.0 * np.pi


def train(s_dim, n, te, dev, enc, y_train, st_train, t_out, tc_train=None, lambda_phys=0.0,
          E=None, Pm=None, M=None, G=None, Bimag=None, Mtot=None, s_train=None):
    torch.manual_seed(0); np.random.seed(0)
    model = DeepONet(s_dim, n, p=512, t_scale=te).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
    nscen = len(y_train); nb = max(1, nscen // 128)
    t_phys = t_out[::5]; tp_col = t_phys[:, 0]
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
            if lambda_phys > 0:
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


def eval_pooled(model, s_te, y_te, st_te, t_out, dev, t_out_arr, tc_te):
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
    t_out_arr = d["t_out"]
    y_train = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_test = torch.as_tensor(d["y_test"], dtype=torch.float32, device=dev)
    st_train = torch.as_tensor(d["s_train"].astype(bool), device=dev)
    st_test = d["s_test"].astype(bool)

    # severity encoding (for residual loss)
    Pm = d["Pm"]; M = d["M"]; pf = d["pe_fault"]
    def sev_enc(fb, tc):
        s = (Pm[None, :] - pf[fb - 1]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / tm
        return torch.cat([torch.as_tensor(s, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    # one-hot encoding
    def oh_enc(fb, tc):
        oh = np.eye(nb_)[fb - 1]
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / tm
        return torch.cat([torch.as_tensor(oh, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    s_tr_sev = sev_enc(d["fb_train"], d["tc_train"]); s_te_sev = sev_enc(d["fb_test"], d["tc_test"])
    s_tr_oh = oh_enc(d["fb_train"], d["tc_train"]); s_te_oh = oh_enc(d["fb_test"], d["tc_test"])

    # one-hot at p=512
    m = train(nb_ + 1, n, te, dev, oh_enc, y_train, st_train, t_out, s_train=s_tr_oh)
    p, a = eval_pooled(m, s_te_oh, y_test, st_test, t_out, dev, t_out_arr, d["tc_test"])
    print(f"one-hot p=512: pooled={p:.2f} deg, clf={a:.3f}")

    # residual loss at p=512
    E = torch.as_tensor(d["E"], dtype=torch.float32, device=dev)
    Pm_t = torch.as_tensor(d["Pm"], dtype=torch.float32, device=dev)
    M_t = torch.as_tensor(d["M"], dtype=torch.float32, device=dev)
    G_t = torch.as_tensor(d["G"], dtype=torch.float32, device=dev)
    B_t = torch.as_tensor(d["B"], dtype=torch.float32, device=dev)
    m = train(n + 1, n, te, dev, sev_enc, y_train, st_train, t_out, s_train=s_tr_sev,
              lambda_phys=0.01, E=E, Pm=Pm_t, M=M_t, G=G_t, Bimag=B_t,
              Mtot=float(d["Mtot"]), tc_train=torch.as_tensor(d["tc_train"], device=dev))
    p, a = eval_pooled(m, s_te_sev, y_test, st_test, t_out, dev, t_out_arr, d["tc_test"])
    print(f"residual loss p=512: pooled={p:.2f} deg, clf={a:.3f}")


if __name__ == "__main__":
    main()
