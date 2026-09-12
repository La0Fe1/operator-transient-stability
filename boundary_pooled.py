"""Train on boundary-aware data and compute the pooled post-fault RMSE (p=512)."""
import numpy as np
import torch

from model import DeepONet

CLIP = 3.0 * np.pi


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39_boundary.npz")
    n = int(d["n_machines"]); te = float(d["t_end"]); tm = float(d["t_max"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    t_out_arr = d["t_out"]
    Pm = d["Pm"]; M = d["M"]; pf = d["pe_fault"]

    def enc(fb, tc):
        sev = (Pm[None, :] - pf[fb - 1]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / tm
        return torch.cat([torch.as_tensor(sev, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    s_tr = enc(d["fb_train"], d["tc_train"]); s_te = enc(d["fb_test"], d["tc_test"])
    y_tr = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_te = torch.as_tensor(d["y_test"], dtype=torch.float32, device=dev)
    st_tr = torch.as_tensor(d["s_train"].astype(bool), device=dev)
    st_te = d["s_test"].astype(bool)

    torch.manual_seed(0); np.random.seed(0)
    model = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
    B = len(y_tr); nb = max(1, B // 128)
    for ep in range(600):
        model.train()
        perm = torch.randperm(B, device=dev)
        for it in range(nb):
            idx = perm[it * 128:(it + 1) * 128]
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
    model.eval()

    with torch.no_grad():
        pred = model(s_te, t_out).cpu().numpy()
    y = d["y_test"]; st = st_te.astype(bool)
    err = np.abs(pred - y)
    fo = []; pf = []
    for i in range(len(d["tc_test"])):
        if not st[i]:
            continue
        msk = t_out_arr < d["tc_test"][i]
        fo.append(err[i][:, msk].ravel()); pf.append(err[i][:, ~msk].ravel())
    fo = np.concatenate(fo); pf = np.concatenate(pf)
    fo_rms = np.rad2deg(np.sqrt((fo ** 2).mean()))
    pf_rms = np.rad2deg(np.sqrt((pf ** 2).mean()))
    overall = np.rad2deg(np.sqrt((np.concatenate([fo, pf]) ** 2).mean()))
    print(f"boundary-aware pooled: fault-on={fo_rms:.1f} deg, post-fault={pf_rms:.1f} deg, overall={overall:.1f} deg")


if __name__ == "__main__":
    main()
