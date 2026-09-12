"""Train base DeepONet vs attention DeepONet on boundary-aware data, compare."""
import numpy as np
import torch

from model import DeepONet, AttentionDeepONet
from power_system import ClassicalModel
from dataset import simulate_batch

CLIP = 3.0 * np.pi


def train(model_cls, s_tr, y_tr, st_tr, t_out, dev, n, te, p=512, epochs=600, **kw):
    torch.manual_seed(0); np.random.seed(0)
    model = model_cls(s_tr.shape[1], n, p=p, t_scale=te, **kw).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = y_tr.shape[0]; nb = max(1, B // 128)
    for ep in range(epochs):
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
    return model


def eval_model(model, s_te, y_te, st_te, t_out, dev, t_out_arr, tc_te):
    with torch.no_grad():
        pred = model(s_te, t_out).cpu().numpy()
    y_te = y_te.cpu().numpy()
    err = np.abs(pred - y_te); st = st_te.astype(bool)
    pooled = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
    fe = []; pe = []
    for i in range(len(tc_te)):
        if not st[i]:
            continue
        msk = t_out_arr < tc_te[i]
        fe.append(np.rad2deg(err[i][:, msk].mean())); pe.append(np.rad2deg(err[i][:, ~msk].mean()))
    fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
    acc = ((fs < np.pi) == st).mean()
    return pooled, np.mean(fe), np.mean(pe), acc


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

    for name, cls, kw in [("DeepONet", DeepONet, {}),
                          ("Attention", AttentionDeepONet, {"d_model": 64, "n_heads": 4})]:
        model = train(cls, s_tr, y_tr, st_tr, t_out, dev, n, te, p=512, **kw)
        pooled, fe, pe, acc = eval_model(model, s_te, y_te, st_te, t_out, dev, t_out_arr, d["tc_test"])
        print(f"{name:12s}: pooled={pooled:6.2f} deg, fault-on={fe:5.2f}, post-fault={pe:5.2f}, clf={acc:.3f}")


if __name__ == "__main__":
    main()
