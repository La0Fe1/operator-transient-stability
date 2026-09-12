"""Architecture ablations at p=512: self-attention branch and FNO trunk.

Re-runs the two architecture variants of Section 6.8 at the final p=512
architecture so their numbers are reported alongside the other baselines.
"""
import numpy as np
import torch
import torch.nn as nn

from model import DeepONet, AttentionDeepONet
from evaluate import severity_enc, predict

CLIP = 3.0 * np.pi


class FNOTrunk(nn.Module):
    """Fourier-neural-operator trunk: spectral convolution over the time axis."""

    def __init__(self, p, modes=16, width=64):
        super().__init__()
        self.modes = modes
        self.lift = nn.Linear(1, width)
        self.w = nn.Parameter(torch.randn(modes, width, width) * 0.02)   # real part
        self.wb = nn.Parameter(torch.randn(modes, width, width) * 0.02)  # imag part
        self.proj = nn.Linear(width, p)

    def forward(self, tn):
        x = self.lift(tn)                        # (T, width)
        xf = torch.fft.rfft(x, dim=0)            # (Tf, width) complex
        out = torch.zeros_like(xf)
        K = min(self.modes, xf.shape[0])
        for k in range(K):
            out[k] = xf[k] @ (self.w[k] + 1j * self.wb[k])
        x = torch.fft.irfft(out, n=x.shape[0], dim=0)
        return self.proj(x)


class FNODeepONet(DeepONet):
    """DeepONet whose Fourier-feature MLP trunk is replaced by an FNO spectral trunk."""

    def __init__(self, s_dim, n, p=512, K=16, t_scale=1.0, modes=16, width=64):
        super().__init__(s_dim, n, p=p, K=K, t_scale=t_scale)
        self.trunk = FNOTrunk(p, modes=modes, width=width)

    def trunk_basis(self, t):
        return self.trunk(t / self.t_scale)


def train_eval(model, s_train, y_train, st_train, s_test, y_test, st_test, t_out, dev, tag):
    torch.manual_seed(0); np.random.seed(0)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
    nscen = len(y_train); nb = max(1, nscen // 128)
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
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(s_test, t_out).cpu().numpy()
    y = y_test.cpu().numpy(); st = st_test.astype(bool)
    err = np.abs(pred - y)
    pooled = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
    fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
    acc = ((fs < np.pi) == st).mean()
    print(f"{tag}: pooled stable RMSE={pooled:.2f} deg, clf acc={acc:.3f}", flush=True)
    torch.save(model.state_dict(), f"model_{tag}.pt")
    return pooled, acc


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    s_train = severity_enc(d["fb_train"], d["tc_train"], d, dev)
    s_test = severity_enc(d["fb_test"], d["tc_test"], d, dev)
    y_train = torch.as_tensor(d["y_train"], dtype=torch.float32, device=dev)
    y_test = torch.as_tensor(d["y_test"], dtype=torch.float32, device=dev)
    st_train = torch.as_tensor(d["s_train"].astype(bool), device=dev)
    st_test = d["s_test"].astype(bool)

    train_eval(AttentionDeepONet(n + 1, n, p=512, t_scale=te),
               s_train, y_train, st_train, s_test, y_test, st_test, t_out, dev, "attn_p512")
    train_eval(FNODeepONet(n + 1, n, p=512, t_scale=te),
               s_train, y_train, st_train, s_test, y_test, st_test, t_out, dev, "fno_p512")


if __name__ == "__main__":
    main()
