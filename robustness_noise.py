"""Robustness of the operator to measurement noise in the severity encoding.

The physics-informed encoding a_i(k)=(Pmi-Pei^f)/Mi is itself computed from
measurements / model parameters; here we perturb the encoded acceleration vector
with Gaussian noise (relative to each machine's scale) and measure the operator's
degradation in trajectory RMSE and stability classification.
"""
import numpy as np
import torch

from model import DeepONet
from evaluate import severity_enc, predict


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); t_end = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=t_end).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()

    fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]; st = d["s_test"].astype(bool)
    s = severity_enc(fb, tc, d, dev)          # (B, n+1): [accel (n), tcn (1)]
    sev = s[:, :n]                             # the n acceleration dims
    tcn = s[:, n:]
    scale = sev.std(dim=0)                     # per-machine scale across test set

    def eval_noise(sev_noisy):
        s_noisy = torch.cat([sev_noisy, tcn], dim=1)
        with torch.no_grad():
            pred = m(s_noisy, t).cpu().numpy()
        err = np.abs(pred - y)
        rmse = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
        fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
        acc = ((fs < np.pi) == st).mean()
        return rmse, acc

    r0, a0 = eval_noise(sev)
    print(f"noise=0.00 (clean): RMSE={r0:.2f} deg, acc={a0:.3f}")

    rng = np.random.default_rng(0)
    for sigma in [0.05, 0.10, 0.20]:
        rmses, accs = [], []
        for _ in range(5):                     # 5 noise draws
            noise = sigma * scale[None, :].cpu().numpy() * rng.standard_normal(sev.shape)
            sev_noisy = torch.as_tensor(sev.cpu().numpy() + noise, dtype=torch.float32, device=dev)
            rmse, acc = eval_noise(sev_noisy)
            rmses.append(rmse); accs.append(acc)
        print(f"noise={sigma:.2f}: RMSE={np.mean(rmses):.2f} +/- {np.std(rmses):.2f} deg, "
              f"acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}")


if __name__ == "__main__":
    main()
