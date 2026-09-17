# -*- coding: utf-8 -*-
"""Table 3 variant metrics under the corrected protocol: max-over-time accuracy
and pooled stable RMSE on the regenerated reference."""
import numpy as np
import torch

from model import DeepONet, AttentionDeepONet
from baseline_mlp import PointMLP
from baseline_rnn import GRUTrajectory
from ablation_arch import FNODeepONet
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    fb, tc, y, st = d["fb_test"], d["tc_test"], d["y_test"], d["s_test"].astype(bool)

    def spread_max(p):
        return (p.max(axis=1) - p.min(axis=1)).max(axis=1)

    def report(model, name, onehot=False):
        if onehot:
            s_in = torch.cat([torch.eye(d["n_buses"])[fb - 1].to(dev),
                              torch.as_tensor(tc / float(d["t_max"]), **F32,
                                              device=dev)[:, None]], dim=1)
        else:
            s_in = severity_enc(fb, tc, d, dev)
        with torch.no_grad():
            pr = model(s_in, t).cpu().numpy()
        acc = ((spread_max(pr) < np.pi) == st).mean()
        rmse = float(np.sqrt(((pr - y)[st] ** 2).mean()))
        print(f"{name}: acc={acc*100:.2f}%  pooled stable RMSE={np.rad2deg(rmse):.1f} deg",
              flush=True)

    for path, name, onehot in [
        ("model_onehot_p512.pt", "one-hot", True),
        ("model_phys_fixed_0.001.pt", "phys 0.001", False),
        ("model_phys_fixed_0.01.pt", "phys 0.01", False),
        ("model_attn_p512.pt", "attention", False),
        ("model_fno_p512.pt", "FNO", False),
        ("model_k8_p512.pt", "K8", False),
        ("model_clip2pi_p512.pt", "clip2pi", False),
    ]:
        if "attn" in path:
            model = AttentionDeepONet(n + 1, n, p=512, t_scale=te).to(dev)
        elif "fno" in path:
            model = FNODeepONet(n + 1, n, p=512, t_scale=te).to(dev)
        else:
            K = 8 if "k8" in path else 16
            sdim = d["n_buses"] + 1 if onehot else n + 1
            model = DeepONet(sdim, n, p=512, K=K, t_scale=te).to(dev)
        model.load_state_dict(torch.load(path)); model.eval()
        report(model, name, onehot)

    # pointwise MLP and GRU (RMSE under new reference)
    mlp = PointMLP(n + 1, n, t_scale=te).to(dev)
    mlp.load_state_dict(torch.load("model_mlp.pt")); mlp.eval()
    report(mlp, "MLP")
    rnn = GRUTrajectory(n + 1, n, t_scale=te).to(dev)
    rnn.load_state_dict(torch.load("model_rnn.pt")); rnn.eval()
    report(rnn, "GRU")


if __name__ == "__main__":
    main()
