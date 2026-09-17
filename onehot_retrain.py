# -*- coding: utf-8 -*-
"""Retrain the one-hot p=512 ablation and save it, then evaluate under the
corrected (max-over-time) criterion on the regenerated reference."""
import numpy as np
import torch

from ablation_p512 import train          # reuse the same training loop/seed

F32 = dict(dtype=torch.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39.npz")
    n = int(d["n_machines"]); nb_ = int(d["n_buses"]); te = float(d["t_end"])
    tm = float(d["t_max"])
    t_out = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    y_train = torch.as_tensor(d["y_train"], **F32, device=dev)
    st_train = torch.as_tensor(d["s_train"].astype(bool), device=dev)

    def oh_enc(fb, tc):
        oh = np.eye(nb_)[fb - 1]
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / tm
        return torch.cat([torch.as_tensor(oh, **F32, device=dev),
                          torch.as_tensor(tcn, **F32, device=dev)], dim=1)

    s_tr_oh = oh_enc(d["fb_train"], d["tc_train"])
    m = train(nb_ + 1, n, te, dev, oh_enc, y_train, st_train, t_out, s_train=s_tr_oh)
    torch.save(m.state_dict(), "model_onehot_p512.pt")

    # evaluate under the corrected criterion
    fb, tc, y, st = d["fb_test"], d["tc_test"], d["y_test"], d["s_test"].astype(bool)
    s_te = oh_enc(fb, tc)
    with torch.no_grad():
        pr = m(s_te, t_out).cpu().numpy()
    sm = (pr.max(axis=1) - pr.min(axis=1)).max(axis=1)
    acc = ((sm < np.pi) == st).mean()
    rmse = float(np.sqrt(((pr - y)[st] ** 2).mean()))
    print(f"one-hot p=512 (max-over-time): acc={acc*100:.2f}%, "
          f"pooled stable RMSE={np.rad2deg(rmse):.1f} deg", flush=True)


if __name__ == "__main__":
    main()
