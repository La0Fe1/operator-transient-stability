# -*- coding: utf-8 -*-
"""Pooled stable RMSE for the N-1/N-2 models on the regenerated references, plus
N-2 split-conformal coverage with the order-statistic quantile."""
import numpy as np
import torch

from model import DeepONet
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)


def order_quantile(scores, alpha=0.1):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- N-1 ----
    dn = np.load("data39_n1.npz")
    n1 = int(dn["n_machines"]); te1 = float(dn["t_end"]); tm1 = float(dn["t_max"])
    t1 = torch.as_tensor(dn["t_out"], **F32, device=dev)[:, None]
    m1 = DeepONet(2 * n1 + 1, n1, p=512, t_scale=te1).to(dev)
    m1.load_state_dict(torch.load("model_n1_p512.pt")); m1.eval()
    Pm = dn["Pm"]; M = dn["M"]; pf = dn["pe_fault"]; pp = dn["pe_post"]

    def enc1(fb_, tl_, tc_):
        sf = (Pm[None, :] - pf[np.asarray(fb_) - 1]) / M[None, :] * 10.0
        sp_ = np.zeros((len(fb_), n1))
        v = np.asarray(tl_) >= 0
        if v.any():
            sp_[v] = (Pm[None, :] - pp[np.asarray(tl_)[v]]) / M[None, :] * 10.0
        tcn = np.asarray(tc_, dtype=np.float32)[:, None] / tm1
        return torch.cat([torch.as_tensor(sf, **F32, device=dev),
                          torch.as_tensor(sp_, **F32, device=dev),
                          torch.as_tensor(tcn, **F32, device=dev)], dim=1)

    for split in ("b", "l"):
        fb_, tl_, tc_, y_, st_ = (dn[f"fb_test{split}"], dn[f"tl_test{split}"],
                                  dn[f"tc_test{split}"], dn[f"y_test{split}"],
                                  dn[f"s_test{split}"].astype(bool))
        with torch.no_grad():
            pr = m1(enc1(fb_, tl_, tc_), t1).cpu().numpy()
        rmse = float(np.sqrt(((pr - y_)[st_] ** 2).mean()))
        scores = np.abs(pr - y_)[st_].max(axis=(1, 2))
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(scores)); nc = len(scores) // 2
        q = order_quantile(scores[idx[:nc]]); cov = (scores[idx[nc:]] <= q).mean()
        print(f"N-1 {split}: pooled stable RMSE={np.rad2deg(rmse):.1f} deg, "
              f"q={np.rad2deg(q):.1f} deg, cov={cov:.3f} "
              f"(n_cal={nc}, n_eva={len(scores)-nc})", flush=True)

    # ---- N-2 ----
    dn2 = np.load("data39_n2.npz")
    n2 = int(dn2["n_machines"]); te2 = float(dn2["t_end"]); tm2 = float(dn2["t_max"])
    t2 = torch.as_tensor(dn2["t_out"], **F32, device=dev)[:, None]
    m2 = DeepONet(2 * n2 + 1, n2, p=512, t_scale=te2).to(dev)
    m2.load_state_dict(torch.load("model_n2_p512.pt")); m2.eval()
    for split in ("n2", "b"):
        fb_, tc_, sev_, y_, st_ = (dn2[f"fb_test{split}"], dn2[f"tc_test{split}"],
                                   dn2[f"sev_test{split}"], dn2[f"y_test{split}"],
                                   dn2[f"s_test{split}"].astype(bool))
        sf = (dn2["Pm"][None, :] - dn2["pe_fault"][np.asarray(fb_) - 1]) / dn2["M"][None, :] * 10.0
        tcn = np.asarray(tc_, dtype=np.float32)[:, None] / tm2
        s2 = torch.cat([torch.as_tensor(sf, **F32, device=dev),
                        torch.as_tensor(sev_, **F32, device=dev),
                        torch.as_tensor(tcn, **F32, device=dev)], dim=1)
        with torch.no_grad():
            pr = m2(s2, t2).cpu().numpy()
        rmse = float(np.sqrt(((pr - y_)[st_] ** 2).mean()))
        scores = np.abs(pr - y_)[st_].max(axis=(1, 2))
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(scores)); nc = len(scores) // 2
        q = order_quantile(scores[idx[:nc]]); cov = (scores[idx[nc:]] <= q).mean()
        print(f"N-2 {split}: pooled stable RMSE={np.rad2deg(rmse):.1f} deg, "
              f"q={np.rad2deg(q):.1f} deg, cov={cov:.3f} "
              f"(n_cal={nc}, n_eva={len(scores)-nc})", flush=True)


if __name__ == "__main__":
    main()
