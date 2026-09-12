"""Evaluate the trained N-1 operator on held-out buses and held-out lines."""
import numpy as np
import torch

from model import DeepONet


def main(tag="n1", p=256):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load("data39_n1.npz")
    n = int(d["n_machines"]); t_end = float(d["t_end"]); t_max = float(d["t_max"])
    t_out = torch.as_tensor(d["t_out"], dtype=torch.float32, device=dev)[:, None]
    Pm = d["Pm"]; M = d["M"]; pe_fault = d["pe_fault"]; pe_post = d["pe_post"]

    def enc(fb, tl, tc):
        sev_f = (Pm[None, :] - pe_fault[fb - 1]) / M[None, :] * 10.0
        sev_p = np.zeros((len(fb), n))
        valid = tl >= 0
        if valid.any():
            sev_p[valid] = (Pm[None, :] - pe_post[tl[valid]]) / M[None, :] * 10.0
        tcn = np.asarray(tc, dtype=np.float32)[:, None] / t_max
        return torch.cat([torch.as_tensor(sev_f, dtype=torch.float32, device=dev),
                          torch.as_tensor(sev_p, dtype=torch.float32, device=dev),
                          torch.as_tensor(tcn, device=dev)], dim=1)

    model = DeepONet(2 * n + 1, n, p=p, t_scale=t_end).to(dev)
    model.load_state_dict(torch.load(f"model_{tag}.pt")); model.eval()

    def report(fb, tl, tc, y, st, name):
        with torch.no_grad():
            pred = model(enc(fb, tl, tc), t_out).cpu().numpy()
        err = np.abs(pred - y)
        st = st.astype(bool)
        rmse = np.rad2deg(np.sqrt((err[st] ** 2).mean()))
        fs = pred[:, :, -1].max(1) - pred[:, :, -1].min(1)
        acc = ((fs < np.pi) == st).mean()
        print(f"{name:10s}: stable_rmse={rmse:6.2f} deg | clf_acc={acc:.3f} (n_stable={st.sum()})")

    report(d["fb_testb"], d["tl_testb"], d["tc_testb"], d["y_testb"], d["s_testb"], "test_bus")
    report(d["fb_testl"], d["tl_testl"], d["tc_testl"], d["y_testl"], d["s_testl"], "test_line")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n1")
    ap.add_argument("--p", type=int, default=256)
    a = ap.parse_args()
    main(a.tag, a.p)
