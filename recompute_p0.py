# -*- coding: utf-8 -*-
"""P0 recomputation after the reviewer's code audit.

A1: signed pairwise-difference score (errors keep their sign).
A2: stability criterion = max-over-time pairwise spread (not the last step).
A3: gated coverage with fixed, disjoint calibration/evaluation indices.
A4: finite-sample quantile via the order statistic ceil((n+1)(1-alpha)).
"""
import numpy as np
import torch

from model import DeepONet, AttentionDeepONet
from baseline_mlp import PointMLP
from baseline_rnn import GRUTrajectory
from ablation_arch import FNODeepONet
from power_system import ClassicalModel
from dataset import simulate_batch
from evaluate import severity_enc

F32 = dict(dtype=torch.float32)
ALPHA = 0.1


def order_quantile(scores_cal, alpha=ALPHA):
    n = len(scores_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return np.sort(scores_cal)[k - 1]


def spread_max(pred):
    """Max over time of the pairwise angle spread (matches the reference label)."""
    return (pred.max(axis=1) - pred.min(axis=1)).max(axis=1)   # (B,)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= main 39-bus model =================
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()
    fb, tc, y, st = d["fb_test"], d["tc_test"], d["y_test"], d["s_test"].astype(bool)
    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()

    # ---- A2: max-over-time classification ----
    sm = spread_max(pred)
    pred_stable_new = sm < np.pi
    pred_stable_old = (pred[:, :, -1].max(1) - pred[:, :, -1].min(1)) < np.pi
    acc_new = (pred_stable_new == st).mean()
    acc_old = (pred_stable_old == st).mean()
    tp = int((pred_stable_new & st).sum()); fp = int((pred_stable_new & ~st).sum())
    fn = int((~pred_stable_new & st).sum()); tn = int((~pred_stable_new & ~st).sum())
    flipped = int((pred_stable_new != pred_stable_old).sum())
    print(f"A2 main: acc(max-over-time)={acc_new:.4f} (old last-step={acc_old:.4f}), "
          f"flipped samples={flipped}/960, TP/FP/FN/TN={tp}/{fp}/{fn}/{tn}", flush=True)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    print(f"A2 main: P={prec:.3f} R={rec:.3f} F1={2*tp/(2*tp+fp+fn):.3f}", flush=True)

    # ---- A1: signed pairwise score ----
    signed = pred - y                                   # keep the sign
    e_st = signed[st]                                   # (S, n, T) signed
    diffs = np.abs(e_st[:, None, :, :] - e_st[:, :, None, :])   # |e_i - e_j| signed diff
    R_pw = diffs.max(axis=(1, 2, 3))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(R_pw)); nc = len(R_pw) // 2
    q_pw = order_quantile(R_pw[idx[:nc]])
    cov_pw = (R_pw[idx[nc:]] <= q_pw).mean()
    print(f"A1 signed pairwise: q={np.rad2deg(q_pw):.1f} deg, coverage={cov_pw:.3f} "
          f"(old abs-based was 101.5 deg)", flush=True)

    # ---- A4: order-statistic quantile for the main per-machine score ----
    err = np.abs(pred - y)
    Rg = err[st].max(axis=(1, 2))
    q_os = order_quantile(Rg[idx[:nc]])
    cov_os = (Rg[idx[nc:]] <= q_os).mean()
    print(f"A4 main score: q(order-stat)={np.rad2deg(q_os):.1f} deg, coverage={cov_os:.3f}", flush=True)

    # ---- A3: gated coverage, correct protocol ----
    # gate built on the evaluation half only; report within that half:
    #   (a) coverage over predicted-stable samples of the evaluation half
    #   (b) coverage over true-stable & predicted-stable of the evaluation half
    #   (c) gate purity and wrong-release rate on the evaluation half
    eva_idx = idx[nc:]
    st_e, ps_e, sm_e = st[eva_idx], pred_stable_new[eva_idx], sm[eva_idx]
    err_e = err[eva_idx]
    purity = st_e[ps_e].mean() if ps_e.sum() else np.nan
    wr = (~st_e & ps_e).sum() / (~st_e).sum() if (~st_e).sum() else np.nan
    cov_all_ps = (err_e[ps_e].max(axis=(1, 2)) <= q_os).mean() if ps_e.sum() else np.nan
    in_gate = st_e & ps_e
    cov_ts_ps = (err_e[in_gate].max(axis=(1, 2)) <= q_os).mean() if in_gate.sum() else np.nan
    print(f"A3 gated (evaluation half, n={len(eva_idx)}): "
          f"purity={purity:.3f}, wrong-release rate={wr:.3f}, "
          f"coverage(pred-stable)={cov_all_ps:.3f}, "
          f"coverage(true-stable & pred-stable)={cov_ts_ps:.3f} "
          f"(n_ps={int(ps_e.sum())}, n_tsps={int(in_gate.sum())})", flush=True)

    # ---- A2 for CCT (12 held-out buses, max-over-time criterion) ----
    model39 = ClassicalModel(case="39")
    cct_true_cache = {}

    def true_cct(bus):
        if bus not in cct_true_cache:
            lo, hi = 0.01, 0.8
            for _ in range(16):
                mid = (lo + hi) / 2
                r = simulate_batch(model39, np.array([bus]), np.array([mid]), np.array([-1]), 4.0)
                lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
            cct_true_cache[bus] = lo
        return cct_true_cache[bus]

    def pred_cct(bus):
        def stable(tcc):
            with torch.no_grad():
                tr = m(severity_enc(np.array([bus]), np.array([tcc]), d, dev), t).cpu().numpy()
            return spread_max(tr)[0] < np.pi
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if stable(mid) else (lo, mid)
        return lo

    buses = np.unique(fb)
    cct_errs, rows = [], []
    for b in buses:
        ct = true_cct(int(b)); cp = pred_cct(int(b))
        cct_errs.append((cp - ct) * 1000)
        rows.append((int(b), round(ct * 1000, 1), round(cp * 1000, 1), round((cp - ct) * 1000, 1)))
    e = np.array(cct_errs)
    print(f"A2 CCT (max-over-time): mean={e.mean():.1f} ms, |mean|={np.abs(e).mean():.1f} ms, "
          f"max={np.abs(e).max():.1f} ms", flush=True)
    print("A2 CCT per-bus (bus, true, pred, err ms):", flush=True)
    for r in rows:
        print("   ", r, flush=True)

    # ================= other models: max-over-time accuracy =================
    def eval_model(loader, s, y, st, name, tag):
        with torch.no_grad():
            pr = loader(s, t).cpu().numpy()
        sm_ = spread_max(pr)
        acc_ = ((sm_ < np.pi) == st).mean()
        print(f"A2 {name}: max-over-time acc={acc_:.4f}", flush=True)

    s_te = severity_enc(fb, tc, d, dev)
    y_te = y; st_te = st

    mlp = PointMLP(n + 1, n, t_scale=te).to(dev)
    mlp.load_state_dict(torch.load("model_mlp.pt")); mlp.eval()
    eval_model(mlp, s_te, y_te, st_te, "MLP", "mlp")

    rnn = GRUTrajectory(n + 1, n, t_scale=te).to(dev)
    rnn.load_state_dict(torch.load("model_rnn.pt")); rnn.eval()
    eval_model(rnn, s_te, y_te, st_te, "GRU", "rnn")

    attn = AttentionDeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    attn.load_state_dict(torch.load("model_attn_p512.pt")); attn.eval()
    eval_model(attn, s_te, y_te, st_te, "attention", "attn")

    fno = FNODeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    fno.load_state_dict(torch.load("model_fno_p512.pt")); fno.eval()
    eval_model(fno, s_te, y_te, st_te, "FNO", "fno")

    phys = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    phys.load_state_dict(torch.load("model_phys_fixed_0.001.pt")); phys.eval()
    eval_model(phys, s_te, y_te, st_te, "phys-loss", "phys")

    k8 = DeepONet(n + 1, n, p=512, K=8, t_scale=te).to(dev)
    k8.load_state_dict(torch.load("model_k8_p512.pt")); k8.eval()
    eval_model(k8, s_te, y_te, st_te, "K8", "k8")

    clip = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    clip.load_state_dict(torch.load("model_clip2pi_p512.pt")); clip.eval()
    eval_model(clip, s_te, y_te, st_te, "clip2pi", "clip")

    def eval_transfer(model_path, datafile, s_dim, K=16, name=""):
        dd = np.load(datafile)
        nn = int(dd["n_machines"]); tee = float(dd["t_end"])
        tt = torch.as_tensor(dd["t_out"], **F32, device=dev)[:, None]
        mm = DeepONet(s_dim, nn, p=512, K=K, t_scale=tee).to(dev)
        mm.load_state_dict(torch.load(model_path)); mm.eval()
        ss = severity_enc(dd["fb_test"], dd["tc_test"], dd, dev)
        with torch.no_grad():
            pr = mm(ss, tt).cpu().numpy()
        sm_ = spread_max(pr)
        stt = dd["s_test"].astype(bool)
        acc_ = ((sm_ < np.pi) == stt).mean()
        print(f"A2 {name}: max-over-time acc={acc_:.4f}", flush=True)

    eval_transfer("model_oneaxis_p512.pt", "data39_oneaxis.npz", n + 1, name="one-axis")
    eval_transfer("model_exciter_p512.pt", "data39_exciter.npz", n + 1, name="exciter")
    eval_transfer("model_damped_p512.pt", "data39_damped.npz", n + 1, name="damped")
    eval_transfer("model_118.pt", "data118.npz", 55, name="118")

    # N-1 / N-2
    def eval_n1(tag, split, name):
        dn = np.load("data39_n1.npz")
        n1 = int(dn["n_machines"]); te1 = float(dn["t_end"]); tm1 = float(dn["t_max"])
        t1 = torch.as_tensor(dn["t_out"], **F32, device=dev)[:, None]
        Pm = dn["Pm"]; M = dn["M"]; pf = dn["pe_fault"]; pp = dn["pe_post"]
        m1 = DeepONet(2 * n1 + 1, n1, p=512, t_scale=te1).to(dev)
        m1.load_state_dict(torch.load(f"model_{tag}.pt")); m1.eval()

        def enc1(fb_, tl_, tc_):
            sf = (Pm[None, :] - pf[fb_ - 1]) / M[None, :] * 10.0
            sp_ = np.zeros((len(fb_), n1)); v = tl_ >= 0
            if v.any():
                sp_[v] = (Pm[None, :] - pp[tl_[v]]) / M[None, :] * 10.0
            tcn = np.asarray(tc_, dtype=np.float32)[:, None] / tm1
            return torch.cat([torch.as_tensor(sf, **F32, device=dev),
                              torch.as_tensor(sp_, **F32, device=dev),
                              torch.as_tensor(tcn, **F32, device=dev)], dim=1)

        if split == "bus":
            fb_, tl_, tc_, y_, st_ = (dn["fb_testb"], dn["tl_testb"], dn["tc_testb"],
                                      dn["y_testb"], dn["s_testb"].astype(bool))
        else:
            fb_, tl_, tc_, y_, st_ = (dn["fb_testl"], dn["tl_testl"], dn["tc_testl"],
                                      dn["y_testl"], dn["s_testl"].astype(bool))
        with torch.no_grad():
            pr = m1(enc1(fb_, tl_, tc_), t1).cpu().numpy()
        sm_ = spread_max(pr)
        print(f"A2 {name}: max-over-time acc={((sm_ < np.pi) == st_).mean():.4f}", flush=True)

    eval_n1("n1_p512", "bus", "N-1 bus")
    eval_n1("n1_p512", "line", "N-1 line")

    dn2 = np.load("data39_n2.npz")
    n2 = int(dn2["n_machines"]); te2 = float(dn2["t_end"]); tm2 = float(dn2["t_max"])
    t2 = torch.as_tensor(dn2["t_out"], **F32, device=dev)[:, None]
    m2 = DeepONet(2 * n2 + 1, n2, p=512, t_scale=te2).to(dev)
    m2.load_state_dict(torch.load("model_n2_p512.pt")); m2.eval()
    sf2 = (dn2["Pm"][None, :] - dn2["pe_fault"][dn2["fb_testn2"] - 1]) / dn2["M"][None, :] * 10.0
    tcn2 = np.asarray(dn2["tc_testn2"], dtype=np.float32)[:, None] / tm2
    s2 = torch.cat([torch.as_tensor(sf2, **F32, device=dev),
                    torch.as_tensor(dn2["sev_testn2"], **F32, device=dev),
                    torch.as_tensor(tcn2, **F32, device=dev)], dim=1)
    with torch.no_grad():
        pr2 = m2(s2, t2).cpu().numpy()
    st2 = dn2["s_testn2"].astype(bool)
    print(f"A2 N-2 pair: max-over-time acc={((spread_max(pr2) < np.pi) == st2).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
