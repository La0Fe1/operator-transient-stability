# -*- coding: utf-8 -*-
"""Single source of truth for all paper numbers after the code audit fixes.

Protocol (audit items):
  A1  signed pairwise-difference score  R_pw = max_{i,j,t} |e_i - e_j| (signed errors)
  A2  stability = max-over-time pairwise spread on the output grid (same as labels)
  A3  gated coverage on the disjoint evaluation half only; purity / wrong-release
  A4  finite-sample quantile k = ceil((n_cal+1)(1-alpha)) order statistic
  B4  decidable fraction: predicted spread + q_pw < pi
  B5  heuristic CCT band endpoints from the shifted criterion
  B6  reference CCT: event-aligned RK4 at dt=0.5 ms (simulate_batch fixed)
  B7  monotonicity: dense tc sweep, count sign flips for operator and reference
  D2  regional pooled RMSE (one-pass sqrt of mean sq over the region)
  D3  uniform-random predictor RMSE computed on the actual test trajectories
  C1  MLP/GRU under the same conformal + CCT protocol
  C3  severity-MLP (same encoding) classifier and CCT regressor

Outputs: recompute_all.json + readable recompute_all.txt (paste into paper.tex macros).
"""
import json
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
DT_REF = 0.0005
T_END = 4.0

R = {}


def order_quantile(scores, alpha=ALPHA):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return np.sort(scores)[k - 1]


def spread_max(pred):
    return (pred.max(axis=1) - pred.min(axis=1)).max(axis=1)   # (B,)


def binom_ci(k, n, z=1.96):
    p = k / n
    h = z * np.sqrt(p * (1 - p) / n)
    return p, (max(0, p - h), min(1, p + h))


def log(*a):
    print(*a, flush=True)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    # ================= main 39-bus model =================
    d = np.load("data39.npz")
    n = int(d["n_machines"]); te = float(d["t_end"])
    t = torch.as_tensor(d["t_out"], **F32, device=dev)[:, None]
    m = DeepONet(n + 1, n, p=512, t_scale=te).to(dev)
    m.load_state_dict(torch.load("model_p512.pt")); m.eval()
    fb, tc, y, st = d["fb_test"], d["tc_test"], d["y_test"], d["s_test"].astype(bool)
    with torch.no_grad():
        pred = m(severity_enc(fb, tc, d, dev), t).cpu().numpy()

    # ---- A2: max-over-time classification + confusion matrix ----
    sm = spread_max(pred)
    ps = sm < np.pi
    tp = int((ps & st).sum()); fp = int((ps & ~st).sum())
    fn = int((~ps & st).sum()); tn = int((~ps & ~st).sum())
    acc = (tp + tn) / len(st)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    log(f"A2 main: acc={acc:.4f}  TP/FP/FN/TN={tp}/{fp}/{fn}/{tn}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    R["acc"], R["confmat"], R["f1"] = acc, [tp, fp, fn, tn], f1

    # ---- D2: regional pooled RMSE (one-pass) ----
    err = pred - y                      # (960, n, T) signed
    es = err[st]
    mask_fault = t[:, 0].cpu().numpy()[None, :] < tc[st][:, None]   # (S, T): fault-on region
    es_f = es * mask_fault[:, None, :]
    es_p = es * (~mask_fault)[:, None, :]
    rmse_fault = float(np.sqrt((es_f ** 2).sum() / mask_fault.sum() / n))
    rmse_post = float(np.sqrt((es_p ** 2).sum() / (~mask_fault).sum() / n))
    rmse_all = float(np.sqrt((es ** 2).mean()))
    log(f"D2 regional pooled RMSE: fault-on={np.rad2deg(rmse_fault):.1f}  "
        f"post-fault={np.rad2deg(rmse_post):.1f}  full={np.rad2deg(rmse_all):.1f} deg")
    R["rmse_fault"], R["rmse_post"], R["rmse_all"] = map(lambda v: np.rad2deg(v),
                                                         [rmse_fault, rmse_post, rmse_all])

    # ---- D3: uniform-random predictor RMSE on the true test trajectories ----
    ur = np.random.default_rng(123).uniform(-np.pi, np.pi, size=es.shape)
    rmse_rand = float(np.sqrt(((ur - es) ** 2).mean()))
    rmse_zero = float(np.sqrt((es ** 2).mean()))
    log(f"D3 random predictor RMSE={np.rad2deg(rmse_rand):.1f} deg, zero predictor={np.rad2deg(rmse_zero):.1f} deg")
    R["rmse_rand"] = np.rad2deg(rmse_rand)

    # ---- A1/A4: conformal scores with correct protocol ----
    abs_err = np.abs(err)
    Rg = abs_err[st].max(axis=(1, 2))
    signed = err[st]
    diffs = np.abs(signed[:, None, :, :] - signed[:, :, None, :])
    R_pw = diffs.max(axis=(1, 2, 3))
    idx = rng.permutation(len(Rg)); nc = len(Rg) // 2
    cal_i, eva_i = idx[:nc], idx[nc:]
    q = order_quantile(Rg[cal_i])
    cov = (Rg[eva_i] <= q).mean()
    q_pw = order_quantile(R_pw[cal_i])
    cov_pw = (R_pw[eva_i] <= q_pw).mean()
    k_cov, ci_cov = binom_ci(int((Rg[eva_i] <= q).sum()), len(eva_i))
    log(f"A1 signed pairwise: q_pw={np.rad2deg(q_pw):.1f} deg, coverage={cov_pw:.3f} ({int((R_pw[eva_i]<=q_pw).sum())}/{len(eva_i)})")
    log(f"A4 per-machine: q={np.rad2deg(q):.1f} deg, coverage={k_cov:.3f} ({int((Rg[eva_i]<=q).sum())}/{len(eva_i)}), 95% CI {ci_cov[0]:.2f}-{ci_cov[1]:.2f}")
    R["q_deg"], R["cov"], R["cov_ci"] = np.rad2deg(q), k_cov, ci_cov
    R["q_pw_deg"], R["cov_pw"], R["n_cal"], R["n_eva"] = np.rad2deg(q_pw), cov_pw, nc, len(eva_i)

    # ---- B4: decidable fraction + wrong-release on the evaluation half ----
    st_e, ps_e, sm_e = st[eva_i], ps[eva_i], sm[eva_i]
    err_e = err[eva_i]
    gate = ps_e
    purity = st_e[gate].mean() if gate.sum() else np.nan
    wr = (~st_e & gate).sum() / (~st_e).sum() if (~st_e).sum() else np.nan
    decidable = sm_e + q_pw < np.pi          # rule: predicted spread + q_pw < pi
    dec_frac = decidable[gate].mean() if gate.sum() else np.nan
    wr_num = int((~st_e & gate).sum()); wr_den = int((~st_e).sum())
    log(f"B4 gate (eval half, n={len(eva_i)}): purity={purity:.3f}, "
        f"wrong-release={wr:.3f} ({wr_num}/{wr_den} of unstable), "
        f"decidable fraction among pred-stable={dec_frac:.3f}")
    R["gate_purity"], R["wr_rate"], R["wr_num"], R["wr_den"] = purity, wr, wr_num, wr_den
    R["dec_frac"] = dec_frac

    # ---- A3: gated coverage, protocol-correct ----
    q_os = q
    cov_all_ps = (abs_err[eva_i][gate].max(axis=(1, 2)) <= q_os).mean() if gate.sum() else np.nan
    in_gate = st_e & gate
    cov_ts_ps = (abs_err[eva_i][in_gate].max(axis=(1, 2)) <= q_os).mean() if in_gate.sum() else np.nan
    log(f"A3 gated coverage (eval half): pred-stable {cov_all_ps:.3f} (n={int(gate.sum())}), "
        f"true-stable&pred-stable (oracle diagnostic) {cov_ts_ps:.3f} (n={int(in_gate.sum())})")
    R["gate_cov_all"], R["gate_cov_ts"], R["gate_n"], R["gate_n_ts"] = \
        cov_all_ps, cov_ts_ps, int(gate.sum()), int(in_gate.sum())

    # ---- B6/B7: CCT reference (event-aligned, dt=0.5ms) + per-bus table + monotonicity ----
    model39 = ClassicalModel(case="39")
    buses = np.unique(fb)

    def true_cct(bus):
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            r = simulate_batch(model39, np.array([bus]), np.array([mid]), np.array([-1]), T_END,
                               dt=DT_REF)
            lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
        return lo

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

    # B7 monotonicity: dense sweep, first crossing from zero
    sweeps = []
    for b in buses:
        ts = np.linspace(0.01, 0.8, 159)
        with torch.no_grad():
            tr = m(severity_enc(np.full(len(ts), b), ts, d, dev), t).cpu().numpy()
        lab = spread_max(tr) < np.pi
        r = simulate_batch(model39, np.full(len(ts), b), ts, np.full(len(ts), -1), T_END,
                           dt=DT_REF)
        lab_t = r["stable"]
        flips = int((np.diff(lab.astype(int)) != 0).sum())
        flips_t = int((np.diff(lab_t.astype(int)) != 0).sum())
        first_un = int(np.argmax(~lab)) if not lab.all() else -1
        first_un_t = int(np.argmax(~lab_t)) if not lab_t.all() else -1
        sweeps.append((int(b), flips, flips_t, first_un, first_un_t))
    log("B7 monotonicity (bus, pred flips, ref flips, first-unstable pred idx, ref idx):")
    for s in sweeps:
        log("   ", s)
    R["sweeps"] = sweeps

    rows = []
    for b in buses:
        ct = true_cct(int(b)); cp = pred_cct(int(b))
        rows.append((int(b), round(ct * 1000, 1), round(cp * 1000, 1),
                     round((cp - ct) * 1000, 1), round((cp - ct) / ct * 100, 1)))
    e = np.array([r[3] for r in rows])
    log(f"B6 CCT (max-over-time, dt=0.5ms ref): mean={e.mean():.2f} |mean|={np.abs(e).mean():.2f} "
        f"max|err|={np.abs(e).max():.1f} ms")
    log("   per-bus (bus, ref ms, pred ms, signed err ms, rel %):")
    for r in rows:
        log("   ", r)
    R["cct_mean"], R["cct_mae"], R["cct_max"], R["cct_rows"] = \
        float(e.mean()), float(np.abs(e).mean()), float(np.abs(e).max()), rows

    # ---- B5: heuristic CCT band from the shifted criterion ----
    def pred_cct_shifted(bus, shift):
        def dec(tcc):
            with torch.no_grad():
                tr = m(severity_enc(np.array([bus]), np.array([tcc]), d, dev), t).cpu().numpy()
            return spread_max(tr)[0] < np.pi - shift
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if dec(mid) else (lo, mid)
        return lo
    band_rows = []
    for b in buses:
        c_hi = pred_cct_shifted(int(b), 2 * q_pw)     # conservative (smaller CCT)
        c_lo = pred_cct_shifted(int(b), -2 * q_pw)    # optimistic (larger CCT)
        band_rows.append((int(b), round(c_hi * 1000, 1), round(c_lo * 1000, 1),
                          round((c_lo - c_hi) * 1000, 1)))
    widths = [r[3] for r in band_rows]
    log(f"B5 CCT band (2*q_pw={np.rad2deg(2*q_pw):.1f} deg): widths(ms) mean={np.mean(widths):.1f} "
        f"min={np.min(widths):.1f} max={np.max(widths):.1f}")
    log("   per-bus (bus, conservative ms, optimistic ms, width ms):")
    for r in band_rows:
        log("   ", r)
    R["band_rows"], R["band_width_mean"], R["band_width_max"] = \
        band_rows, float(np.mean(widths)), float(np.max(widths))

    # ================= fair baselines: MLP / GRU conformal + CCT =================
    def fair_protocol(model, s_in, name):
        with torch.no_grad():
            pr = model(s_in, t).cpu().numpy()
        sm_ = spread_max(pr)
        acc_ = ((sm_ < np.pi) == st).mean()
        e_ = np.abs(pr - y)[st]
        Rg_ = e_.max(axis=(1, 2))
        signed_ = (pr - y)[st]
        diff_ = np.abs(signed_[:, None, :, :] - signed_[:, :, None, :])
        R_pw_ = diff_.max(axis=(1, 2, 3))
        rng2 = np.random.default_rng(0)
        idx2 = rng2.permutation(len(Rg_)); n2 = len(Rg_) // 2
        q_ = order_quantile(Rg_[idx2[:n2]]); cov_ = (Rg_[idx2[n2:]] <= q_).mean()
        q_pw_ = order_quantile(R_pw_[idx2[:n2]]); cov_pw_ = (R_pw_[idx2[n2:]] <= q_pw_).mean()

        def pcct(bus):
            def stb(tcc):
                with torch.no_grad():
                    tr = model(severity_enc(np.array([bus]), np.array([tcc]), d, dev), t).cpu().numpy()
                return spread_max(tr)[0] < np.pi
            lo, hi = 0.01, 0.8
            for _ in range(16):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if stb(mid) else (lo, mid)
            return lo
        errs = [abs(pcct(int(b)) - true_cct(int(b))) * 1000 for b in buses]
        log(f"C1 {name}: acc={acc_:.4f}  q={np.rad2deg(q_):.1f} (cov {cov_:.3f})  "
            f"q_pw={np.rad2deg(q_pw_):.1f} (cov {cov_pw_:.3f})  "
            f"CCT mean={np.mean(errs):.2f} ms max={np.max(errs):.1f} ms")
        return dict(acc=acc_, q=np.rad2deg(q_), cov=cov_, q_pw=np.rad2deg(q_pw_),
                    cov_pw=cov_pw_, cct_mean=float(np.mean(errs)), cct_max=float(np.max(errs)))

    s_te = severity_enc(fb, tc, d, dev)
    mlp = PointMLP(n + 1, n, t_scale=te).to(dev)
    mlp.load_state_dict(torch.load("model_mlp.pt")); mlp.eval()
    R["mlp"] = fair_protocol(mlp, s_te, "MLP")
    rnn = GRUTrajectory(n + 1, n, t_scale=te).to(dev)
    rnn.load_state_dict(torch.load("model_rnn.pt")); rnn.eval()
    R["gru"] = fair_protocol(rnn, s_te, "GRU")

    # ================= severity-MLP baselines (C3) =================
    from severity_mlp import train_severity_mlp
    R["sevmlp"] = train_severity_mlp(dev)
    log(f"C3 severity-MLP: acc={R['sevmlp']['acc']:.4f}  CCT mean={R['sevmlp']['cct_mean']:.2f} ms")

    # ================= transfers: one-axis / exciter / damped / 118 =================
    def eval_transfer(model_path, datafile, s_dim, K=16, name="", damped=False):
        dd = np.load(datafile)
        nn = int(dd["n_machines"]); tee = float(dd["t_end"])
        tt = torch.as_tensor(dd["t_out"], **F32, device=dev)[:, None]
        mm = DeepONet(s_dim, nn, p=512, K=K, t_scale=tee).to(dev)
        mm.load_state_dict(torch.load(model_path)); mm.eval()
        ss = severity_enc(dd["fb_test"], dd["tc_test"], dd, dev)
        with torch.no_grad():
            pr = mm(ss, tt).cpu().numpy()
        stt = dd["s_test"].astype(bool)
        sm_ = spread_max(pr)
        acc_ = ((sm_ < np.pi) == stt).mean()
        es_ = (pr - dd["y_test"])[stt]
        rmse_ = float(np.sqrt((es_ ** 2).mean()))
        Rg_ = np.abs(pr - dd["y_test"])[stt].max(axis=(1, 2))
        rng2 = np.random.default_rng(0)
        idx2 = rng2.permutation(len(Rg_)); n2 = len(Rg_) // 2
        q_ = order_quantile(Rg_[idx2[:n2]]); cov_ = (Rg_[idx2[n2:]] <= q_).mean()
        log(f"A2/transfer {name}: acc={acc_:.4f}  pooled RMSE={np.rad2deg(rmse_):.1f} deg  "
            f"q={np.rad2deg(q_):.1f} cov={cov_:.3f}  n_stable={int(stt.sum())}")
        return dict(acc=acc_, rmse=float(np.rad2deg(rmse_)), q=float(np.rad2deg(q_)),
                    cov=float(cov_), n_stable=int(stt.sum()))

    R["oneaxis"] = eval_transfer("model_oneaxis_p512.pt", "data39_oneaxis.npz", n + 1, name="one-axis")
    R["exciter"] = eval_transfer("model_exciter_p512.pt", "data39_exciter.npz", n + 1, name="exciter")
    R["damped"] = eval_transfer("model_damped_p512.pt", "data39_damped.npz", n + 1, name="damped")
    R["b118"] = eval_transfer("model_118.pt", "data118.npz", 55, name="118")

    # ================= N-1 / N-2 =================
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
        acc_ = ((sm_ < np.pi) == st_).mean()
        Rg_ = np.abs(pr - y_)[st_].max(axis=(1, 2))
        rng2 = np.random.default_rng(0)
        idx2 = rng2.permutation(len(Rg_)); n2 = len(Rg_) // 2
        q_ = order_quantile(Rg_[idx2[:n2]]); cov_ = (Rg_[idx2[n2:]] <= q_).mean()
        log(f"N-1 {name}: acc={acc_:.4f}  q={np.rad2deg(q_):.1f} cov={cov_:.3f} "
            f"(n_cal={n2}, n_eva={len(Rg_)-n2}, n_stable={int(st_.sum())})")
        return dict(acc=acc_, q=float(np.rad2deg(q_)), cov=float(cov_), n_cal=n2,
                    n_eva=len(Rg_) - n2, n_stable=int(st_.sum()))
    R["n1_bus"] = eval_n1("n1_p512", "bus", "held-out buses")
    R["n1_line"] = eval_n1("n1_p512", "line", "held-out lines")

    # N-2 (data regenerated by generate_data_n2.py with the split integrator)
    dn2 = np.load("data39_n2.npz")
    n2m = int(dn2["n_machines"]); te2 = float(dn2["t_end"]); tm2 = float(dn2["t_max"])
    t2 = torch.as_tensor(dn2["t_out"], **F32, device=dev)[:, None]
    m2 = DeepONet(2 * n2m + 1, n2m, p=512, t_scale=te2).to(dev)
    m2.load_state_dict(torch.load("model_n2_p512.pt")); m2.eval()
    sf2 = (dn2["Pm"][None, :] - dn2["pe_fault"][dn2["fb_testn2"] - 1]) / dn2["M"][None, :] * 10.0
    tcn2 = np.asarray(dn2["tc_testn2"], dtype=np.float32)[:, None] / tm2
    s2 = torch.cat([torch.as_tensor(sf2, **F32, device=dev),
                    torch.as_tensor(dn2["sev_testn2"], **F32, device=dev),
                    torch.as_tensor(tcn2, **F32, device=dev)], dim=1)
    with torch.no_grad():
        pr2 = m2(s2, t2).cpu().numpy()
    st2 = dn2["s_testn2"].astype(bool)
    acc2 = ((spread_max(pr2) < np.pi) == st2).mean()
    log(f"N-2 pairs: acc={acc2:.4f}  n_stable={int(st2.sum())}")
    R["n2"] = dict(acc=acc2, n_stable=int(st2.sum()))

    # ================= extras =================
    # C8: parameter-level perturbation (H and Pm) at 10%
    d8 = np.load("data39.npz")
    s8 = severity_enc(fb, tc, d8, dev)
    with torch.no_grad():
        p8 = m(s8, t).cpu().numpy()
    acc0 = ((spread_max(p8) < np.pi) == st).mean()
    for key, mag in (("M", 0.1), ("Pm", 0.1)):
        d9 = {k: d8[k] for k in d8.files}
        d9[key] = d8[key] * (1 + mag)
        s9 = severity_enc(fb, tc, d9, dev)
        with torch.no_grad():
            p9 = m(s9, t).cpu().numpy()
        acc9 = ((spread_max(p9) < np.pi) == st).mean()
        rmse9 = float(np.sqrt(((p9 - y)[st] ** 2).mean()))
        log(f"C8 {key}+{int(mag*100)}%: acc={acc9:.4f} (clean {acc0:.4f}), "
            f"pooled RMSE={np.rad2deg(rmse9):.1f} deg")
        R[f"c8_{key}"] = dict(acc=acc9, rmse=float(np.rad2deg(rmse9)))

    # B3: denser-grid check on a subset (2 ms grid for 200 scenarios)
    import time
    t0 = time.time()
    sub = np.arange(0, 200)
    out_dt = 0.002
    r_dense = simulate_batch(model39, fb[sub], tc[sub], np.full(len(sub), -1), T_END,
                             dt=DT_REF, out_dt=out_dt)
    flips = (r_dense["stable"] != st[sub]).sum()
    log(f"B3 denser grid (2 ms, n=200): label flips vs 10 ms labels = {flips}")
    R["b3_flips"] = int(flips)

    with open("recompute_all.json", "w") as f:
        json.dump(R, f, indent=1, default=float)
    log("wrote recompute_all.json")


if __name__ == "__main__":
    main()
