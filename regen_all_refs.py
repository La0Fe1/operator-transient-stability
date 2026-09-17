# -*- coding: utf-8 -*-
"""B6: regenerate the TEST reference (y/s) of every auxiliary dataset with the
event-split integrator at dt=0.5 ms. The classical/damped/boundary/N-1/118/post
datasets use dataset.simulate_batch (already fixed); one-axis, exciter, and N-2
have local simulators, reproduced here with the same event-split pattern.
Each npz is backed up to <name>_preB6.npz before rewriting."""
import shutil
import numpy as np

from dataset import simulate_batch
from power_system import ClassicalModel
from power_system_oneaxis import OneAxisModel, ExciterModel


# ---------- shared split-integration helper ----------
def _split_loop(n_out, out_t, B, n, M, Mtot, t_clears, Yf, Yp, delta, omega, extra_states,
                deriv):
    """RK4 with step split at t_c and linear interpolation onto the output grid."""
    delta_out = np.zeros((B, n, n_out))
    stable = np.ones(B, dtype=bool)
    delta_out[:, :, 0] = delta - (M * delta).sum(axis=1, keepdims=True) / Mtot
    t_cur = np.zeros(B)
    prev = [np.zeros_like(x) for x in [delta, omega, *extra_states]]
    prev_t = np.zeros(B)
    states = [delta, omega, *extra_states]
    for k_out in range(1, n_out):
        target = out_t[k_out]
        while (t_cur < target - 1e-15).any():
            step_mask = t_cur < target - 1e-15
            for p, s in zip(prev, states):
                p[step_mask] = s[step_mask]
            prev_t[step_mask] = t_cur[step_mask]
            fault_on = step_mask & (t_cur < t_clears)
            h = np.where(fault_on, np.minimum(0.0005, t_clears - t_cur), 0.0005)
            Ynow = np.where(fault_on[:, None, None], Yf, Yp)
            k1 = deriv(*states, Ynow)
            k2 = deriv(*[s + 0.5 * h[:, None] * kk for s, kk in zip(states, k1)], Ynow)
            k3 = deriv(*[s + 0.5 * h[:, None] * kk for s, kk in zip(states, k2)], Ynow)
            k4 = deriv(*[s + h[:, None] * kk for s, kk in zip(states, k3)], Ynow)
            for s, (a, b, c, d) in zip(states, zip(k1, k2, k3, k4)):
                s[step_mask] += (h[step_mask, None] / 6.0) * (a + 2 * b + 2 * c + d)[step_mask]
            t_cur[step_mask] += h[step_mask]
        hw = t_cur - prev_t
        alpha = np.where(hw > 0, (target - prev_t) / hw, 0.0)[:, None]
        coi = prev[0] + alpha * (delta - prev[0])
        coi = coi - (M * coi).sum(axis=1, keepdims=True) / Mtot
        delta_out[:, :, k_out] = coi
        stable &= (coi.max(1) - coi.min(1)) < np.pi
    return delta_out, stable


def sim_oneaxis(model, fb, tc, t_end=4.0):
    B = len(fb); n = model.n
    yf_map = {int(b): model.fault_yint(int(b)) for b in np.unique(fb)}
    Yf = np.stack([yf_map[int(b)] for b in fb])
    Yp = np.tile(model.yint_pre[None], (B, 1, 1))
    Eq0 = model.Eq0[None, :]; Pm = model.Pm[None, :]; M = model.M[None, :]
    Efd = model.Efd[None, :]; Td0 = model.Td0[None, :]; xdm = (model.xd_s - model.xd)[None, :]
    ws = model.ws; D = model.damping; Mtot = model.Mtot
    delta = np.tile(model.delta0[None, :], (B, 1)); omega = np.ones((B, n))
    Eq = np.tile(model.Eq0[None, :], (B, 1))
    out_t = np.arange(0.0, t_end + 0.005, 0.01)

    def deriv(d, w, e, Y):
        Ec = e * np.exp(1j * d)
        I = np.einsum("bij,bj->bi", Y, Ec)
        Iq = (I * np.exp(-1j * d)).real
        Id = -(I * np.exp(-1j * d)).imag
        Pe = e * Iq
        return ws * (w - 1.0), (Pm - Pe - D * (w - 1.0)) / M, (Efd - e - xdm * Id) / Td0

    y, s = _split_loop(len(out_t), out_t, B, n, M, Mtot, tc, Yf, Yp, delta, omega, [Eq], deriv)
    return {"t": out_t, "delta_coi": y, "stable": s}


def sim_exciter(model, fb, tc, t_end=4.0):
    B = len(fb); n = model.n
    yf_map = {int(b): model.fault_yint(int(b)) for b in np.unique(fb)}
    Yf = np.stack([yf_map[int(b)] for b in fb])
    Yp = np.tile(model.yint_pre[None], (B, 1, 1))
    Pm = model.Pm[None, :]; M = model.M[None, :]
    Td0 = model.Td0[None, :]; xdm = (model.xd_s - model.xd)[None, :]; xd = model.xd[None, :]
    Ka = model.Ka; Ta = model.Ta; Vref = model.Vref[None, :]
    rate_max = model.rate_max; Efd_min = model.Efd_min; Efd_max = model.Efd_max
    ws = model.ws; D = model.damping; Mtot = model.Mtot
    delta = np.tile(model.delta0[None, :], (B, 1)); omega = np.ones((B, n))
    Eq = np.tile(model.Eq0[None, :], (B, 1)); Efd = np.tile(model.Efd0[None, :], (B, 1))
    out_t = np.arange(0.0, t_end + 0.005, 0.01)

    def deriv(d, w, e, ef, Y):
        Ec = e * np.exp(1j * d)
        I = np.einsum("bij,bj->bi", Y, Ec)
        Iq = (I * np.exp(-1j * d)).real
        Id = -(I * np.exp(-1j * d)).imag
        Pe = e * Iq
        Vt = np.abs(Ec - 1j * xd * I)
        dd = ws * (w - 1.0)
        dw = (Pm - Pe - D * (w - 1.0)) / M
        de = (ef - e - xdm * Id) / Td0
        df = np.clip((Ka * (Vref - Vt) - ef) / Ta, -rate_max, rate_max)
        df = np.where((ef <= Efd_min) & (df < 0), 0.0, df)
        df = np.where((ef >= Efd_max) & (df > 0), 0.0, df)
        return dd, dw, de, df

    y, s = _split_loop(len(out_t), out_t, B, n, M, Mtot, tc, Yf, Yp, delta, omega,
                       [Eq, Efd], deriv)
    return {"t": out_t, "delta_coi": y, "stable": s}


def sim_multi(model, fb, trips, tc, t_end=4.0):
    B = len(fb); n = model.n
    yf = np.stack([model.fault_yint(int(fb[i])) for i in range(B)])
    yp = np.stack([model.post_yint_multi(list(trips[i])) for i in range(B)])
    Pm = model.Pm[None, :]; M = model.M[None, :]; ws = model.ws
    D = model.damping; Mtot = model.Mtot
    delta = np.tile(model.delta0[None, :], (B, 1)); omega = np.ones((B, n))
    out_t = np.arange(0.0, t_end + 0.005, 0.01)

    def deriv(d, w, Y):
        Ec = model.E[None, :] * np.exp(1j * d)
        pe = (Ec * np.conj(np.einsum("bij,bj->bi", Y, Ec))).real
        return ws * (w - 1.0), (Pm - pe - D * (w - 1.0)) / M

    y, s = _split_loop(len(out_t), out_t, B, n, M, Mtot, tc, yf, yp, delta, omega, [], deriv)
    return {"t": out_t, "delta_coi": y, "stable": s}


def regen_classical(npz, model):
    import os
    if os.path.exists(npz.replace(".npz", "_preB6.npz")):
        print(f"{npz}: backup exists, skipping (already regenerated)", flush=True)
        return
    d = np.load(npz)
    fb, tc = d["fb_test"], d["tc_test"]
    out_dt = float(d["t_out"][1] - d["t_out"][0])
    y, s = None, np.zeros(len(fb), dtype=bool)
    parts = []
    B = 120
    for i in range(0, len(fb), B):
        r = simulate_batch(model, fb[i:i+B], tc[i:i+B], np.full(min(B, len(fb)-i), -1),
                           4.0, dt=0.0005, out_dt=out_dt)
        parts.append(r["delta_coi"]); s[i:i+B] = r["stable"]
    y = np.concatenate(parts)
    flips = (s != d["s_test"].astype(bool)).sum()
    shutil.copy(npz, npz.replace(".npz", "_preB6.npz"))
    out = {k: d[k] for k in d.files}
    out["y_test"], out["s_test"] = y, s
    np.savez(npz, **out)
    print(f"{npz}: test label flips {flips}/{len(fb)}", flush=True)


def regen_n1():
    d = np.load("data39_n1.npz")
    model = ClassicalModel(case="39")
    out = {k: d[k] for k in d.files}
    for split in ("b", "l"):
        fb = d[f"fb_test{split}"]; tl = d[f"tl_test{split}"]; tc = d[f"tc_test{split}"]
        parts, s = [], np.zeros(len(fb), dtype=bool)
        B = 120
        for i in range(0, len(fb), B):
            r = simulate_batch(model, fb[i:i+B], tc[i:i+B], tl[i:i+B], 4.0, dt=0.0005)
            parts.append(r["delta_coi"]); s[i:i+B] = r["stable"]
        flips = (s != d[f"s_test{split}"].astype(bool)).sum()
        out[f"y_test{split}"], out[f"s_test{split}"] = np.concatenate(parts), s
        print(f"n1 {split}: flips {flips}/{len(fb)}", flush=True)
    shutil.copy("data39_n1.npz", "data39_n1_preB6.npz")
    np.savez("data39_n1.npz", **out)


def main():
    # classical-family datasets via the fixed dataset.simulate_batch
    regen_classical("data39_damped.npz", ClassicalModel(case="39", damping=1.0))
    regen_classical("data39_boundary.npz", ClassicalModel(case="39"))

    # 118-bus: same custom model class used to generate the data
    from generate_118_authoritative import build_gendyn_118
    model118 = ClassicalModel(case="118", gendyn=build_gendyn_118())
    regen_classical("data118.npz", model118)

    # one-axis / exciter
    for npz, cls, sim in (("data39_oneaxis.npz", OneAxisModel, sim_oneaxis),
                          ("data39_exciter.npz", ExciterModel, sim_exciter)):
        dd = np.load(npz)
        model = cls(case="39")
        fb, tc = dd["fb_test"], dd["tc_test"]
        parts, s = [], np.zeros(len(fb), dtype=bool)
        B = 120
        for i in range(0, len(fb), B):
            r = sim(model, fb[i:i+B], tc[i:i+B])
            parts.append(r["delta_coi"]); s[i:i+B] = r["stable"]
        flips = (s != dd["s_test"].astype(bool)).sum()
        shutil.copy(npz, npz.replace(".npz", "_preB6.npz"))
        out = {k: dd[k] for k in dd.files}
        out["y_test"], out["s_test"] = np.concatenate(parts), s
        np.savez(npz, **out)
        print(f"{npz}: flips {flips}/{len(fb)}", flush=True)

    regen_n1()


if __name__ == "__main__":
    main()
