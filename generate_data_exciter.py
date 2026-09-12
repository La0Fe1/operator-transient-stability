"""Generate exciter-model (one-axis + first-order AVR) TSA data and train."""
from __future__ import annotations

import numpy as np

from power_system_oneaxis import ExciterModel

T_END = 4.0
DT = 0.0005
OUT_DT = 0.01
SEED = 0


def simulate_batch_exc(model, fault_buses, t_clears, t_end, dt, out_dt):
    B = len(fault_buses); n = model.n
    yf_map = {int(fb): model.fault_yint(int(fb)) for fb in np.unique(fault_buses)}
    Yf = np.stack([yf_map[int(fb)] for fb in fault_buses])
    Yp = np.tile(model.yint_pre[None], (B, 1, 1))

    Pm = model.Pm[None, :]; M = model.M[None, :]
    Td0 = model.Td0[None, :]; xdm = (model.xd_s - model.xd)[None, :]
    xd = model.xd[None, :]; Ka = model.Ka; Ta = model.Ta
    Vref = model.Vref[None, :]; rate_max = model.rate_max
    Efd_min = model.Efd_min; Efd_max = model.Efd_max
    ws = model.ws; D = model.damping; Mtot = model.Mtot

    delta = np.tile(model.delta0[None, :], (B, 1)); omega = np.ones((B, n))
    Eq = np.tile(model.Eq0[None, :], (B, 1)); Efd = np.tile(model.Efd0[None, :], (B, 1))

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
        defd = (Ka * (Vref - Vt) - ef) / Ta
        defd = np.clip(defd, -rate_max, rate_max)
        defd = np.where((ef <= Efd_min) & (defd < 0), 0.0, defd)
        defd = np.where((ef >= Efd_max) & (defd > 0), 0.0, defd)
        return dd, dw, de, defd

    out_t = np.arange(0.0, t_end + out_dt / 2, out_dt)
    n_out = out_t.size
    delta_out = np.zeros((B, n, n_out)); stable = np.ones(B, dtype=bool)
    step_per_out = max(1, int(round(out_dt / dt)))
    k = 0; k_out = 0; t_cur = 0.0
    while t_cur < t_end + dt / 2:
        if k % step_per_out == 0 and k_out < n_out:
            coi = delta - (M * delta).sum(axis=1, keepdims=True) / Mtot
            delta_out[:, :, k_out] = coi
            stable &= ((coi.max(axis=1) - coi.min(axis=1)) < np.pi)
            k_out += 1
        if t_cur >= t_end:
            break
        Ynow = np.where((t_cur < t_clears)[:, None, None], Yf, Yp)
        dd1, dw1, de1, df1 = deriv(delta, omega, Eq, Efd, Ynow)
        dd2, dw2, de2, df2 = deriv(delta + .5 * dt * dd1, omega + .5 * dt * dw1, Eq + .5 * dt * de1, Efd + .5 * dt * df1, Ynow)
        dd3, dw3, de3, df3 = deriv(delta + .5 * dt * dd2, omega + .5 * dt * dw2, Eq + .5 * dt * de2, Efd + .5 * dt * df2, Ynow)
        dd4, dw4, de4, df4 = deriv(delta + dt * dd3, omega + dt * dw3, Eq + dt * de3, Efd + dt * df3, Ynow)
        delta += (dt / 6) * (dd1 + 2 * dd2 + 2 * dd3 + dd4)
        omega += (dt / 6) * (dw1 + 2 * dw2 + 2 * dw3 + dw4)
        Eq += (dt / 6) * (de1 + 2 * de2 + 2 * de3 + de4)
        Efd = np.clip(Efd + (dt / 6) * (df1 + 2 * df2 + 2 * df3 + df4), Efd_min, Efd_max)
        t_cur += dt; k += 1
    return {"t": out_t, "delta_coi": delta_out, "stable": stable}


def main():
    model = ExciterModel(case="39")
    n = model.n; n_bus = 39
    pe_fault = np.array([model._stator(model.Eq0, model.delta0, model.fault_yint(b))[2]
                         for b in range(1, n_bus + 1)])

    rng = np.random.default_rng(SEED)
    all_buses = np.arange(1, n_bus + 1); rng.shuffle(all_buses)
    test_buses = all_buses[:12]; tr_buses = all_buses[12:]

    def draw(buses, n_tc, seed):
        rr = np.random.default_rng(seed)
        fb = np.repeat(buses, n_tc); tc = rr.uniform(0.03, 0.5, size=len(fb))
        idx = rr.permutation(len(fb)); return fb[idx], tc[idx]

    fb_tr, tc_tr = draw(tr_buses, 80, 1000)
    fb_va, tc_va = draw(tr_buses, 20, 1001)
    fb_te, tc_te = draw(test_buses, 80, 1002)

    def run(fb, tc): return simulate_batch_exc(model, fb, tc, T_END, DT, OUT_DT)
    print(f"generating train {len(fb_tr)} / val {len(fb_va)} / test {len(fb_te)} ...")
    tr = run(fb_tr, tc_tr); va = run(fb_va, tc_va); te = run(fb_te, tc_te)

    np.savez_compressed(
        "data39_exciter.npz",
        fb_train=fb_tr, tc_train=tc_tr, y_train=tr["delta_coi"], s_train=tr["stable"],
        fb_val=fb_va, tc_val=tc_va, y_val=va["delta_coi"], s_val=va["stable"],
        fb_test=fb_te, tc_test=tc_te, y_test=te["delta_coi"], s_test=te["stable"],
        t_out=np.arange(0.0, T_END + OUT_DT / 2, OUT_DT),
        E=model.Eq0, Pm=model.Pm, M=model.M, G=np.real(model.yint_pre), B=np.imag(model.yint_pre),
        Mtot=model.Mtot, delta0=model.delta0, pe_fault=pe_fault,
        n_buses=n_bus, n_machines=n, t_end=T_END, out_dt=OUT_DT, t_max=0.6,
    )
    print("saved data39_exciter.npz")
    print("train stable %.2f, test stable %.2f" % (tr["stable"].mean(), te["stable"].mean()))


if __name__ == "__main__":
    main()
