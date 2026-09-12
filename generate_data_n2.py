"""Generate N-0 + N-1 + N-2 data with the multi-line post-fault severity encoding.

The post-fault severity vector (Pm - Pe_post)/M is computed for 0/1/2 tripped
lines and stored per scenario, so the operator can generalize across topologies
of any (up to N-2) line-trip cardinality.
"""
from __future__ import annotations

import numpy as np

from power_system import ClassicalModel
from dataset import simulate_batch

T_END = 4.0
DT = 0.0005
OUT_DT = 0.01
SEED = 0


def main():
    model = ClassicalModel(case="39")
    n = model.n; n_bus = 39; n_line = len(model.net.line)

    pe_fault = np.array([model.fault_pe(b) for b in range(1, n_bus + 1)])  # (39, n)
    cache = {}

    def sev_post(trip_lines):
        key = tuple(sorted(l for l in trip_lines if l >= 0))
        if key not in cache:
            pe = model._pe(model.delta0, model.post_yint_multi(list(key)))
            cache[key] = (model.Pm - pe) / model.M * 10.0
        return cache[key]

    rng = np.random.default_rng(SEED)
    all_buses = np.arange(1, n_bus + 1); rng.shuffle(all_buses)
    all_lines = np.arange(n_line); rng.shuffle(all_lines)
    test_buses = all_buses[:12]; tr_buses = all_buses[12:]
    test_lines = all_lines[:10]; tr_lines = all_lines[10:]

    # line pairs for N-2 (train pairs vs held-out pairs)
    pairs = [(a, b) for i, a in enumerate(all_lines) for b in all_lines[i + 1:]]
    rng.shuffle(pairs)
    test_pairs = pairs[:150]; tr_pairs = pairs[150:]

    def draw(trip_sets, buses, n_total, seed):
        rr = np.random.default_rng(seed)
        fb = rr.choice(buses, n_total)
        tr = [trip_sets[i] for i in rr.integers(0, len(trip_sets), n_total)]
        tc = rr.uniform(0.03, 0.5, size=n_total)
        return fb, tr, tc

    # train = N-0 + N-1 + N-2
    n0 = [( ), ] * 1
    fb0, tr0, tc0 = draw([()], tr_buses, 27 * 60, 1000)
    fb1, tr1, tc1 = draw([(l,) for l in tr_lines], tr_buses, 25 * 27 * 8, 1001)
    fb2, tr2, tc2 = draw(tr_pairs, tr_buses, 27 * 8 * 8, 1002)
    fb_tr = np.concatenate([fb0, fb1, fb2]); tr_tr = tr0 + tr1 + tr2; tc_tr = np.concatenate([tc0, tc1, tc2])

    # test N-2 (held-out pairs) + held-out buses N-0
    fb_t2, tr_t2, tc_t2 = draw(test_pairs, tr_buses, 27 * 6 * 6, 1003)
    fb_tb, tr_tb, tc_tb = draw([()], test_buses, 12 * 60, 1004)

    def sev_matrix(trips):
        return np.stack([sev_post(t) for t in trips])

    sev_tr = sev_matrix(tr_tr)
    sev_t2 = sev_matrix(tr_t2)
    sev_tb = sev_matrix(tr_tb)

    def run(fb, trips, tc):
        return simulate_batch_multi(model, fb, trips, tc, T_END, DT, OUT_DT)

    print(f"generating train {len(fb_tr)} / test_N2 {len(fb_t2)} / test_bus {len(fb_tb)} ...")
    tr = run(fb_tr, tr_tr, tc_tr)
    t2 = run(fb_t2, tr_t2, tc_t2)
    tb = run(fb_tb, tr_tb, tc_tb)

    np.savez_compressed(
        "data39_n2.npz",
        fb_train=fb_tr, tc_train=tc_tr, sev_train=sev_tr, y_train=tr["delta_coi"], s_train=tr["stable"],
        fb_testn2=fb_t2, tc_testn2=tc_t2, sev_testn2=sev_t2, y_testn2=t2["delta_coi"], s_testn2=t2["stable"],
        fb_testb=fb_tb, tc_testb=tc_tb, sev_testb=sev_tb, y_testb=tb["delta_coi"], s_testb=tb["stable"],
        t_out=np.arange(0.0, T_END + OUT_DT / 2, OUT_DT),
        Pm=model.Pm, M=model.M, pe_fault=pe_fault, n_buses=n_bus, n_machines=n,
        t_end=T_END, out_dt=OUT_DT, t_max=0.6,
    )
    print("saved data39_n2.npz")
    print("train stable %.2f, test_N2 stable %.2f" % (tr["stable"].mean(), t2["stable"].mean()))


def simulate_batch_multi(model, fb, trips, tc, t_end, dt, out_dt):
    """Batched simulation where each scenario can trip a different line set."""
    B = len(fb); n = model.n
    yf = np.stack([model.fault_yint(int(fb[i])) for i in range(B)])
    yp = np.stack([model.post_yint_multi(list(trips[i])) for i in range(B)])
    Pm = model.Pm[None, :]; M = model.M[None, :]; ws = model.ws; D = model.damping; Mtot = model.Mtot

    delta = np.tile(model.delta0[None, :], (B, 1)); omega = np.ones((B, n))

    def pe(d, Y):
        Ec = model.E[None, :] * np.exp(1j * d)
        return (Ec * np.conj(np.einsum("bij,bj->bi", Y, Ec))).real

    out_t = np.arange(0.0, t_end + out_dt / 2, out_dt)
    n_out = out_t.size
    delta_out = np.zeros((B, n, n_out)); stable = np.ones(B, dtype=bool)
    step_per_out = max(1, int(round(out_dt / dt)))
    k = 0; k_out = 0; t_cur = 0.0
    while t_cur < t_end + dt / 2:
        if k % step_per_out == 0 and k_out < n_out:
            coi = delta - (M * delta).sum(axis=1, keepdims=True) / Mtot
            delta_out[:, :, k_out] = coi
            stable &= ((coi.max(1) - coi.min(1)) < np.pi)
            k_out += 1
        if t_cur >= t_end:
            break
        Ynow = np.where((t_cur < tc)[:, None, None], yf, yp)
        def deriv(d, w, Y):
            dd = ws * (w - 1.0); dw = (Pm - pe(d, Y) - D * (w - 1.0)) / M
            return dd, dw
        d1, w1 = deriv(delta, omega, Ynow)
        d2, w2 = deriv(delta + .5 * dt * d1, omega + .5 * dt * w1, Ynow)
        d3, w3 = deriv(delta + .5 * dt * d2, omega + .5 * dt * w2, Ynow)
        d4, w4 = deriv(delta + dt * d3, omega + dt * w3, Ynow)
        delta += (dt / 6) * (d1 + 2 * d2 + 2 * d3 + d4)
        omega += (dt / 6) * (w1 + 2 * w2 + 2 * w3 + w4)
        t_cur += dt; k += 1
    return {"t": out_t, "delta_coi": delta_out, "stable": stable}


if __name__ == "__main__":
    main()
