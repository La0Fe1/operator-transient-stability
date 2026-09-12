"""Boundary-aware sampling: concentrate t_clear near each bus's CCT.

Samples t_clear in [0.4 x CCT, 1.2 x CCT] per fault bus, giving many near-boundary
training examples (both stable and marginally unstable), which is where the
trajectory operator needs to be accurate for CCT estimation.
"""
from __future__ import annotations

import numpy as np

from power_system import ClassicalModel
from dataset import simulate_batch

T_END = 4.0
DT = 0.0005
OUT_DT = 0.01
SEED = 0


def compute_cct(model, bus, hi=0.8):
    lo = 0.005
    for _ in range(16):
        mid = (lo + hi) / 2
        r = simulate_batch(model, np.array([bus]), np.array([mid]), np.array([-1]), 4.0, dt=0.001, out_dt=0.02)
        lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
    return lo


def main():
    model = ClassicalModel(case="39")
    n = model.n; n_bus = 39
    # CCT per bus
    cct = np.array([compute_cct(model, b) for b in range(1, n_bus + 1)])
    print("CCT range:", cct.min().round(3), "-", cct.max().round(3))

    pe_fault = np.array([model.fault_pe(b) for b in range(1, n_bus + 1)])

    rng = np.random.default_rng(SEED)
    all_buses = np.arange(1, n_bus + 1); rng.shuffle(all_buses)
    test_buses = all_buses[:12]; tr_buses = all_buses[12:]

    def draw(buses, n_tc, seed):
        rr = np.random.default_rng(seed)
        fb = np.repeat(buses, n_tc)
        # sample t_clear in [0.4, 1.2] x CCT per bus
        lo = 0.4 * cct[fb - 1]; hi = 1.2 * cct[fb - 1]
        tc = lo + rr.uniform(0, 1, size=len(fb)) * (hi - lo)
        idx = rr.permutation(len(fb))
        return fb[idx], tc[idx]

    fb_tr, tc_tr = draw(tr_buses, 80, 1000)
    fb_va, tc_va = draw(tr_buses, 20, 1001)
    fb_te, tc_te = draw(test_buses, 80, 1002)
    tl = np.full(len(fb_tr), -1); tl_va = np.full(len(fb_va), -1); tl_te = np.full(len(fb_te), -1)

    def run(fb, tc, tl):
        return simulate_batch(model, fb, tc, tl, T_END, dt=DT, out_dt=OUT_DT)

    print("generating ...")
    tr = run(fb_tr, tc_tr, tl)
    va = run(fb_va, tc_va, tl_va)
    te = run(fb_te, tc_te, tl_te)

    np.savez_compressed(
        "data39_boundary.npz",
        fb_train=fb_tr, tc_train=tc_tr, y_train=tr["delta_coi"], s_train=tr["stable"],
        fb_val=fb_va, tc_val=tc_va, y_val=va["delta_coi"], s_val=va["stable"],
        fb_test=fb_te, tc_test=tc_te, y_test=te["delta_coi"], s_test=te["stable"],
        t_out=np.arange(0.0, T_END + OUT_DT / 2, OUT_DT),
        E=model.E, Pm=model.Pm, M=model.M, G=np.real(model.yint_pre), B=np.imag(model.yint_pre),
        Mtot=model.Mtot, delta0=model.delta0, pe_fault=pe_fault,
        n_buses=n_bus, n_machines=n, t_end=T_END, out_dt=OUT_DT, t_max=0.6,
    )
    print(f"saved data39_boundary.npz (train stable {tr['stable'].mean():.2f}, test stable {te['stable'].mean():.2f})")


if __name__ == "__main__":
    main()
