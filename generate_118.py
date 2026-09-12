"""IEEE 118-bus scalability experiment.

Builds a classical model of the 118-bus system (54 machines: 53 pandapower gens
+ 1 slack) with documented typical inertia values (H=5 s / x'd=0.2 p.u. for
generating units, H=1 s / x'd=0.3 p.u. for synchronous condensers), generates
transient-stability data, and trains the operator to demonstrate scalability
from 10 to 54 machines.
"""
from __future__ import annotations

import numpy as np
import torch

import pandapower as pp
import pandapower.networks as pn
from power_system import ClassicalModel
from dataset import simulate_batch

T_END = 4.0
DT = 0.001
OUT_DT = 0.01
SEED = 0


def build_gendyn_118():
    """Assign H/x'd per 1-based bus using documented typical values."""
    net = pn.case118()
    gendyn = {}
    for _, g in net.gen.iterrows():
        b1 = int(g["bus"]) + 1
        if g["p_mw"] > 0:
            gendyn[b1] = (5.0, 0.2)      # generating unit
        else:
            gendyn[b1] = (1.0, 0.3)      # synchronous condenser
    for _, e in net.ext_grid.iterrows():
        gendyn[int(e["bus"]) + 1] = (5.0, 0.2)   # slack
    return gendyn


def main():
    gendyn = build_gendyn_118()
    print(f"118-bus machines: {len(gendyn)}")
    model = ClassicalModel(case="118", gendyn=gendyn)
    n = model.n
    print(f"machines n={n}, total Pm (MW)={model.Pm.sum()*100:.0f}")
    n_bus = 118

    # severity encoding: pe_fault per fault bus
    pe_fault = np.array([model.fault_pe(b) for b in range(1, n_bus + 1)])  # (118, n)

    rng = np.random.default_rng(SEED)
    all_buses = np.arange(1, n_bus + 1); rng.shuffle(all_buses)
    test_buses = all_buses[:24]; tr_buses = all_buses[24:]

    def draw(buses, n_tc, seed):
        rr = np.random.default_rng(seed)
        fb = np.repeat(buses, n_tc); tc = rr.uniform(0.03, 0.55, size=len(fb))
        idx = rr.permutation(len(fb)); return fb[idx], tc[idx]

    fb_tr, tc_tr = draw(tr_buses, 15, 1000)   # 94 buses x 15
    fb_te, tc_te = draw(test_buses, 15, 1002)  # 24 buses x 15
    tl = np.full(len(fb_tr), -1); tl_te = np.full(len(fb_te), -1)

    print(f"generating train {len(fb_tr)} / test {len(fb_te)} ...")
    tr = simulate_batch(model, fb_tr, tc_tr, tl, T_END, dt=DT, out_dt=OUT_DT)
    te = simulate_batch(model, fb_te, tc_te, tl_te, T_END, dt=DT, out_dt=OUT_DT)

    np.savez_compressed(
        "data118.npz",
        fb_train=fb_tr, tc_train=tc_tr, y_train=tr["delta_coi"], s_train=tr["stable"],
        fb_test=fb_te, tc_test=tc_te, y_test=te["delta_coi"], s_test=te["stable"],
        t_out=np.arange(0.0, T_END + OUT_DT / 2, OUT_DT),
        E=model.E, Pm=model.Pm, M=model.M, G=np.real(model.yint_pre), B=np.imag(model.yint_pre),
        Mtot=model.Mtot, delta0=model.delta0, pe_fault=pe_fault,
        n_buses=n_bus, n_machines=n, t_end=T_END, out_dt=OUT_DT, t_max=0.6,
    )
    print(f"saved data118.npz  (train stable {tr['stable'].mean():.2f}, test stable {te['stable'].mean():.2f})")


if __name__ == "__main__":
    main()
