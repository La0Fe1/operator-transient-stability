"""IEEE 118-bus scalability experiment with AUTHORITATIVE dynamic data.

Uses the KIOS/Demetriou "IEEE 118-Bus Modified Test System" machine parameters
(H, x'd) extracted from IEEE-118.pdf, mapped to pandapower's case118 54 machines
by rated MVA (largest-to-largest). This replaces the earlier "typical values".
"""
from __future__ import annotations

import numpy as np
import torch

import pandapower as pp
import pandapower.networks as pn
from power_system import ClassicalModel
from dataset import simulate_batch

T_END = 4.0
DT = 0.0005
OUT_DT = 0.01
SEED = 0

# KIOS authoritative machine data: (rated MVA, H [s], x'd [p.u.]) — 54 machines.
# Generators (19), condensers (20), motors (15); expanded from the grouped table.
KIOS_MACHINES = (
    # generators
    [(590, 2.319, 0.280)] * 3 +
    [(125, 4.768, 0.174)] * 1 +
    [(330, 3.006, 0.317)] * 3 +
    [(410, 3.704, 0.2738)] * 1 +
    [(75, 6.187, 0.185)] * 3 +
    [(100, 4.985, 0.220)] * 3 +
    [(233, 4.122, 0.324)] * 2 +
    [(512, 2.631, 0.270)] * 2 +
    [(835, 2.6419, 0.413)] * 1 +
    # condensers (20)
    [(25, 1.200, 0.304)] * 18 +
    [(40, 1.520, 0.343)] * 2 +
    # motors (15)
    [(25, 5.016, 0.232)] * 5 +
    [(35.29, 4.4893, 0.231)] * 3 +
    [(51.2, 5.078, 0.209)] * 2 +
    [(75, 6.186, 0.185)] * 2 +
    [(100, 4.985, 0.220)] * 2 +
    [(384, 2.621, 0.324)] * 1
)
assert len(KIOS_MACHINES) == 54


def build_gendyn_118():
    net = pn.case118()
    # pandapower gens: (bus0, sn_mva); ext_grid: bus0
    gens = [(int(g["bus"]), float(g["sn_mva"])) for _, g in net.gen.iterrows()]
    slack = [int(e["bus"]) for _, e in net.ext_grid.iterrows()]

    # sort pandapower gens by MVA (descending), match to sorted KIOS machines
    kios = sorted(KIOS_MACHINES, key=lambda x: -x[0])   # (MVA, H, xd) largest first
    gsorted = sorted(gens, key=lambda x: -x[1])         # (bus0, MVA) largest first

    gendyn = {}
    # 53 gens matched to the 53 largest-MVA KIOS machines (by MVA)
    for i, (b0, _) in enumerate(gsorted):
        _, H, xd = kios[i]
        gendyn[b0 + 1] = (H, xd)
    # slack gets the smallest remaining KIOS machine
    _, H, xd = kios[-1]
    for b0 in slack:
        gendyn[b0 + 1] = (H, xd)
    return gendyn


def main():
    gendyn = build_gendyn_118()
    print(f"118-bus authoritative machines: {len(gendyn)}")
    model = ClassicalModel(case="118", gendyn=gendyn)
    n = model.n
    print(f"machines n={n}, total Pm (MW)={model.Pm.sum()*100:.0f}")
    n_bus = 118

    pe_fault = np.array([model.fault_pe(b) for b in range(1, n_bus + 1)])

    rng = np.random.default_rng(SEED)
    all_buses = np.arange(1, n_bus + 1); rng.shuffle(all_buses)
    test_buses = all_buses[:24]; tr_buses = all_buses[24:]

    def draw(buses, n_tc, seed):
        rr = np.random.default_rng(seed)
        fb = np.repeat(buses, n_tc); tc = rr.uniform(0.005, 0.2, size=len(fb))
        idx = rr.permutation(len(fb)); return fb[idx], tc[idx]

    fb_tr, tc_tr = draw(tr_buses, 40, 1000)
    fb_te, tc_te = draw(test_buses, 40, 1002)
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
    print(f"saved data118.npz (train stable {tr['stable'].mean():.2f}, test stable {te['stable'].mean():.2f})")


if __name__ == "__main__":
    main()
