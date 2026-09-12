"""Generate the transient-stability dataset (IEEE 39-bus) and cache to .npz.

Train/val fault buses are disjoint from the test fault buses so the test set
evaluates generalization to *unseen fault locations* (not just unseen samples).
"""
from __future__ import annotations

import numpy as np

from power_system import ClassicalModel
from dataset import simulate_batch, encode_scenario

# --- configuration ----------------------------------------------------------
T_END = 4.0          # simulation horizon [s]
DT = 0.0005          # reference integration step (0.5 ms)
OUT_DT = 0.01        # output resolution for the operator (10 ms)
T_MAX = 0.6          # normalization constant for clearing time
N_TRAIN_TC = 80      # clearing-time samples per fault bus
N_TEST_TC = 80
SEED = 0


def main():
    model = ClassicalModel(case="39")
    n_buses = model.N                      # 39
    n = model.n                            # 10 machines
    t_out = np.arange(0.0, T_END + OUT_DT / 2, OUT_DT)

    rng = np.random.default_rng(SEED)
    all_buses = np.arange(1, n_buses + 1)
    rng.shuffle(all_buses)
    n_test_buses = 12
    test_buses = all_buses[:n_test_buses]
    trval_buses = all_buses[n_test_buses:]
    print(f"train/val fault buses ({len(trval_buses)}): {sorted(trval_buses)}")
    print(f"test  fault buses ({len(test_buses)}): {sorted(test_buses)}")

    def draw(buses, n_tc, seed):
        rr = np.random.default_rng(seed)
        fb = np.repeat(buses, n_tc)
        tc = rr.uniform(0.03, 0.55, size=len(fb))
        # shuffle to decorrelate the sampling order
        idx = rr.permutation(len(fb))
        return fb[idx], tc[idx]

    fb_train, tc_train = draw(trval_buses, N_TRAIN_TC, seed=1000)
    fb_val, tc_val = draw(trval_buses, 20, seed=1001)
    fb_test, tc_test = draw(test_buses, N_TEST_TC, seed=1002)

    tl_train = np.full_like(fb_train, -1)
    tl_val = np.full_like(fb_val, -1)
    tl_test = np.full_like(fb_test, -1)

    def run(fb, tc, tl):
        return simulate_batch(model, fb, tc, tl, T_END, dt=DT, out_dt=OUT_DT)

    print(f"generating train ({len(fb_train)} scenarios)...")
    tr = run(fb_train, tc_train, tl_train)
    print(f"generating val ({len(fb_val)})...")
    va = run(fb_val, tc_val, tl_val)
    print(f"generating test ({len(fb_test)})...")
    te = run(fb_test, tc_test, tl_test)

    # model constants needed for the physics loss (post-fault, N-0 topology)
    G = np.real(model.yint_pre)
    Bsus = np.imag(model.yint_pre)

    # per-bus fault-on electrical power at the pre-fault angles (severity encoding)
    pe_fault = np.array([model.fault_pe(b) for b in range(1, n_buses + 1)])

    np.savez_compressed(
        "data39.npz",
        # train
        fb_train=fb_train, tc_train=tc_train, y_train=tr["delta_coi"],
        s_train=tr["stable"],
        # val
        fb_val=fb_val, tc_val=tc_val, y_val=va["delta_coi"], s_val=va["stable"],
        # test
        fb_test=fb_test, tc_test=tc_test, y_test=te["delta_coi"], s_test=te["stable"],
        # shared constants
        t_out=t_out, E=model.E, Pm=model.Pm, M=model.M, G=G, B=Bsus,
        Mtot=model.Mtot, delta0=model.delta0,
        pe_fault=pe_fault,
        n_buses=n_buses, n_machines=n, t_end=T_END, out_dt=OUT_DT, t_max=T_MAX,
    )
    print("saved data39.npz")
    print("train stable ratio:", tr["stable"].mean().round(3))
    print("test  stable ratio:", te["stable"].mean().round(3))


if __name__ == "__main__":
    main()
