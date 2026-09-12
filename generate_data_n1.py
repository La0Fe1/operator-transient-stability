"""Generate N-0 + N-1 transient-stability data with line-trip topologies.

Extends the fault encoding to include the *post-fault* severity, so the operator
can generalize across both fault locations and line-trip topologies. Test splits
hold out fault buses (test_bus) and trip lines (test_line) separately.
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
    n = model.n
    n_bus = 39
    n_line = len(model.net.line)

    # post-fault electrical power per tripped line (for the post-fault severity)
    pe_post = np.array([model._pe(model.delta0, model.post_yint(trip_line=l))
                        for l in range(n_line)])                    # (35, n)
    pe_fault = np.array([model.fault_pe(b) for b in range(1, n_bus + 1)])  # (39, n)

    rng = np.random.default_rng(SEED)
    all_buses = np.arange(1, n_bus + 1); rng.shuffle(all_buses)
    all_lines = np.arange(n_line); rng.shuffle(all_lines)
    test_buses = all_buses[:12]
    tr_buses = all_buses[12:]
    test_lines = all_lines[:10]
    tr_lines = all_lines[10:]

    def draw_n0(buses, n_tc, seed):
        rr = np.random.default_rng(seed)
        fb = np.repeat(buses, n_tc)
        tc = rr.uniform(0.03, 0.55, size=len(fb))
        tl = np.full(len(fb), -1)
        idx = rr.permutation(len(fb))
        return fb[idx], tl[idx], tc[idx]

    def draw_n1(buses, lines, n_total, seed):
        rr = np.random.default_rng(seed)
        fb = rr.choice(buses, n_total)
        tl = rr.choice(lines, n_total)
        tc = rr.uniform(0.03, 0.55, size=n_total)
        return fb, tl, tc

    print(f"train buses {len(tr_buses)}, test buses {len(test_buses)} | "
          f"train lines {len(tr_lines)}, test lines {len(test_lines)}")

    # train = N-0 + N-1 (seen buses x seen lines)
    fb0, tl0, tc0 = draw_n0(tr_buses, 120, 1000)
    fb1, tl1, tc1 = draw_n1(tr_buses, tr_lines, 25 * 27 * 10, 1001)
    fb_tr = np.concatenate([fb0, fb1]); tl_tr = np.concatenate([tl0, tl1]); tc_tr = np.concatenate([tc0, tc1])
    # validation
    fb_va, tl_va, tc_va = draw_n1(tr_buses, tr_lines, 800, 1002)

    # test_bus: held-out buses (seen lines), N-0 + N-1
    fb_b0, tl_b0, tc_b0 = draw_n0(test_buses, 60, 1003)
    fb_b1, tl_b1, tc_b1 = draw_n1(test_buses, tr_lines, 12 * 25 * 10, 1004)
    fb_tb = np.concatenate([fb_b0, fb_b1]); tl_tb = np.concatenate([tl_b0, tl_b1]); tc_tb = np.concatenate([tc_b0, tc_b1])

    # test_line: held-out lines (seen buses)
    fb_l, tl_l, tc_l = draw_n1(tr_buses, test_lines, 27 * 10 * 10, 1005)

    def run(fb, tl, tc):
        return simulate_batch(model, fb, tc, tl, T_END, dt=DT, out_dt=OUT_DT)

    print(f"generating train {len(fb_tr)} / val {len(fb_va)} / test_bus {len(fb_tb)} / test_line {len(fb_l)} ...")
    tr = run(fb_tr, tl_tr, tc_tr)
    va = run(fb_va, tl_va, tc_va)
    tb = run(fb_tb, tl_tb, tc_tb)
    tl_ = run(fb_l, tl_l, tc_l)

    np.savez_compressed(
        "data39_n1.npz",
        fb_train=fb_tr, tl_train=tl_tr, tc_train=tc_tr, y_train=tr["delta_coi"], s_train=tr["stable"],
        fb_val=fb_va, tl_val=tl_va, tc_val=tc_va, y_val=va["delta_coi"], s_val=va["stable"],
        fb_testb=fb_tb, tl_testb=tl_tb, tc_testb=tc_tb, y_testb=tb["delta_coi"], s_testb=tb["stable"],
        fb_testl=fb_l, tl_testl=tl_l, tc_testl=tc_l, y_testl=tl_["delta_coi"], s_testl=tl_["stable"],
        t_out=model_t_out(), E=model.E, Pm=model.Pm, M=model.M, G=np.real(model.yint_pre),
        B=np.imag(model.yint_pre), Mtot=model.Mtot, delta0=model.delta0,
        pe_fault=pe_fault, pe_post=pe_post, n_buses=n_bus, n_machines=n, n_lines=n_line,
        t_end=T_END, out_dt=OUT_DT, t_max=0.6,
    )
    print("saved data39_n1.npz")
    print("train stable %.2f, test_bus stable %.2f, test_line stable %.2f" %
          (tr["stable"].mean(), tb["stable"].mean(), tl_["stable"].mean()))


def model_t_out():
    return np.arange(0.0, T_END + OUT_DT / 2, OUT_DT)


if __name__ == "__main__":
    main()
