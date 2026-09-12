"""Generate data at 5 load levels for operating-point generalization.

The operator is trained on 4 load levels and evaluated on a held-out level, with
the load level included as an explicit input feature so the severity-to-trajectory
mapping generalizes across operating points.
"""
from __future__ import annotations

import numpy as np

from power_system import ClassicalModel
from dataset import simulate_batch

T_END = 4.0
DT = 0.0005
OUT_DT = 0.01
SEED = 0
SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]


def main():
    rng = np.random.default_rng(SEED)
    all_buses = np.arange(1, 40); rng.shuffle(all_buses)
    test_buses = all_buses[:12]; tr_buses = all_buses[12:]

    def draw(buses, n_tc, seed):
        rr = np.random.default_rng(seed)
        fb = np.repeat(buses, n_tc); tc = rr.uniform(0.03, 0.55, size=len(fb))
        idx = rr.permutation(len(fb)); return fb[idx], tc[idx]

    fb_tr, tc_tr = draw(tr_buses, 40, 1000)
    fb_te, tc_te = draw(test_buses, 40, 1002)
    tl = np.full(len(fb_tr), -1); tl_te = np.full(len(fb_te), -1)

    out = {}
    for scale in SCALES:
        model = ClassicalModel(case="39", load_scale=scale)
        pe_fault = np.array([model.fault_pe(b) for b in range(1, 40)])
        tr = simulate_batch(model, fb_tr, tc_tr, tl, T_END, dt=DT, out_dt=OUT_DT)
        te = simulate_batch(model, fb_te, tc_te, tl_te, T_END, dt=DT, out_dt=OUT_DT)
        out[scale] = dict(
            fb_train=fb_tr, tc_train=tc_tr, y_train=tr["delta_coi"], s_train=tr["stable"],
            fb_test=fb_te, tc_test=tc_te, y_test=te["delta_coi"], s_test=te["stable"],
            pe_fault=pe_fault, Pm=model.Pm, M=model.M,
        )
        print(f"scale {scale}: train stable {tr['stable'].mean():.2f}, test stable {te['stable'].mean():.2f}")

    # assemble multi-level training set (4 levels: 0.8, 0.9, 1.1, 1.2) + held-out 1.0 test
    tr_scales = [0.8, 0.9, 1.1, 1.2]
    fb_tr_all = np.concatenate([out[s]["fb_train"] for s in tr_scales])
    tc_tr_all = np.concatenate([out[s]["tc_train"] for s in tr_scales])
    ls_tr_all = np.concatenate([np.full(len(out[s]["fb_train"]), s - 1.0) for s in tr_scales])
    y_tr_all = np.concatenate([out[s]["y_train"] for s in tr_scales])
    s_tr_all = np.concatenate([out[s]["s_train"] for s in tr_scales])
    sev_tr_all = np.concatenate([
        (out[s]["Pm"][None, :] - out[s]["pe_fault"][out[s]["fb_train"] - 1]) / out[s]["M"][None, :] * 10.0
        for s in tr_scales])

    base = out[1.0]
    np.savez_compressed(
        "data39_loadscale.npz",
        fb_train=fb_tr_all, tc_train=tc_tr_all, ls_train=ls_tr_all,
        y_train=y_tr_all, s_train=s_tr_all, sev_train=sev_tr_all,
        fb_test=base["fb_test"], tc_test=base["tc_test"],
        y_test=base["y_test"], s_test=base["s_test"],
        sev_test=(base["Pm"][None, :] - base["pe_fault"][base["fb_test"] - 1]) / base["M"][None, :] * 10.0,
        ls_test=np.zeros(len(base["fb_test"])),
        t_out=np.arange(0.0, T_END + OUT_DT / 2, OUT_DT),
        n_buses=39, n_machines=base["Pm"].shape[0], t_end=T_END, out_dt=OUT_DT, t_max=0.6,
    )
    print("saved data39_loadscale.npz (multi-level train, held-out 1.0 test)")


if __name__ == "__main__":
    main()
