# -*- coding: utf-8 -*-
"""Regenerate the 39-bus test reference (y_test, s_test) with the event-split
integrator at dt=0.5 ms, and count training-label flips for the same fix.
Backs up the original npz first."""
import shutil
import numpy as np

from power_system import ClassicalModel
from dataset import simulate_batch

SRC = "data39.npz"


def main():
    d = np.load(SRC)
    model = ClassicalModel(case="39")

    # ---- training label flips (labels only; trajectories untouched) ----
    fb = d["fb_train"]; tc = d["tc_train"]
    st_old = d["s_train"].astype(bool)
    st_new = np.zeros(len(fb), dtype=bool)
    B = 300
    for i in range(0, len(fb), B):
        r = simulate_batch(model, fb[i:i+B], tc[i:i+B], np.full(min(B, len(fb)-i), -1), 4.0, dt=0.0005)
        st_new[i:i+B] = r["stable"]
        print(f"  train batch {i//B+1}/{(len(fb)+B-1)//B} done", flush=True)
    flips = (st_new != st_old).sum()
    print(f"training label flips: {flips}/{len(fb)} (old stable {st_old.sum()}, new {st_new.sum()})", flush=True)

    # ---- regenerate test reference ----
    fb_t = d["fb_test"]; tc_t = d["tc_test"]
    s_old = d["s_test"].astype(bool)
    y_new = np.zeros_like(d["y_test"])
    s_new = np.zeros(len(fb_t), dtype=bool)
    B = 120
    for i in range(0, len(fb_t), B):
        r = simulate_batch(model, fb_t[i:i+B], tc_t[i:i+B], np.full(min(B, len(fb_t)-i), -1), 4.0, dt=0.0005)
        y_new[i:i+B] = r["delta_coi"]; s_new[i:i+B] = r["stable"]
        print(f"  test batch {i//B+1}/{(len(fb_t)+B-1)//B} done", flush=True)
    tf = (s_new != s_old).sum()
    dy = np.rad2deg(np.abs(y_new - d["y_test"]).max())
    print(f"test label flips: {tf}/{len(fb_t)} (old stable {s_old.sum()}, new {s_new.sum()}); "
          f"max |y_new - y_old| = {dy:.3f} deg", flush=True)

    # ---- write back (backup first) ----
    shutil.copy(SRC, SRC.replace(".npz", "_preB6.npz"))
    out = {k: d[k] for k in d.files}
    out["y_test"] = y_new
    out["s_test"] = s_new
    np.savez(SRC, **out)
    print(f"wrote {SRC} (backup data39_preB6.npz)", flush=True)


if __name__ == "__main__":
    main()
