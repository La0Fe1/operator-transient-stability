# -*- coding: utf-8 -*-
"""B6: verify the event-split integrator against the old one and count label flips.

1. Sanity: new simulate_batch reproduces a stored trajectory to high accuracy on
   scenarios where the fault clears at a grid point (h==dt for every step).
2. Label flips: regenerate the 960 test labels with the event-split integrator and
   compare against the stored labels.
3. CCT convergence: true CCT for the 12 held-out buses at dt = 0.5 / 0.25 / 0.125 ms.
"""
import numpy as np

from power_system import ClassicalModel
from dataset import simulate_batch


def main():
    d = np.load("data39.npz")
    fb = d["fb_test"]; tc = d["tc_test"]; y = d["y_test"]
    tl = np.full(len(fb), -1)
    st_old = d["s_test"].astype(bool)
    model = ClassicalModel(case="39")

    # --- 1. sanity on a small batch ---
    r = simulate_batch(model, fb[:8], tc[:8], tl[:8], 4.0, dt=0.0005)
    err = np.rad2deg(np.abs(r["delta_coi"] - y[:8]).max())
    print(f"sanity: max |new - stored| = {err:.4f} deg over 8 scenarios (8 x 10 x 401 pts)", flush=True)

    # --- 2. label flips on the full test set ---
    B = 200
    st_new = np.zeros(len(fb), dtype=bool)
    for i in range(0, len(fb), B):
        r = simulate_batch(model, fb[i:i+B], tc[i:i+B], tl[i:i+B], 4.0, dt=0.0005)
        st_new[i:i+B] = r["stable"]
        print(f"  batch {i//B + 1}/5 done", flush=True)
    flips = (st_new != st_old)
    print(f"label flips: {flips.sum()}/{len(fb)} test scenarios "
          f"(old stable {st_old.sum()}, new stable {st_new.sum()})", flush=True)
    if flips.any():
        print("flip scenario indices:", np.where(flips)[0][:20], flush=True)

    # --- 3. CCT convergence across step sizes ---
    def cct(bus, dt):
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            r = simulate_batch(model, np.array([bus]), np.array([mid]), np.array([-1]), 4.0, dt=dt)
            lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
        return lo

    buses = np.unique(fb)
    for dt in (0.001, 0.0005, 0.00025):
        ccts = np.array([cct(int(b), dt) for b in buses])
        print(f"dt={dt*1000:.2f} ms  CCTs(ms): " +
              " ".join(f"{c*1000:.1f}" for c in ccts), flush=True)


if __name__ == "__main__":
    main()
