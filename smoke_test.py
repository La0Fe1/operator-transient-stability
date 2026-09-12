"""Smoke test: verify the classical-model solver produces physically sane
trajectories (equilibrium holds, stable vs unstable fault, CCT is reasonable)."""
import numpy as np
from power_system import ClassicalModel

model = ClassicalModel(case="39", damping=0.0)
print(f"machines n={model.n}, M={model.M.round(2)}")
print(f"|E| = {model.E.round(4)}")
print(f"delta0 (deg) = {np.rad2deg(model.delta0).round(2)}")

# 1) Equilibrium check: run with no fault (very large t_clear => no fault effect)
#    simulate with a tiny fault cleared instantly, angles should stay ~constant
r = model.simulate(fault_bus=5, t_clear=0.0, t_end=2.0)
d = np.rad2deg(r["delta_coi"])
print("\n[equilibrium-ish] max |delta_coi| over 2s =", np.abs(d).max().round(4), "deg")

# 2) Stable vs unstable fault at bus 8
for tc in [0.10, 0.25, 0.40]:
    r = model.simulate(fault_bus=8, t_clear=tc, t_end=5.0)
    d = np.rad2deg(r["delta_coi"])
    print(f"\nfault bus 8, t_clear={tc}s -> stable={r['stable']}, "
          f"max|delta_coi|={np.abs(d).max():.1f} deg, "
          f"final max-min={np.ptp(d[:,-1]):.1f} deg")

# 3) Critical clearing time (CCT) at bus 8 via binary search
def is_stable(tc):
    return model.simulate(fault_bus=8, t_clear=tc, t_end=4.0)["stable"]

lo, hi = 0.01, 0.5
for _ in range(14):
    mid = (lo + hi) / 2
    if is_stable(mid):
        lo = mid
    else:
        hi = mid
print(f"\n[CCT at bus 8] ~ {lo:.4f} s  (typical range 0.1-0.4 s for 39-bus)")

# 4) N-1: trip a line at clearing, check it still runs
r = model.simulate(fault_bus=8, t_clear=0.15, t_end=4.0, trip_line=0)
print(f"\n[N-1 line trip] stable={r['stable']}, max|delta_coi|={np.rad2deg(r['delta_coi']).max():.1f} deg")
