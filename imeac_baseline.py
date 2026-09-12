"""IMEAC-style per-machine equal-area CCT on the 12 held-out fault buses.

Follows the individual-machine equal-area criterion (IMEAC, Wang et al. 2018)
unity principle: in the COI frame, a machine is unstable iff its rotor keeps
advancing without reaching a dynamic stationary point (its COI speed never
crosses zero after clearing); the system is unstable iff any machine is
unstable. The CCT is found by binary search on the clearing time.
"""
import numpy as np

from power_system import ClassicalModel
from dataset import simulate_batch


def main():
    model = ClassicalModel(case="39")
    n = model.n

    def fault_on(bus, t_end=0.82, dt=0.001):
        """Sustained-fault trajectory (fault never cleared)."""
        yf = model.fault_yint(bus)
        T = int(t_end / dt)
        delta = np.zeros((n, T)); omega = np.zeros((n, T))
        delta[:, 0] = model.delta0; omega[:, 0] = 1.0
        state = np.concatenate([model.delta0, np.ones(n)])
        for k in range(T - 1):
            state = model._rk4_step(state, yf, dt)
            delta[:, k + 1] = state[:n]; omega[:, k + 1] = state[n:]
        return delta, omega

    def per_machine_stable(delta_c, omega_c, t_end=4.0, dt=0.001):
        """Unity principle: every machine with positive COI speed at clearing must
        reach a dynamic stationary point (COI speed crosses zero) after clearing."""
        yp = model.yint_pre
        T = int(t_end / dt)
        state = np.concatenate([delta_c, omega_c])
        coi0 = model.M @ omega_c / model.Mtot
        w0 = omega_c - coi0
        w = np.zeros((n, T)); w[:, 0] = w0
        for k in range(T - 1):
            state = model._rk4_step(state, yp, dt)
            om = state[n:]
            coi = model.M @ om / model.Mtot
            w[:, k + 1] = om - coi
        for i in range(n):
            if w0[i] > 1e-4 and not (w[i, :] <= 0).any():
                return False
        return True

    def imeac_cct(bus):
        delta, omega = fault_on(bus)
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            k = int(mid / 0.001)
            if per_machine_stable(delta[:, k].copy(), omega[:, k].copy()):
                lo = mid
            else:
                hi = mid
        return lo

    def tds_cct(bus):
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            r = simulate_batch(model, np.array([bus]), np.array([mid]), np.array([-1]), 4.0)
            lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
        return lo

    d = np.load("data39.npz")
    buses = np.unique(d["fb_test"])
    print(f"{'bus':>4} {'TDS CCT(ms)':>12} {'IMEAC CCT(ms)':>14} {'diff(ms)':>9}")
    diffs = []
    for b in buses:
        c_tds = tds_cct(int(b)) * 1000
        c_im = imeac_cct(int(b)) * 1000
        diffs.append(c_im - c_tds)
        print(f"{int(b):>4} {c_tds:>12.1f} {c_im:>14.1f} {c_im - c_tds:>9.1f}")
    diffs = np.array(diffs)
    print(f"\nIMEAC vs TDS: mean diff={diffs.mean():.1f} ms, |diff| mean={np.abs(diffs).mean():.1f} ms")
    print("(PEBS: mean |err| 18.7 ms; operator: 12.3 ms)")


if __name__ == "__main__":
    main()
