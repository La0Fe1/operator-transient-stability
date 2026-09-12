"""Batch (vectorized) transient-stability simulation for dataset generation.

Mirrors ClassicalModel.simulate but vectorizes the RK4 integration across a
batch of scenarios that may differ in fault bus, clearing time, and line-trip
topology. Returns center-of-inertia (COI) referenced rotor angles.
"""
from __future__ import annotations

import numpy as np

from power_system import ClassicalModel


def simulate_batch(
    model: ClassicalModel,
    fault_buses: np.ndarray,
    t_clears: np.ndarray,
    trip_lines: np.ndarray,
    t_end: float,
    dt: float = 0.001,
    out_dt: float = 0.01,
) -> dict:
    """Vectorized RK4 over a batch of scenarios.

    Args:
        fault_buses: (B,) int, 1-based fault bus indices.
        t_clears:    (B,) float, fault clearing time [s].
        trip_lines:  (B,) int, 0-based branch index tripped at clearing; use -1
                     for no line trip (N-0).
    Returns:
        dict with 't' (T_out,), 'delta_coi' (B, n, T_out), 'stable' (B, bool).
    """
    B = fault_buses.shape[0]
    n = model.n

    # Precompute reduced internal admittances per unique fault bus / trip line.
    yf_map = {int(fb): model.fault_yint(int(fb)) for fb in np.unique(fault_buses)}
    yp_map = {int(tl): model.post_yint(None if tl < 0 else int(tl))
              for tl in np.unique(trip_lines)}
    Yf = np.stack([yf_map[int(fb)] for fb in fault_buses])   # (B, n, n)
    Yp = np.stack([yp_map[int(tl)] for tl in trip_lines])    # (B, n, n)

    E = model.E[None, :]            # (1, n)
    Pm = model.Pm[None, :]
    M = model.M[None, :]
    ws = model.ws
    D = model.damping
    Mtot = model.Mtot

    delta = np.tile(model.delta0[None, :], (B, 1)).astype(np.float64)
    omega = np.ones((B, n))

    def _pe(d: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Ec = E * np.exp(1j * d)
        return (Ec * np.conj(np.einsum("bij,bj->bi", Y, Ec))).real

    def _deriv(d, w, Y):
        dd = ws * (w - 1.0)
        dw = (Pm - _pe(d, Y) - D * (w - 1.0)) / M
        return dd, dw

    out_t = np.arange(0.0, t_end + out_dt / 2, out_dt)
    n_out = out_t.size
    delta_out = np.zeros((B, n, n_out))
    stable = np.ones(B, dtype=bool)

    step_per_out = max(1, int(round(out_dt / dt)))
    k = 0
    k_out = 0
    t_cur = 0.0

    # integrate with fixed dt
    while t_cur < t_end + dt / 2:
        if k % step_per_out == 0 and k_out < n_out:
            coi = delta - (M * delta).sum(axis=1, keepdims=True) / Mtot
            delta_out[:, :, k_out] = coi
            # stability: pairwise angle spread < pi over the window so far
            spread = coi.max(axis=1) - coi.min(axis=1)
            stable &= (spread < np.pi)
            k_out += 1
        if t_cur >= t_end:
            break
        mask = (t_cur < t_clears)                # (B,)
        Ynow = np.where(mask[:, None, None], Yf, Yp)
        d1, w1 = _deriv(delta, omega, Ynow)
        d2, w2 = _deriv(delta + 0.5 * dt * d1, omega + 0.5 * dt * w1, Ynow)
        d3, w3 = _deriv(delta + 0.5 * dt * d2, omega + 0.5 * dt * w2, Ynow)
        d4, w4 = _deriv(delta + dt * d3, omega + dt * w3, Ynow)
        delta = delta + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
        omega = omega + (dt / 6.0) * (w1 + 2 * w2 + 2 * w3 + w4)
        t_cur += dt
        k += 1

    return {"t": out_t, "delta_coi": delta_out, "stable": stable}


def encode_scenario(
    fault_buses: np.ndarray,
    t_clears: np.ndarray,
    n_buses: int,
    t_max: float = 0.6,
) -> np.ndarray:
    """Branch-net input: one-hot fault bus + normalized clearing time.

    Returns (B, n_buses + 1) float array.
    """
    fb_oh = np.eye(n_buses)[fault_buses - 1]          # (B, n_buses)
    tc = (t_clears / t_max)[:, None]                   # (B, 1)
    return np.concatenate([fb_oh, tc], axis=1)


if __name__ == "__main__":
    import time

    model = ClassicalModel(case="39")
    B = 200
    rng = np.random.default_rng(0)
    fb = rng.integers(1, 40, size=B)
    tc = rng.uniform(0.03, 0.6, size=B)
    tl = np.full(B, -1)

    # 1) correctness: batched == single-simulator for a few scenarios
    from power_system import ClassicalModel as _M

    for i in [0, 1, 2]:
        s = model.simulate(fault_bus=int(fb[i]), t_clear=float(tc[i]), t_end=3.0)
        # single simulator uses dt=1ms; sample every 10th point to match out_dt=10ms
        single = s["delta_coi"][:, ::10]
        b = simulate_batch(model, fb[i : i + 1], tc[i : i + 1], tl[i : i + 1], 3.0)
        diff = np.abs(single - b["delta_coi"][0])
        print(f"scenario {i}: max |single-batch| = {diff.max():.2e} rad")

    # 2) speed
    t0 = time.time()
    simulate_batch(model, fb, tc, tl, 5.0)
    print(f"batched {B} scenarios x 5s (dt=1ms): {time.time()-t0:.2f}s")

    # 3) dt accuracy check (5ms vs 1ms)
    fb1 = fb[:20]
    tc1 = tc[:20]
    tl1 = tl[:20]
    fine = simulate_batch(model, fb1, tc1, tl1, 5.0, dt=0.001)
    coarse = simulate_batch(model, fb1, tc1, tl1, 5.0, dt=0.005)
    print(f"dt=5ms vs 1ms max angle error = {np.abs(fine['delta_coi'] - coarse['delta_coi']).max():.2e} rad")
