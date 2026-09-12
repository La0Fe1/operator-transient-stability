"""Classical direct (Lyapunov / transient-energy-function) CCT estimation.

Implements the lossless transient energy function (TEF) in the COI frame and the
potential-energy-boundary-surface (PEBS) method to estimate the critical clearing
time, as the classical direct method against which the operator's CCT is compared.

Validation: the total energy KE+PE must be conserved along a post-fault trajectory
(it is a Lyapunov invariant). The PEBS CCT is then compared against the TDS CCT.
"""
import numpy as np

from power_system import ClassicalModel
from dataset import simulate_batch


def coi_ref(model, x):
    return x - (model.M @ x) / model.Mtot


def build_energy(model):
    """Full transient energy function (Athay et al. 1979) in the COI frame.

    Includes the transfer-conductance (path-dependent) term via the straight-line
    approximation, so the energy is (approximately) conserved for the lossy
    reduced network.
    """
    G = model.yint_pre.real
    B = model.yint_pre.imag
    C = np.outer(model.E, model.E) * B          # C_ij = E_i E_j B_ij
    D = np.outer(model.E, model.E) * G          # D_ij = E_i E_j G_ij
    d0 = coi_ref(model, model.delta0)           # SEP in COI frame
    ws = model.ws

    def PE(delta):
        th = coi_ref(model, delta)              # current COI angle
        dc = th - d0                            # COI-referenced, relative to SEP
        pe = -np.sum(model.Pm * dc)
        for i in range(model.n):
            for j in range(i + 1, model.n):
                dij = th[i] - th[j]
                dij0 = d0[i] - d0[j]
                sumd = (th[i] + th[j]) - (d0[i] + d0[j])
                pe -= C[i, j] * (np.cos(dij) - np.cos(dij0))
                if abs(dij - dij0) > 1e-9:      # conductance term, straight-line path
                    Iij = sumd * (np.sin(dij) - np.sin(dij0)) / (dij - dij0)
                else:
                    Iij = sumd * np.cos(dij)
                pe += D[i, j] * Iij
        return ws * pe

    def KE(omega):
        oc = coi_ref(model, omega)              # COI-referenced speed
        return 0.5 * np.sum(model.M * ws ** 2 * oc ** 2)

    return PE, KE


def simulate_fault(model, fault_bus, t_end, dt=0.001):
    """Sustained-fault trajectory (fault never cleared) from the SEP."""
    yf = model.fault_yint(fault_bus)
    T = int(t_end / dt)
    delta = np.zeros((model.n, T)); omega = np.zeros((model.n, T))
    delta[:, 0] = model.delta0; omega[:, 0] = 1.0
    state = np.concatenate([model.delta0, np.ones(model.n)])
    for k in range(T - 1):
        state = model._rk4_step(state, yf, dt)
        delta[:, k + 1] = state[:model.n]; omega[:, k + 1] = state[model.n:]
    return delta, omega


def energy_margin_cct(model, fault_bus, PE, KE, t_end=5.0, dt=0.001):
    delta, omega = simulate_fault(model, fault_bus, t_end, dt)
    T = delta.shape[1]
    pe = np.array([PE(delta[:, k]) for k in range(T)])
    ke = np.array([KE(omega[:, k]) for k in range(T)])
    # PEBS crossing: first local maximum of PE along the fault-on trajectory
    V_cr = None
    for k in range(1, T - 1):
        if pe[k] >= pe[k - 1] and pe[k] >= pe[k + 1]:
            V_cr = pe[k]
            break
    if V_cr is None:
        V_cr = pe.max()
    # CCT: first time total energy reaches the critical energy
    total = ke + pe
    idx = np.where(total >= V_cr)[0]
    return idx[0] * dt if len(idx) else np.nan, V_cr


def main():
    model = ClassicalModel(case="39")
    PE, KE = build_energy(model)

    # ---- validation: energy conservation on a post-fault trajectory ----
    # cleared at t_c, then post-fault only -> energy must be conserved
    d = np.load("data39.npz")
    fb = int(d["fb_test"][0]); tc = float(d["tc_test"][0])
    yf = model.fault_yint(fb); yp = model.yint_pre
    dt = 0.001; T = int(4.0 / dt)
    state = np.concatenate([model.delta0, np.ones(model.n)])
    Es = []
    for k in range(T):
        yint = yf if (k * dt) < tc else yp
        state = model._rk4_step(state, yint, dt)
        if k % 200 == 0 and (k * dt) >= tc:
            d_, o_ = state[:model.n], state[model.n:]
            Es.append(PE(d_) + KE(o_))
    Es = np.array(Es)
    rel = (Es.max() - Es.min()) / max(abs(Es).mean(), 1e-12)
    print(f"energy conservation (post-fault): max drift / mean = {rel:.2e}  (should be ~0)")

    # ---- TDS CCT (binary search) ----
    def tds_cct(bus):
        lo, hi = 0.01, 0.8
        for _ in range(16):
            mid = (lo + hi) / 2
            r = simulate_batch(model, np.array([bus]), np.array([mid]), np.array([-1]), 4.0)
            lo, hi = (mid, hi) if r["stable"][0] else (lo, mid)
        return lo

    # ---- energy-margin CCT on the 12 held-out buses ----
    buses = np.unique(d["fb_test"])
    print(f"\n{'bus':>4} {'TDS CCT(ms)':>12} {'PEBS CCT(ms)':>13} {'err(ms)':>8}")
    errs = []
    for b in buses:
        cct_tds = tds_cct(int(b)) * 1000
        cct_pebs_s, _ = energy_margin_cct(model, int(b), PE, KE)
        cct_pebs = cct_pebs_s * 1000
        e = cct_pebs - cct_tds
        errs.append(e)
        print(f"{int(b):>4} {cct_tds:>12.1f} {cct_pebs:>13.1f} {e:>8.1f}")
    errs = np.array(errs)
    print(f"\nPEBS vs TDS: mean err={errs.mean():.1f} ms, |err| mean={np.abs(errs).mean():.1f} ms")
    print("(operator vs TDS: mean 12.3 ms, max 48.8 ms)")


if __name__ == "__main__":
    main()
