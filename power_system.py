"""Classical-model transient stability simulator for IEEE 39/118-bus systems.

Solves the multi-machine swing equations with the classical (constant voltage
behind transient reactance) generator model, using a Kron-reduced network to
the internal generator buses.

Reference:
  - P. W. Sauer and M. A. Pai, "Power System Dynamics and Stability."
  - M. A. Pai, "Energy Function Analysis for Power System Stability."

Dynamic data (H, x'd) for the 39-bus 10-machine New England system are the
standard published values (see PLAN.md Section 7 for full citations). The
static network (bus/line/load) is taken from pandapower's IEEE case files.
"""
from __future__ import annotations

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus

# ---------------------------------------------------------------------------
# Canonical dynamic data for the IEEE 39-bus 10-machine New England system.
# Key: 1-based bus -> (inertia H [s], transient reactance x'd [p.u. on 100 MVA]).
# ---------------------------------------------------------------------------
GENDYN_39 = {
    30: (42.0, 0.0310),
    31: (30.3, 0.0697),
    32: (35.8, 0.0531),
    33: (28.6, 0.0436),
    34: (26.0, 0.1320),
    35: (34.8, 0.0500),
    36: (26.4, 0.0490),
    37: (24.3, 0.0570),
    38: (34.5, 0.0570),
    39: (500.0, 0.0060),
}

# 118-bus dynamic data: 53 generators + 1 slack = 54 machines. Placeholder for
# the canonical 54-machine dynamic test case; filled in a later stage. For now
# the 118-bus system is not instantiated with dynamics until data is sourced.
GENDYN_118: dict[int, tuple[float, float]] = {}


def _kron_reduce(Y: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Kron-reduce a (complex) admittance matrix to the nodes in ``keep``.

    Y_red = Y[keep,keep] - Y[keep,drop] @ solve(Y[drop,drop], Y[drop,keep]).
    """
    drop = np.setdiff1d(np.arange(Y.shape[0]), keep)
    if drop.size == 0:
        return Y[np.ix_(keep, keep)].copy()
    Ydd = Y[np.ix_(drop, drop)]
    Ydk = Y[np.ix_(drop, keep)]
    Ykd = Y[np.ix_(keep, drop)]
    Ykk = Y[np.ix_(keep, keep)]
    # Solve Ydd @ X = Ydk  ->  X = inv(Ydd) @ Ydk
    X = np.linalg.solve(Ydd, Ydk)
    return Ykk - Ykd @ X


class ClassicalModel:
    """Classical-model transient stability model for a pandapower network."""

    def __init__(
        self,
        case: str = "39",
        f0: float = 60.0,
        damping: float = 0.0,
        gendyn: dict[int, tuple[float, float]] | None = None,
        load_scale: float = 1.0,
    ):
        self.f0 = f0
        self.ws = 2.0 * np.pi * f0          # synchronous speed [rad/s]
        self.damping = damping
        self.case = case
        self.load_scale = load_scale

        # Static network + power flow
        if case == "39":
            net = pn.case39()
            gendyn = GENDYN_39 if gendyn is None else gendyn
        elif case == "118":
            net = pn.case118()
            gendyn = GENDYN_118 if gendyn is None else gendyn
        else:
            raise ValueError(f"unknown case {case}")

        # scale the load to represent a different operating point
        if load_scale != 1.0:
            net.load["p_mw"] = net.load["p_mw"] * load_scale
            net.load["q_mvar"] = net.load["q_mvar"] * load_scale

        pp.runpp(net, algorithm="nr")
        self.net = net
        self.baseMVA = net._ppc["baseMVA"]

        # Machine terminal buses: generators + slack(s)
        gen_bus = net.gen["bus"].to_numpy().astype(int)          # 0-based
        ext_bus = net.ext_grid["bus"].to_numpy().astype(int)     # 0-based
        self.mach_bus = np.concatenate([gen_bus, ext_bus])       # 0-based terminal buses
        self.n = self.mach_bus.size                              # number of machines

        # Dynamic parameters per machine (H, x'd) indexed by 1-based bus
        H = np.zeros(self.n)
        xd = np.zeros(self.n)
        for i, b0 in enumerate(self.mach_bus):
            b1 = int(b0) + 1
            if b1 not in gendyn:
                raise ValueError(f"missing dynamic data for 1-based bus {b1}")
            H[i], xd[i] = gendyn[b1]
        self.H = H
        self.xd = xd
        # Per-unit speed formulation: ddelta/dt = ws*(w-1), dw/dt = (Pm-Pe)/M.
        # Here M = 2H (seconds); the /ws is already absorbed by using per-unit
        # rotor speed w (not rad/s). See Sauer-Pai Eq. 5.24 / Kundur Ch. 13.
        self.M = 2.0 * H

        # Full bus admittance matrix (N x N) for the intact network
        self.N = net._ppc["bus"].shape[0]
        self.branch = net._ppc["branch"].copy()
        self.bus = net._ppc["bus"].copy()
        self.ybus_branch = makeYbus(self.baseMVA, self.bus, self.branch)[0].toarray()

        # Constant-impedance load admittances (classical-model assumption: each
        # PQ load is converted to a shunt admittance at the power-flow voltage).
        self.yload = self._load_admittances()

        # Network admittance with loads absorbed, then internal-node reduction
        self.ybus_pre = self.ybus_branch + np.diag(self.yload)
        self.yint_pre = self._build_yint(self.ybus_pre)

        # Internal voltage magnitude |E| and angle delta (initial equilibrium)
        self.E, self.delta0 = self._initial_emf()

        # Mechanical power = electrical power at the initial equilibrium
        self.Pm = self._pe(self.delta0, self.yint_pre)

        # COI reference quantities (for the reference frame)
        self.Mtot = self.M.sum()

    # ------------------------------------------------------------------ build
    def _build_yint(self, ybus: np.ndarray) -> np.ndarray:
        """Extend Ybus with internal generator buses (via j x'd) and Kron-reduce."""
        N = self.N
        n = self.n
        Y = np.zeros((N + n, N + n), dtype=np.complex128)
        Y[:N, :N] = ybus
        for i in range(n):
            g = int(self.mach_bus[i])
            y = 1.0 / (1j * self.xd[i])
            Y[g, g] += y
            Y[N + i, N + i] += y
            Y[g, N + i] -= y
            Y[N + i, g] -= y
        keep = np.arange(N, N + n)
        return _kron_reduce(Y, keep)

    def _load_admittances(self) -> np.ndarray:
        """Constant-impedance shunt admittance per bus from the PQ loads."""
        y = np.zeros(self.N, dtype=np.complex128)
        res_bus = self.net.res_bus
        for _, ld in self.net.load.iterrows():
            b = int(ld["bus"])
            p = ld["p_mw"] / self.baseMVA
            q = ld["q_mvar"] / self.baseMVA
            Vm = res_bus["vm_pu"].iloc[b]
            y[b] += (p - 1j * q) / (Vm * Vm)
        return y

    def _initial_emf(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute internal EMF E_i / delta_i from the power-flow terminal state."""
        res_bus = self.net.res_bus
        E = np.zeros(self.n, dtype=np.complex128)
        # Bus net injections from the solved power flow (res_gen + res_ext_grid)
        p_pu = np.zeros(self.N)
        q_pu = np.zeros(self.N)
        for gi in self.net.gen.index:
            b = int(self.net.gen.at[gi, "bus"])
            p_pu[b] += self.net.res_gen.at[gi, "p_mw"] / self.baseMVA
            q_pu[b] += self.net.res_gen.at[gi, "q_mvar"] / self.baseMVA
        for ei in self.net.ext_grid.index:
            b = int(self.net.ext_grid.at[ei, "bus"])
            p_pu[b] += self.net.res_ext_grid.at[ei, "p_mw"] / self.baseMVA
            q_pu[b] += self.net.res_ext_grid.at[ei, "q_mvar"] / self.baseMVA
        for i, b0 in enumerate(self.mach_bus):
            b = int(b0)
            Vm = res_bus["vm_pu"].iloc[b]
            Va = np.deg2rad(res_bus["va_degree"].iloc[b])
            V = Vm * np.exp(1j * Va)
            S = p_pu[b] + 1j * q_pu[b]
            I = np.conj(S / V)
            E[i] = V + 1j * self.xd[i] * I
        Emag = np.abs(E)
        delta = np.angle(E)
        return Emag, delta

    # ------------------------------------------------------------- electrical
    def _pe(self, delta: np.ndarray, yint: np.ndarray) -> np.ndarray:
        """Electrical power P_ei = Re[ E_i * conj( sum_j Y_ij E_j ) ]."""
        Ec = self.E * np.exp(1j * delta)
        return (Ec * np.conj(yint @ Ec)).real

    # ------------------------------------------------------------- integration
    def _deriv(self, state: np.ndarray, yint: np.ndarray) -> np.ndarray:
        n = self.n
        delta = state[:n]
        omega = state[n:]
        ddelta = self.ws * (omega - 1.0)
        Pe = self._pe(delta, yint)
        domega = (self.Pm - Pe - self.damping * (omega - 1.0)) / self.M
        return np.concatenate([ddelta, domega])

    def _rk4_step(self, state: np.ndarray, yint: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._deriv(state, yint)
        k2 = self._deriv(state + 0.5 * dt * k1, yint)
        k3 = self._deriv(state + 0.5 * dt * k2, yint)
        k4 = self._deriv(state + dt * k3, yint)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------------------------------------- precompute
    def fault_yint(self, fault_bus: int) -> np.ndarray:
        """Internal-node admittance during a 3-phase fault at ``fault_bus`` (1-based)."""
        ybus = self.ybus_pre.copy()
        ybus[fault_bus - 1, fault_bus - 1] += 1e6 + 0j
        return self._build_yint(ybus)

    def fault_pe(self, fault_bus: int) -> np.ndarray:
        """Electrical power P_ei of each machine during the fault (at delta0)."""
        return self._pe(self.delta0, self.fault_yint(fault_bus))

    def post_yint(self, trip_line: int | None = None) -> np.ndarray:
        """Internal-node admittance after clearing; ``trip_line`` = 0-based branch index."""
        if trip_line is None:
            return self.yint_pre
        branch = self.branch.copy()
        branch[trip_line, 10] = 0
        ybus = makeYbus(self.baseMVA, self.bus, branch)[0].toarray() + np.diag(self.yload)
        return self._build_yint(ybus)

    def post_yint_multi(self, trip_lines) -> np.ndarray:
        """Internal-node admittance after tripping several branches (list of indices)."""
        trip_lines = [l for l in (trip_lines or []) if l >= 0]
        if not trip_lines:
            return self.yint_pre
        branch = self.branch.copy()
        for l in trip_lines:
            branch[l, 10] = 0
        ybus = makeYbus(self.baseMVA, self.bus, branch)[0].toarray() + np.diag(self.yload)
        return self._build_yint(ybus)

    # ----------------------------------------------------------------- simulate
    def simulate(
        self,
        fault_bus: int,
        t_clear: float,
        t_end: float,
        dt: float = 0.001,
        trip_line: int | None = None,
    ) -> dict:
        """Simulate a 3-phase fault at ``fault_bus`` (1-based), cleared at ``t_clear``.

        ``trip_line``: index (0-based, into self.branch) of a line tripped at
        clearing time (N-1 topology); None = intact post-fault network.

        Returns dict with 't', 'delta' (n x T, absolute), 'delta_coi' (COI-
        referenced), 'omega' (n x T), 'stable' (bool).
        """
        # post-fault network (optional line trip), no fault
        ybus_post = self.ybus_pre
        if trip_line is not None:
            branch = self.branch.copy()
            branch[trip_line, 10] = 0            # BR_STATUS column = 0 (out of service)
            ybus_branch_post = makeYbus(self.baseMVA, self.bus, branch)[0].toarray()
            ybus_post = ybus_branch_post + np.diag(self.yload)
        yint_post = self._build_yint(ybus_post)

        # fault-on network: add large shunt at fault bus
        ybus_fault = self.ybus_pre.copy()
        ybus_fault[fault_bus - 1, fault_bus - 1] += 1e6 + 0j
        yint_fault = self._build_yint(ybus_fault)

        # integrate
        t = np.arange(0.0, t_end + dt / 2, dt)
        T = t.size
        delta = np.zeros((self.n, T))
        omega = np.zeros((self.n, T))
        state = np.concatenate([self.delta0, np.ones(self.n)])
        delta[:, 0] = self.delta0
        omega[:, 0] = 1.0

        for k in range(T - 1):
            yint = yint_fault if t[k] < t_clear else yint_post
            state = self._rk4_step(state, yint, dt)
            delta[:, k + 1] = state[: self.n]
            omega[:, k + 1] = state[self.n :]

        # COI-referenced angles / speeds
        delta_coi = self._coi(delta)
        stable = self._is_stable(delta_coi)

        return {
            "t": t,
            "delta": delta,
            "delta_coi": delta_coi,
            "omega": omega,
            "stable": bool(stable),
        }

    # ----------------------------------------------------------------- helpers
    def _coi(self, delta: np.ndarray) -> np.ndarray:
        """Center-of-inertia referenced rotor angles."""
        coi = (self.M[:, None] * delta).sum(axis=0) / self.Mtot
        return delta - coi

    def _is_stable(self, delta_coi: np.ndarray) -> bool:
        """Stability criterion: no machine's COI angle diverges beyond a threshold.

        Uses the standard max-angle-difference criterion: stable iff every
        pairwise angle difference stays below 180 deg over the whole window.
        """
        peak = np.max(np.abs(delta_coi), axis=0)
        # pairwise differences via max-min per column is a fast upper bound proxy;
        # use the stricter pairwise check for correctness.
        diff_max = 0.0
        for t in range(delta_coi.shape[1]):
            col = delta_coi[:, t]
            d = col.max() - col.min()
            diff_max = max(diff_max, d)
        return diff_max < np.pi  # < 180 deg
