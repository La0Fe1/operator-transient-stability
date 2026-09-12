"""One-axis (flux-decay) generator model for transient stability.

Extends the classical model (constant E behind x'd) with the q-axis transient
EMF dynamics E'_q, giving a third-order per-machine model (delta, omega, E'_q).
The field voltage E_fd is held constant (no AVR), so this is the standard
flux-decay model with constant excitation. Reference: Sauer & Pai, Ch. 8.

The network is Kron-reduced to internal buses exactly as in the classical
model; only the internal voltage magnitude E'_q is now a dynamic state, and the
electrical power includes the transient-saliency term (x_q - x'_d) I_d I_q.
"""
from __future__ import annotations

import numpy as np

from power_system import ClassicalModel, GENDYN_39

# One-axis parameters (x_d, x_q, T'_d0) per 1-based bus, on 100 MVA base.
GENDYN39_ONE_AXIS = {
    30: (0.100, 0.069, 10.2),
    31: (0.295, 0.282, 6.56),
    32: (0.2495, 0.237, 5.7),
    33: (0.262, 0.258, 5.69),
    34: (0.670, 0.620, 5.4),
    35: (0.254, 0.241, 7.3),
    36: (0.295, 0.292, 5.66),
    37: (0.290, 0.280, 6.7),
    38: (0.2106, 0.205, 4.79),
    39: (0.020, 0.019, 7.0),
}


class OneAxisModel(ClassicalModel):
    """Flux-decay (one-axis) model: state (delta, omega, E'_q) per machine."""

    def __init__(self, case="39", f0=60.0, damping=0.0):
        super().__init__(case=case, f0=f0, damping=damping, gendyn=GENDYN_39)
        self._setup_oneaxis()

    def _setup_oneaxis(self):
        n = self.n
        xd_s = np.zeros(n); xq = np.zeros(n); Td0 = np.zeros(n)
        for i, b0 in enumerate(self.mach_bus):
            b1 = int(b0) + 1
            xd_s[i], xq[i], Td0[i] = GENDYN39_ONE_AXIS[b1]
        self.xd_s = xd_s      # synchronous d-axis reactance x_d
        self.xq = xq          # q-axis synchronous reactance x_q
        self.Td0 = Td0        # d-axis transient open-circuit time constant
        # note: self.xd (from parent) is the transient reactance x'_d

        # initial currents and field voltage at the pre-fault equilibrium
        E0 = self.E            # |E'_q| from the classical initial EMF
        d0 = self.delta0
        Ec = E0 * np.exp(1j * d0)
        I = self.yint_pre @ Ec
        Iq0 = (I * np.exp(-1j * d0)).real
        Id0 = -(I * np.exp(-1j * d0)).imag      # d-axis current (positive convention)
        self.Iq0 = Iq0; self.Id0 = Id0
        # round-rotor: no saliency term, P_e = E'_q I_q
        self.Pm = E0 * Iq0
        # field voltage so dE'_q/dt = 0 at equilibrium
        self.Efd = E0 + (self.xd_s - self.xd) * Id0
        self.Eq0 = E0.copy()

    def _stator(self, Eq, delta, yint):
        """Return (Iq, Id, Pe) for given (Eq, delta) states and network Y."""
        Ec = Eq * np.exp(1j * delta)
        I = yint @ Ec
        Iq = (I * np.exp(-1j * delta)).real
        Id = -(I * np.exp(-1j * delta)).imag
        Pe = Eq * Iq          # round-rotor: no saliency term
        return Iq, Id, Pe

    def _deriv_oa(self, state, yint):
        n = self.n
        delta = state[:n]; omega = state[n:2 * n]; Eq = state[2 * n:]
        Iq, Id, Pe = self._stator(Eq, delta, yint)
        ddelta = self.ws * (omega - 1.0)
        domega = (self.Pm - Pe - self.damping * (omega - 1.0)) / self.M
        dEq = (self.Efd - Eq - (self.xd_s - self.xd) * Id) / self.Td0
        return np.concatenate([ddelta, domega, dEq])

    def simulate(self, fault_bus, t_clear, t_end, dt=0.001, trip_line=None):
        from pandapower.pypower.makeYbus import makeYbus
        ybus_post = self.ybus_pre
        if trip_line is not None:
            branch = self.branch.copy()
            branch[trip_line, 10] = 0
            ybus_post = makeYbus(self.baseMVA, self.bus, branch)[0].toarray() + np.diag(self.yload)
        yint_post = self._build_yint(ybus_post)
        ybus_fault = self.ybus_pre.copy()
        ybus_fault[fault_bus - 1, fault_bus - 1] += 1e6 + 0j
        yint_fault = self._build_yint(ybus_fault)

        t = np.arange(0.0, t_end + dt / 2, dt)
        T = t.size
        delta = np.zeros((self.n, T)); omega = np.zeros((self.n, T)); Eq = np.zeros((self.n, T))
        state = np.concatenate([self.delta0, np.ones(self.n), self.Eq0])
        delta[:, 0] = self.delta0; omega[:, 0] = 1.0; Eq[:, 0] = self.Eq0

        def rk4(s, yint):
            k1 = self._deriv_oa(s, yint)
            k2 = self._deriv_oa(s + 0.5 * dt * k1, yint)
            k3 = self._deriv_oa(s + 0.5 * dt * k2, yint)
            k4 = self._deriv_oa(s + dt * k3, yint)
            return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        for k in range(T - 1):
            yint = yint_fault if t[k] < t_clear else yint_post
            state = rk4(state, yint)
            delta[:, k + 1] = state[:self.n]
            omega[:, k + 1] = state[self.n:2 * self.n]
            Eq[:, k + 1] = state[2 * self.n:]

        delta_coi = self._coi(delta)
        stable = self._is_stable(delta_coi)
        return {"t": t, "delta": delta, "delta_coi": delta_coi,
                "omega": omega, "Eq": Eq, "stable": bool(stable)}


class ExciterModel(OneAxisModel):
    """One-axis model with a first-order exciter (simplified IEEE DC1A).

    Adds the field voltage E_fd as a fourth state per machine, regulated to
    hold the terminal voltage: T_A dE_fd/dt = K_A (V_ref - V_t) - E_fd, with
    E_fd clipped to [E_fd_min, E_fd_max]. The terminal voltage magnitude V_t is
    recovered from the internal reduction as |E'_q e^{j delta} - j x'_d I|.
    """

    def __init__(self, case="39", f0=60.0, damping=0.0,
                 Ka=20.0, Ta=0.2, Efd_min=0.0, Efd_max=5.0, rate_max=5.0):
        super().__init__(case=case, f0=f0, damping=damping)
        self.Ka = Ka; self.Ta = Ta; self.Efd_min = Efd_min; self.Efd_max = Efd_max
        self.rate_max = rate_max
        Ec = self.Eq0 * np.exp(1j * self.delta0)
        I = self.yint_pre @ Ec
        self.Vt0 = np.abs(Ec - 1j * self.xd * I)
        self.Vref = self.Vt0 + self.Efd / self.Ka     # so equilibrium holds
        self.Efd0 = self.Efd.copy()

    def _vt(self, Eq, delta, yint):
        Ec = Eq * np.exp(1j * delta)
        I = yint @ Ec
        return np.abs(Ec - 1j * self.xd * I)

    def _deriv_exc(self, state, yint):
        n = self.n
        delta = state[:n]; omega = state[n:2 * n]
        Eq = state[2 * n:3 * n]; Efd = state[3 * n:]
        Iq, Id, Pe = self._stator(Eq, delta, yint)
        Vt = self._vt(Eq, delta, yint)
        ddelta = self.ws * (omega - 1.0)
        domega = (self.Pm - Pe - self.damping * (omega - 1.0)) / self.M
        dEq = (Efd - Eq - (self.xd_s - self.xd) * Id) / self.Td0
        dEfd = (self.Ka * (self.Vref - Vt) - Efd) / self.Ta
        dEfd = np.clip(dEfd, -self.rate_max, self.rate_max)     # rate limit
        dEfd = np.where((Efd <= self.Efd_min) & (dEfd < 0), 0.0, dEfd)
        dEfd = np.where((Efd >= self.Efd_max) & (dEfd > 0), 0.0, dEfd)
        return np.concatenate([ddelta, domega, dEq, dEfd])

    def simulate(self, fault_bus, t_clear, t_end, dt=0.001, trip_line=None):
        from pandapower.pypower.makeYbus import makeYbus
        ybus_post = self.ybus_pre
        if trip_line is not None:
            branch = self.branch.copy()
            branch[trip_line, 10] = 0
            ybus_post = makeYbus(self.baseMVA, self.bus, branch)[0].toarray() + np.diag(self.yload)
        yint_post = self._build_yint(ybus_post)
        ybus_fault = self.ybus_pre.copy()
        ybus_fault[fault_bus - 1, fault_bus - 1] += 1e6 + 0j
        yint_fault = self._build_yint(ybus_fault)

        t = np.arange(0.0, t_end + dt / 2, dt)
        T = t.size
        delta = np.zeros((self.n, T)); omega = np.zeros((self.n, T))
        Eq = np.zeros((self.n, T)); Efd = np.zeros((self.n, T))
        state = np.concatenate([self.delta0, np.ones(self.n), self.Eq0, self.Efd0])
        delta[:, 0] = self.delta0; omega[:, 0] = 1.0; Eq[:, 0] = self.Eq0; Efd[:, 0] = self.Efd0

        def rk4(s, yint):
            k1 = self._deriv_exc(s, yint)
            k2 = self._deriv_exc(s + 0.5 * dt * k1, yint)
            k3 = self._deriv_exc(s + 0.5 * dt * k2, yint)
            k4 = self._deriv_exc(s + dt * k3, yint)
            return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        for k in range(T - 1):
            yint = yint_fault if t[k] < t_clear else yint_post
            state = rk4(state, yint)
            delta[:, k + 1] = state[:self.n]
            omega[:, k + 1] = state[self.n:2 * self.n]
            Eq[:, k + 1] = state[2 * self.n:3 * self.n]
            Efd[:, k + 1] = state[3 * self.n:]
            Efd[:, k + 1] = np.clip(Efd[:, k + 1], self.Efd_min, self.Efd_max)

        delta_coi = self._coi(delta)
        stable = self._is_stable(delta_coi)
        return {"t": t, "delta": delta, "delta_coi": delta_coi, "omega": omega,
                "Eq": Eq, "Efd": Efd, "stable": bool(stable)}


if __name__ == "__main__":
    m = OneAxisModel(case="39")
    print("Efd:", m.Efd.round(3))
    print("Pm sum (MW):", (m.Pm * 100).sum().round(1))
    r = m.simulate(fault_bus=5, t_clear=0.0, t_end=2.0)
    print("no-fault drift (deg):", np.rad2deg(np.abs(r["delta"] - r["delta"][:, :1])).max().round(6))
    print("Eq drift (no-fault):", np.abs(r["Eq"] - r["Eq"][:, :1]).max().round(8))

    def cct(bus):
        lo, hi = 0.01, 1.0
        for _ in range(16):
            mid = (lo + hi) / 2
            rr = m.simulate(fault_bus=bus, t_clear=mid, t_end=4.0)
            lo, hi = (mid, hi) if rr["stable"] else (lo, mid)
        return lo
    for b in [4, 8, 16, 29, 39]:
        print(f"CCT bus {b}: {cct(b):.4f} s")
