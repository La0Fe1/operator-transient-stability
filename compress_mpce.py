# -*- coding: utf-8 -*-
"""Assertion-based compression of paper_mpce.tex (34p single-col -> <=10p two-col)."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PATH = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
raw = open(PATH, 'rb').read()
t = raw.decode('utf-8').replace('\r\n', '\n')

R = []  # (name, old, new)

# ---------------- abstract ----------------
R.append(('abstract', r"""Transient stability assessment (TSA) requires predicting whether a power system remains
synchronized after a large disturbance and estimating the critical clearing time (CCT) of faults.
Time-domain simulation (TDS) is accurate but too slow for online screening, whereas many
classification-based machine-learning surrogates are fast but return only a stable/unstable
label: they discard the rotor-angle trajectory and provide no error bound. We introduce a
physics-encoded neural operator that maps a fault scenario to the full center-of-inertia
rotor-angle trajectory, and we attach a distribution-free, finite-sample error band to the
trajectory through split conformal prediction. The key enabler is a physics-informed
\emph{input encoding}: a fault is represented by the vector of initial accelerations it induces
on each machine, computed from the swing equation, rather than by an arbitrary bus index. This
encoding allows the operator to generalize to fault locations and to $N{-}1$ and $N{-}2$
topologies not seen during training. On the IEEE 39-bus test system, the operator classifies
stability with \pctACC{} accuracy under a max-over-time $\pi$-separation criterion, estimates the
CCT to within \ccterr{} ms on average against an event-aligned RK4 reference, and runs
$\approx\timSU{}\times$ faster than TDS on identical CPU hardware (throughput protocol; the
end-to-end CCT search gap is $\approx\latSU{}\times$). A directly calibrated pairwise score
bounds the machine-angle spread entering the stability criterion to \pairq{}$^\circ$ at
miscoverage level $\alpha{=}0.1$; the band is coverage-valid, but at this level the screening
rule (predicted spread plus the bound below $\pi$) resolves none of the test scenarios, so the
resolvable fraction is reported as a function of $\alpha$ together with a TDS fallback. The
conformal output is a calibrated trajectory band, not a stability certificate. The architecture
retrained on the IEEE 118-bus system reaches \pctACCb{}$\%$ classification accuracy, and the
encoding carries over unchanged to one-axis, exciter, and damped generator models. A
physics-residual loss does not improve on the physics-informed encoding, and generalization
across operating points (load levels) remains open. To our knowledge, this is the first work to
couple a trajectory-level operator with conformal trajectory bands and CCT extraction from
rotor-angle trajectories in TSA.""",
r"""Transient stability assessment (TSA) predicts whether a power system remains synchronized
after a large disturbance and estimates the critical clearing time (CCT) of faults. Time-domain
simulation (TDS) is accurate but too slow for online screening, whereas machine-learning
classifiers are fast but discard the rotor-angle trajectory and provide no error bound. We
introduce a physics-encoded neural operator that maps a fault scenario to the full
center-of-inertia rotor-angle trajectory, with a distribution-free, finite-sample error band
attached through split conformal prediction. The key enabler is a physics-informed input
encoding: a fault is represented by the vector of initial accelerations it induces on each
machine, computed from the swing equation, rather than by an arbitrary bus index, which lets the
operator generalize to fault locations and $N{-}1$/$N{-}2$ topologies not seen during training.
On the IEEE 39-bus system, the operator classifies stability with \pctACC{} accuracy under a
max-over-time $\pi$-separation criterion, estimates the CCT to within \ccterr{} ms against an
event-aligned RK4 reference, and runs $\approx\timSU{}\times$ faster than TDS on identical CPU
hardware ($\approx\latSU{}\times$ end-to-end). A directly calibrated pairwise score bounds the
machine-angle spread entering the criterion to \pairq{}$^\circ$ at $\alpha{=}0.1$; the band is
coverage-valid, but at this level the screening rule resolves none of the test scenarios, so the
resolvable fraction is reported as a function of $\alpha$ together with a TDS fallback---the
conformal output is a calibrated trajectory band, not a stability certificate. Retrained on the
IEEE 118-bus system the operator reaches \pctACCb{}$\%$ accuracy, and the encoding carries over
unchanged to one-axis, exciter, and damped generator models. A physics-residual loss does not
improve on the physics-informed encoding, and generalization across operating points remains
open. To our knowledge, this is the first work to couple a trajectory-level operator with
conformal trajectory bands and CCT extraction in TSA."""))

# ---------------- intro bullets ----------------
R.append(('intro-bullet1', r"""label~\cite{lu2026confidence}. Section~\ref{sec:related} compares these works along a
capability matrix, and the ``first'' claim is scoped to rotor-angle trajectory bands with CCT
extraction.""",
r"""label~\cite{lu2026confidence}. Section~\ref{sec:related} positions these works relative to
ours, and the ``first'' claim is scoped to rotor-angle trajectory bands with CCT extraction."""))

R.append(('intro-bullet3', r"""\item We provide a split-conformal calibration layer that yields a finite-sample,
distribution-free error band on the trajectory and a directly calibrated pairwise-angle bound
below the $\pi$ threshold at $\alpha{=}0.1$, together with the fraction of scenarios the
screening rule actually resolves, the wrong-release rate, and a CCT estimate compared against
classical direct criteria; the derived CCT band is a heuristic interval without a formal
coverage guarantee (Section~\ref{sec:method}).""",
r"""\item We provide a split-conformal calibration layer that yields a finite-sample,
distribution-free error band on the trajectory and a directly calibrated pairwise-angle bound
below the $\pi$ threshold at $\alpha{=}0.1$, together with the resolvable fraction, the
wrong-release rate, and a CCT estimate compared against classical direct criteria; the derived
CCT band is a heuristic interval without a formal guarantee (Section~\ref{sec:method})."""))

R.append(('intro-bullet5', r"""\item We report trajectory accuracy together with an ablation of an auxiliary physics-residual
loss, including a negative result for this system and protocol: at the mildest weight the residual
term gives no improvement over the physics-informed encoding, and at larger weights it degrades
accuracy, so that here physics is injected more reliably through the input encoding than through a
residual loss.""",
r"""\item We report trajectory accuracy together with an ablation of an auxiliary physics-residual
loss, including a negative result for this system and protocol: physics is injected more reliably
through the input encoding than through a residual loss."""))

# ---------------- related work ----------------
R.append(('rw-2.1', r"""Recent work has begun applying these tools to power systems. Karampinis et
al.~\cite{karampinis2025neural} propose a DeepONet-based framework for modeling power-system
\emph{components} with physics-informed variants, targeting fast dynamic simulation and
digital twins, but do not address transient stability assessment, the CCT, or uncertainty
quantification. For operator learning specifically, Moya et
al.~\cite{moya2025conformal} attach split-conformal intervals to DeepONets
(Conformalized-DeepONet), yielding distribution-free coverage for operator regression on
canonical dynamical systems; our work instead targets the post-fault power-system trajectory and
the CCT. A line of physics-informed graphical and neural-network surrogates has been
applied to power flow and optimal power flow~\cite{huang2025optimal, chen2025power}, focusing on
steady-state problems rather than post-fault dynamics. Our work differs in targeting the
post-fault rotor-angle trajectory operator and in coupling it with conformal certification.""",
r"""Recent work has begun applying these tools to power systems. Karampinis et
al.~\cite{karampinis2025neural} propose a DeepONet-based framework for modeling power-system
\emph{components}, targeting fast dynamic simulation and digital twins, but do not address TSA,
the CCT, or uncertainty quantification. Moya et al.~\cite{moya2025conformal} attach
split-conformal intervals to DeepONets (Conformalized-DeepONet) for operator regression on
canonical dynamical systems, and physics-informed graphical surrogates have been applied to power
flow and optimal power flow~\cite{huang2025optimal, chen2025power}. Our work instead targets the
post-fault rotor-angle trajectory operator and couples it with conformal certification."""))

R.append(('rw-gcn', r"""Learning-based TSA has long been dominated by classifiers that predict a stable/unstable label
from hand-crafted features such as voltage magnitudes, active power, or rotor-angle snapshots
\cite{alimi2020review}. More recent work uses graph neural networks to exploit the network
topology and deep sequence models for the post-fault time series. Modern graph models with
distribution-aware pooling~\cite{chen2022distribution} and related designs report high
classification accuracies on the IEEE 39-bus system under in-sample or per-sample test protocols.
Under our held-out-bus protocol with the same raw features, however, a standard graph
convolutional classifier reaches only $76.0\%$, and a stronger residual graph model $77.3\%$
(Section~\ref{sec:results}), and graph models produce no trajectory, CCT, or uncertainty bound.""",
r"""Learning-based TSA has long been dominated by classifiers that predict a stable/unstable label
from hand-crafted features~\cite{alimi2020review}; more recent graph models with
distribution-aware pooling~\cite{chen2022distribution} report high accuracies on the IEEE 39-bus
system under in-sample or per-sample protocols. Under our held-out-bus protocol with the same raw
features, however, a standard graph convolutional classifier reaches only $76.0\%$ and a stronger
residual graph model $77.3\%$ (Section~\ref{sec:results}), and graph models produce no
trajectory, CCT, or uncertainty bound."""))

R.append(('rw-traj', r"""A small number of works predict the post-fault trajectory rather than only the label; these are
closest in spirit to ours, but they provide no error bounds. Event-structured PINNs (ES-PINN)
model the pre-fault, fault-on, and post-fault stages explicitly with hard state chaining and a
differentiable CCT boundary~\cite{hao2026event} (preprint, arXiv:2607.27681, July 2026); their
residual-to-trajectory-to-CCT error theorem is complementary to our conformal band, which provides
a finite-sample error bound rather than a theoretical error-propagation analysis. PINN-based
acceleration of time-domain simulation~\cite{stiasny2021transient} trains a network with
physics residuals and evaluates the trained network directly; our operator instead learns the
scenario-to-trajectory map with a physics-informed input encoding, avoiding residual minimization
altogether. Fast CCT calculation for systems mixing synchronous and asynchronous
generation~\cite{wang2025fast} is complementary to the TDS-based reference we validate against;
our CCT comes from the learned operator and is validated against event-aligned TDS.""",
r"""A few works predict the post-fault trajectory rather than only the label; these are closest in
spirit to ours, but they provide no error bounds. Event-structured PINNs model the pre-fault,
fault-on, and post-fault stages explicitly with hard state chaining and a differentiable CCT
boundary~\cite{hao2026event}; their error-propagation theorem is complementary to our
finite-sample conformal band. PINN-based acceleration of time-domain simulation
\cite{stiasny2021transient} trains a network with physics residuals; our operator instead learns
the scenario-to-trajectory map with a physics-informed input encoding, avoiding residual
minimization altogether. Fast CCT calculation for systems mixing synchronous and asynchronous
generation~\cite{wang2025fast} is complementary to the TDS-based reference we validate against."""))

R.append(('rw-conf', r"""Confidence-aware TSA has begun to emerge. Lu et al.~\cite{lu2026confidence} combine a large
language model with conformal prediction to produce prediction sets for a binary stability label,
but they predict neither the trajectory nor the CCT and do not use a physics-informed operator.
The two closest works to ours are the conformalized operator regression of Mollaali et
al.~\cite{mollaali2025conformalized} and the Conformalized-DeepONet framework of Moya et
al.~\cite{moya2025conformal}. The former predicts post-fault \emph{bus-voltage} trajectories with
per-output quantile-head calibration for voltage-interval coverage, without rotor-angle
trajectories, CCT extraction, or topology generalization; the latter establishes
distribution-free operator-regression bands on canonical dynamical systems, without a
power-system-specific encoding or TSA decision targets. Probabilistic and Bayesian operator
frameworks for post-fault trajectories (DeepONet-grid-UQ)~\cite{moya2023deeponet} quantify
uncertainty through Gaussian or posterior-sampling assumptions; our split-conformal intervals are
distribution-free with finite-sample guarantees. Risk-controlling quantile operators
(UQNO)~\cite{ma2024calibrated} learn a second operator for pointwise uncertainty radii with a
PAC-style functional calibration; our bands require only a calibration set on top of the trained
operator.""",
r"""Confidence-aware TSA has begun to emerge. Lu et al.~\cite{lu2026confidence} combine a large
language model with conformal prediction to produce prediction sets for a binary stability label,
but they predict neither the trajectory nor the CCT and do not use a physics-informed operator.
The closest works to ours are the conformalized operator regression of Mollaali et
al.~\cite{mollaali2025conformalized}, which predicts post-fault \emph{bus-voltage} trajectories
without rotor angles, CCT extraction, or topology generalization, and the Conformalized-DeepONet
framework of Moya et al.~\cite{moya2025conformal}, which establishes operator-regression bands on
canonical dynamical systems. Probabilistic and Bayesian operator frameworks
(DeepONet-grid-UQ)~\cite{moya2023deeponet} assume Gaussian or posterior-sampling uncertainty,
and risk-controlling quantile operators (UQNO)~\cite{ma2024calibrated} learn a second operator
for pointwise radii; our split-conformal bands are distribution-free and require only a
calibration set on top of the trained operator."""))

R.append(('rw-capability-del', r"""Table~\ref{tab:capability} positions our work against the closest trajectory-level and conformal
methods along the axes that matter for TSA: physics-derived input encoding, the predicted
trajectory variable, topology and operating-point extrapolation, the calibration unit, the
coverage object, and whether the CCT carries a formal guarantee. The combination relevant to
rotor-angle TSA---a physics encoding that generalizes the operator across fault locations and
topologies, a directly calibrated bound on the pairwise spread entering the stability criterion,
and CCT extraction---is not present in any single prior work.

\begin{table}[t]
\centering
\caption{Capability comparison with the closest trajectory-level and conformal operator methods.
``n.r.'' = not reported in that work. The CCT band of this work is heuristic; no entry provides a
formally guaranteed CCT interval.}
\label{tab:capability}
\small
\begin{tabular}{lccccc}
\toprule
 & ES-PINN~\cite{hao2026event} & Conformalized-DeepONet~\cite{moya2025conformal} &
 Voltage conf.~\cite{mollaali2025conformalized} & DeepONet-grid-UQ~\cite{moya2023deeponet} & Ours \\
\midrule
Physics-derived input encoding & partial & no & no & no & yes \\
Trajectory variable & rotor angle & canonical PDEs & bus voltage & bus voltage/angle & rotor angle \\
Unseen-topology generalization & no & n.r. & no & n.r. & $N{-}1$/$N{-}2$ \\
Unseen fault-bus generalization & no & n.r. & n.r. & n.r. & yes \\
Calibration unit & n.a.\ (error theorem) & operator band & per-output quantile & Bayesian/Gaussian & split conformal \\
Coverage object & theoretical bound & trajectory & voltage interval & posterior interval & angle spread / trajectory \\
CCT extraction & yes (differentiable) & no & no & no & yes (heuristic band) \\
\bottomrule
\end{tabular}
\end{table}

Classical direct methods remain strong CCT baselines. The transient-energy-function (TEF/PEBS)""",
r"""Across the axes that matter for TSA---physics-derived input encoding, unseen fault-bus and
topology generalization, the calibration unit, and CCT extraction---no single prior work combines
a physics encoding that generalizes across fault locations and topologies, a directly calibrated
bound on the pairwise spread entering the stability criterion, and CCT extraction.

Classical direct methods remain strong CCT baselines. The transient-energy-function (TEF/PEBS)"""))

R.append(('rw-tef', r"""method~\cite{athay1979transient} and the individual-machine equal-area criterion
(IMEAC)~\cite{wang2018individual} extract the CCT from machine-level energy criteria;
Section~\ref{sec:results} compares our CCT against both, using a full TEF implementation with the
conductance path-integral terms and an IMEAC-style criterion rather than a claimed reproduction
of the standard method. Our contribution is the combination of a trajectory-level neural operator
with a physics-informed fault encoding and a conformal error band, which yields a CCT estimate
alongside the trajectory; the derived CCT band is heuristic and carries no formal coverage
guarantee (Section~\ref{sec:method}).""",
r"""method~\cite{athay1979transient} and the individual-machine equal-area criterion
(IMEAC)~\cite{wang2018individual} extract the CCT from machine-level energy criteria;
Section~\ref{sec:results} compares our CCT against both. Our contribution is the combination of a
trajectory-level neural operator with a physics-informed fault encoding and a conformal error
band, which yields a CCT estimate alongside the trajectory; the derived CCT band is heuristic and
carries no formal coverage guarantee (Section~\ref{sec:method})."""))

# ---------------- problem formulation ----------------
R.append(('pf-3.1', r"""A three-phase fault at a bus is modeled by adding a large shunt admittance to that bus and
reducing the network during the fault; the fault is cleared after a clearing time $t_c$,
optionally tripping a line (an $N{-}1$ contingency). Reference trajectories are obtained by
integrating \eqref{eq:swing1}--\eqref{eq:swing2} with a fourth-order Runge--Kutta scheme at a
$0.5$ ms step, splitting each step at the clearing instant $t_c$ so that the fault-on network is
used exactly until $t_c$ (event alignment); the state is recorded on an output grid with
$0.01$ s spacing. A trajectory is \emph{stable} if the maximum pairwise angle difference remains
below $\pi$ at every output-grid point over the simulation horizon---the $\pi$-separation
criterion. This criterion is the operational label used throughout the paper: it applies to the
selected model, time horizon, and criterion, and is not claimed to coincide with infinite-horizon
synchronism for arbitrary systems (Section~\ref{sec:discussion}).""",
r"""A three-phase fault at a bus is modeled by adding a large shunt admittance to that bus and
reducing the network during the fault; the fault is cleared after a clearing time $t_c$,
optionally tripping a line (an $N{-}1$ contingency). Reference trajectories integrate
\eqref{eq:swing1}--\eqref{eq:swing2} with a fourth-order Runge--Kutta scheme at a $0.5$ ms step,
splitting each step at the clearing instant $t_c$ so the fault-on network is used exactly until
$t_c$ (event alignment); the state is recorded on a $0.01$ s output grid. A trajectory is
\emph{stable} if the maximum pairwise angle difference remains below $\pi$ at every output-grid
point over the simulation horizon---the $\pi$-separation criterion, the operational label used
throughout the paper. It applies to the selected model, horizon, and criterion and is not claimed
to coincide with infinite-horizon synchronism for arbitrary systems
(Section~\ref{sec:discussion})."""))

R.append(('pf-3.2', r"""that returns the COI-referenced rotor-angle trajectory, together with (i) a stability
classification under the $\pi$-separation criterion, (ii) the critical clearing time
$t_{\mathrm{CCT}}$, defined as the first clearing time at which the criterion is violated as
$t_c$ increases from zero---a dense sweep over $t_c$ found a single dominant stable-to-unstable
crossing on most buses (1--5 local flips at the $5$ ms sweep resolution, concentrated on the two
largest-CCT buses), so on the held-out buses this coincides with the largest stable $t_c$---and
(iii) a calibrated error band on the trajectory, in the statistical sense made precise in
Section~\ref{sec:method}.""",
r"""returning the COI-referenced rotor-angle trajectory, together with (i) a stability
classification under the $\pi$-separation criterion, (ii) the critical clearing time
$t_{\mathrm{CCT}}$, the first clearing time at which the criterion is violated as $t_c$
increases from zero---a dense sweep found a single dominant stable-to-unstable crossing on most
buses (1--5 local flips at the $5$ ms resolution, concentrated on the two largest-CCT buses), so
on the held-out buses this coincides with the largest stable $t_c$---and (iii) a calibrated error
band on the trajectory, in the statistical sense made precise in Section~\ref{sec:method}."""))

# ---------------- method ----------------
R.append(('m-4.1', r"""The branch input is the concatenation of
$\bm{a}(k)\in\mathbb{R}^n$ and the normalized clearing time. The encoding is not claimed to be
injective: the initial acceleration describes the vector field at the pre-fault state only, and
two scenarios with similar encodings can still differ in later response, in particular near the
stability boundary; the experiments of Section~\ref{sec:results} show that the encoding
nonetheless transfers the information needed for this task, while the conformal band
(Section~\ref{sec:method}) flags the scenarios where the prediction is least reliable.""",
r"""The branch input is the concatenation of
$\bm{a}(k)\in\mathbb{R}^n$ and the normalized clearing time. The encoding is not claimed to be
injective---two scenarios with similar encodings can still differ in later response near the
stability boundary---but Section~\ref{sec:results} shows it transfers the information needed for
this task, while the conformal band flags the scenarios where the prediction is least reliable."""))

R.append(('m-4.2', r"""Because the swing dynamics are oscillatory, we feed the trunk a Fourier feature embedding of
time, $\{\cos(2\pi k t/T),\sin(2\pi k t/T)\}_{k=1}^{K}$, before a small multilayer perceptron.
This lets the trunk represent the natural oscillation modes more efficiently than a plain
multilayer perceptron on this problem (the FNO-trunk variant of Table~\ref{tab:ablation} is the
closest comparison). The trunk is smooth in $t$, so its time derivatives are available by
automatic differentiation, which we use for the optional physics residual below. Fig.~\ref{fig:arch} summarizes the proposed pipeline.""",
r"""Because the swing dynamics are oscillatory, we feed the trunk a Fourier feature embedding of
time, $\{\cos(2\pi k t/T),\sin(2\pi k t/T)\}_{k=1}^{K}$, before a small multilayer perceptron,
so the trunk represents the natural oscillation modes more efficiently than a plain perceptron
(the FNO-trunk variant of Table~\ref{tab:ablation} is the closest comparison). The trunk is smooth
in $t$, so its time derivatives are available by automatic differentiation for the optional
physics residual below. Fig.~\ref{fig:arch} summarizes the proposed pipeline."""))

R.append(('m-4.3', r"""Two aspects warrant clarification. First, the scaling: the residual is divided by
$M_i$ before squaring, yielding a per-machine term comparable across machines; it retains
acceleration units (rad/s$^2$) and is a soft penalty rather than a hard initial-condition or
state-chaining constraint. Second, the role of $\omega_s$: it enters through the
frequency-to-angle relation $d\delta_i/dt=\omega_s(\omega_i-1)$ and therefore multiplies the power
mismatch; it fixes the relative scaling between the curvature and power-mismatch terms and cannot
be omitted (a scaling check in Section~\ref{sec:results} verifies this). The total training loss
is the data loss plus $\lambda\mathcal{L}_{\mathrm{phys}}$. The residual is enforced only in the
post-fault region because the fault-on dynamics use the fault-dependent reduced admittance, which
varies per scenario; collocation points are every fifth output time point (a $0.05$ s stride),
and denser collocation is ablated in Section~\ref{sec:results}.""",
r"""The per-machine division by $M_i$ makes the term comparable across machines; it retains
acceleration units and is a soft penalty rather than a hard constraint. The $\omega_s$ factor
enters through the frequency-to-angle relation and fixes the scaling between the curvature and
power-mismatch terms. The total loss is the data loss plus $\lambda\mathcal{L}_{\mathrm{phys}}$,
enforced only in the post-fault region (the fault-on admittance varies per scenario) at every
fifth output time point; denser collocation is ablated in Section~\ref{sec:results}."""))

R.append(('m-4.4-limits', r"""i.e., with probability at least $1-\alpha$ over exchangeable stable scenarios, every single-machine
COI angle at every \emph{output-grid} point lies within $\hat q$ of the truth. Three limits are
explicit: the guarantee is marginal over the stable subpopulation (it is not unconditional and
does not cover misclassified unstable scenarios); it applies to the finite output grid (behavior
between grid points is not covered, though a denser-grid check in Section~\ref{sec:results} found
no missed criterion crossings); and it is a calibrated trajectory band, not a stability
certificate. By the triangle inequality, the pairwise angle difference---the quantity relevant to
the stability criterion---is bounded by $2\hat q$, which at $\alpha{=}0.1$ exceeds the
$\pi$-separation threshold (the per-machine envelope becomes informative for the criterion only at
larger miscoverage).""",
r"""i.e., with probability at least $1-\alpha$ over exchangeable stable scenarios, every single-machine
COI angle at every \emph{output-grid} point lies within $\hat q$ of the truth. Three limits are
explicit: the guarantee is marginal over the stable subpopulation (it does not cover misclassified
unstable scenarios); it applies to the finite output grid (a denser-grid check found no missed
criterion crossings, Section~\ref{sec:results}); and it is a calibrated trajectory band, not a
stability certificate. By the triangle inequality, the pairwise angle difference relevant to the
stability criterion is bounded by $2\hat q$, which at $\alpha{=}0.1$ exceeds the $\pi$ threshold."""))

R.append(('m-4.4-pw', r"""which scores exactly the pairwise angle spread appearing in the $\pi$-separation criterion; the
errors $\hat e_i$ keep their sign, so that oppositely signed machine errors add rather than
cancel. As shown in Section~\ref{sec:results}, $R_{\mathrm{pw}}$ bounds the pairwise difference to
\pairq{}$^\circ$ at $\alpha{=}0.1$---below the $\pi$ threshold, but too wide to certify decisions
in practice: on the evaluation scenarios the rule
$\mathrm{spread}+\hat q_{\mathrm{pw}}<\pi$ resolves \decfrac{} of the predicted-stable cases at
$\alpha{=}0.1$, rising to $31\%$ at $\alpha{=}0.3$ and $67\%$ at $\alpha{=}0.5$ as the bound
tightens. The pairwise bound is therefore a screening aid used together with a TDS fallback for
the unresolved fraction, not a stand-alone decision rule. Both bounds are
conservative: the per-machine half-width $\hat q$ is
substantially larger than the typical error because the worst-case score covers near-boundary
scenarios where the trajectory error is concentrated. Multiplicity-corrected per-machine bounds
(Bonferroni/\v{S}id\'ak) do not tighten the pairwise bound (Section~\ref{sec:results}), and any
joint guarantee for multiple machines would additionally require the structural conditions of the
chosen correction rather than following automatically from marginal coverage.""",
r"""which scores exactly the pairwise angle spread appearing in the $\pi$-separation criterion; the
errors $\hat e_i$ keep their sign, so that oppositely signed machine errors add rather than
cancel. As shown in Section~\ref{sec:results}, $R_{\mathrm{pw}}$ bounds the pairwise difference to
\pairq{}$^\circ$ at $\alpha{=}0.1$---below the $\pi$ threshold, but too wide to certify decisions
in practice: the rule $\mathrm{spread}+\hat q_{\mathrm{pw}}<\pi$ resolves \decfrac{} of the
predicted-stable cases at $\alpha{=}0.1$, rising to $31\%$ at $\alpha{=}0.3$ and $67\%$ at
$\alpha{=}0.5$ as the bound tightens. The pairwise bound is therefore a screening aid used with a
TDS fallback for the unresolved fraction, not a stand-alone decision rule. Multiplicity-corrected
per-machine bounds (Bonferroni/\v{S}id\'ak) do not tighten the pairwise bound
(Section~\ref{sec:results}); any joint guarantee would additionally require the structural
conditions of the chosen correction."""))

R.append(('m-4.4-cctband', r"""A CCT band can be derived from the per-machine interval: each clearing time whose predicted
trajectory, shifted by $\pm 2\hat q$ in spread space, cannot reach the $\pi$ threshold is declared
stable with margin, and each clearing time whose shifted spread exceeds $\pi$ for \emph{all}
shifts is declared unstable; clearing times in between form the band. The band endpoints are found
by the same binary search carried out with the shifted criterion. Because the per-scenario
trajectory bound at one $t_c$ does not transfer to a simultaneous guarantee across all $t_c$ of
the same fault (the CCT is a nonlinear functional of the trajectory, and the coverage object of
split conformal is the trajectory at a given scenario, not the functional), this CCT band is
\emph{heuristic}: it carries no formal coverage guarantee and is reported as a screening aid.
Section~\ref{sec:results} gives its width, its empty/trivial cases, and the fraction of scenarios
it resolves.""",
r"""A CCT band can be derived from the per-machine interval: clearing times whose predicted
trajectory, shifted by $\pm 2\hat q$ in spread space, cannot reach the $\pi$ threshold are
declared stable with margin, and clearing times whose shifted spread exceeds $\pi$ for \emph{all}
shifts are declared unstable; the remainder forms the band, whose endpoints are found by the same
binary search with the shifted criterion. Because the trajectory bound at one $t_c$ does not
transfer to a simultaneous guarantee across $t_c$ (the CCT is a nonlinear functional of the
trajectory), this CCT band is \emph{heuristic}: it carries no formal coverage guarantee and is
reported as a screening aid. Section~\ref{sec:results} gives its width and the fraction of
scenarios it resolves."""))

R.append(('m-4.5', r"""\subsection{Critical clearing time estimation}
Given the operator, the CCT of a fault is estimated by binary search over the clearing time: for
each candidate $t_c$, the operator predicts the trajectory, and the scenario is declared unstable
if the maximum pairwise angle spread exceeds $\pi$ at \emph{any} output-grid point, and stable
otherwise---the same max-over-time criterion used for the reference labels. The search uses
$16$ iterations over $[0.01,0.8]$ s for the 39-bus system, so the interval shrinks to
$(0.8-0.01)/2^{16}\approx12\,\mu$s; the same protocol (with a system-dependent interval---$[0.005,0.2]$ s
for the 118-bus system, whose CCTs are much shorter) is applied to the TDS reference, so the two
CCT estimates are directly comparable. Three distinct error sources are kept separate: the
$12\,\mu$s figure is the \emph{search tolerance} of the bisection, not the physical accuracy of
the CCT; the reference itself is computed by the event-aligned RK4 scheme of
Section~\ref{sec:problem} at a $0.5$ ms step, whose convergence was verified by repeating the
reference CCT computation at $0.25$ ms (identical CCTs at $0.1$ ms resolution on all $12$ held-out
buses); and both the operator and the reference judge stability on the $0.01$ s output grid, so
criterion crossings between grid points are invisible to both. A per-bus table of reference and
estimated CCTs with signed errors is given in Section~\ref{sec:results} (Table~\ref{tab:cct}).
The same search, carried out with the shifted criterion of the previous subsection, yields the
heuristic CCT band.""",
r"""\subsection{Critical clearing time estimation}
Given the operator, the CCT of a fault is estimated by binary search over the clearing time: for
each candidate $t_c$, the operator predicts the trajectory, and the scenario is declared unstable
if the maximum pairwise angle spread exceeds $\pi$ at \emph{any} output-grid point---the same
max-over-time criterion used for the reference labels. The search uses $16$ iterations over
$[0.01,0.8]$ s ($[0.005,0.2]$ s for the 118-bus system); the same protocol is applied to the TDS
reference, so the two CCT estimates are directly comparable. The $12\,\mu$s interval is the
\emph{search tolerance} of the bisection, not the physical accuracy of the CCT; the reference
itself uses the event-aligned RK4 scheme of Section~\ref{sec:problem}, whose convergence was
verified at a $0.25$ ms step, and both judge stability on the $0.01$ s output grid. The same
search with the shifted criterion of the previous subsection yields the heuristic CCT band."""))

# ---------------- setup ----------------
R.append(('s-5.1', r"""\subsection{Test system and data}
We use the IEEE 39-bus New England system with $10$ machines, whose static data are the standard
published values and whose dynamic parameters ($H$, $x'_{d}$) are the canonical
classical-model values~\cite{sauer1998dynamics, athay1979transient}; the specific parameter
values and their source are tabulated in the code repository. The reference
simulator integrates \eqref{eq:swing1}--\eqref{eq:swing2} with the event-aligned RK4 scheme of
Section~\ref{sec:problem} at a $0.5$ ms step over a $4$ s horizon. We sample fault scenarios by
drawing a fault bus uniformly over all $39$ buses and a clearing time uniformly in $[0.03,0.55]$ s,
producing a balanced mix of stable and unstable cases (approximately $40\%$ stable). The dataset
has $2160$ training, $540$ validation, and $960$ test scenarios ($80$ scenarios per bus on the
$27$ training buses, $80$ per bus on the $12$ test buses, and $20$ validation scenarios per
training bus). The test set is constructed from $12$ fault buses held out entirely from training,
so that test performance measures generalization to unseen fault locations; the validation set is
drawn from the training buses. The reference labels and trajectories are generated with the
event-aligned integrator; replacing the pre-alignment reference (which kept the fault on until
the first grid point after $t_c$) changes exactly one of $960$ test labels and one of $2160$
training labels, both at the stability boundary, so the reported metrics are insensitive to this
correction. The CCT search range $[0.01,0.8]$ s covers the observed 39-bus CCTs ($0.13$--$0.58$ s
over all buses, $0.13$--$0.45$ s on the held-out buses); any CCT above the sampled clearing-time
range would be an extrapolation and is reported as such.""",
r"""\subsection{Test system and data}
We use the IEEE 39-bus New England system with $10$ machines, whose static data are the standard
published values and whose dynamic parameters ($H$, $x'_{d}$) are the canonical
classical-model values~\cite{sauer1998dynamics, athay1979transient}; the specific parameter
values and their source are tabulated in the code repository. The reference simulator integrates
\eqref{eq:swing1}--\eqref{eq:swing2} with the event-aligned RK4 scheme of Section~\ref{sec:problem}
at a $0.5$ ms step over a $4$ s horizon. We draw a fault bus uniformly over all $39$ buses and a
clearing time uniformly in $[0.03,0.55]$ s, producing approximately $40\%$ stable scenarios. The
dataset has $2160$ training, $540$ validation, and $960$ test scenarios; the test set is
constructed from $12$ fault buses held out entirely from training, so test performance measures
generalization to unseen fault locations, while the validation set is drawn from the training
buses. The event-aligned labels differ from the pre-alignment reference (which kept the fault on
until the first grid point after $t_c$) on exactly one of $960$ test and one of $2160$ training
scenarios, both at the stability boundary, so the reported metrics are insensitive to this
correction. The CCT search range $[0.01,0.8]$ s covers the observed 39-bus CCTs ($0.13$--$0.58$ s
over all buses, $0.13$--$0.45$ s on the held-out buses)."""))

R.append(('s-5.2', r"""\subsection{Baselines and ablations}
We compare against (i) the same DeepONet with a one-hot fault encoding, an ablation isolating the
effect of the physics-informed encoding; (ii) a pointwise multilayer perceptron baseline that
maps (scenario, time) to the angle directly; (iii) the operator trained with the auxiliary
physics-residual loss at several weights; (iv) a graph convolutional network (GCN) that predicts
the stability label from raw node features (voltage magnitudes and angles, loads, and a fault
flag) and, in a regression variant, predicts the CCT directly; (v) a DeepONet with a self-attention
branch; (vi) a Fourier-neural-operator trunk; and (vii) a recurrent (GRU) sequence-to-sequence
baseline. We additionally compare the CCT against two classical direct methods: the
transient-energy-function (PEBS) method~\cite{athay1979transient}, implemented with the
conductance path-integral terms of Athay et al.\ (straight-line approximation) rather than a
lossless simplification, and an IMEAC-style individual-machine criterion~\cite{wang2018individual}
implemented as described in Section~\ref{sec:results}. The DeepONet variants, the MLP, and the
GRU share the same dataset, optimizer, and class-balanced mean-squared loss (the loss target
saturates beyond $\pm 3\pi$ so that the diverging unstable trajectories do not dominate the fit)
and the same $600$-epoch budget. The two GCN baselines are the exception: they are classifiers
or CCT regressors, so they use binary cross-entropy (or MSE for the CCT regression) and
$300$ epochs; their input features are raw observables, so the GCN comparison measures the value
of the severity encoding, not of the operator architecture alone (Section~\ref{sec:results}).
To control for the input representation, we additionally train a severity-MLP baseline that
receives exactly the same encoding as the operator and predicts either the stability label or the
CCT, and we equip the pointwise MLP and the GRU with the same conformal calibration and CCT
search as the operator so that all trajectory-level models are compared under one evaluation
protocol (Section~\ref{sec:results}). Parameter counts and training budgets are listed in the
repository.""",
r"""\subsection{Baselines and ablations}
We compare against (i) the same DeepONet with a one-hot fault encoding, isolating the effect of
the physics-informed encoding; (ii) a pointwise multilayer perceptron baseline mapping (scenario,
time) to the angle directly; (iii) the operator trained with the auxiliary physics-residual loss
at several weights; (iv) graph convolutional networks (GCNs) that predict the stability label or
the CCT from raw node features (voltage magnitudes and angles, loads, and a fault flag); (v) a
self-attention branch; (vi) a Fourier-neural-operator trunk; and (vii) a recurrent (GRU)
sequence-to-sequence baseline. The CCT is additionally compared against the transient-energy
function (PEBS) method~\cite{athay1979transient} with the conductance path-integral terms of Athay
et al., and an IMEAC-style individual-machine criterion~\cite{wang2018individual}. The DeepONet
variants, MLP, and GRU share the same dataset, optimizer, class-balanced mean-squared loss
(saturating beyond $\pm 3\pi$ so diverging unstable trajectories do not dominate the fit), and
$600$-epoch budget; the GCNs use binary cross-entropy (or MSE) and $300$ epochs on raw
observables, so the GCN comparison measures the value of the severity encoding rather than of the
operator architecture. A severity-MLP baseline receives exactly the same encoding as the operator
and predicts the stability label or the CCT, and the pointwise MLP and GRU are equipped with the
same conformal calibration and CCT search, so all trajectory-level models are compared under one
protocol. Parameter counts and training budgets are listed in the repository."""))

R.append(('s-5.3-metrics', r"""\subsection{Evaluation}
We report the root-mean-square error (RMSE) of the predicted trajectory over stable test
scenarios, decomposed into the fault-on ($t<t_c$) and post-fault ($t\ge t_c$) regions; the
stability classification accuracy under the max-over-time $\pi$-separation criterion, together
with the confusion matrix and the precision, recall, and $F_1$ of the stable class; the conformal
band half-width and its empirical coverage with coverage counts ($k$ of $n$) and binomial
intervals; the CCT error against the event-aligned TDS binary search; and the wall-clock speedup
of the operator over the RK4 reference. Throughout, ``RMSE'' refers to the \emph{pooled} RMSE,
i.e., the square root of the mean squared error over all stable scenarios, machines, and time
steps in the relevant region, computed in one pass. The pooled RMSE equals the square root of the
sample-size-weighted mean of the squared per-region RMSEs---it is not a linear average of
regional RMSEs and is not comparable to a per-scenario time-averaged \emph{mean absolute error}
(MAE); where a regional decomposition is reported, both quantities are defined separately so they
are not mixed. The pooled RMSE is dominated by near-boundary scenarios and is therefore larger
than a per-scenario time-averaged error. All main-table numbers are exported by a single
evaluation script from one model and one dataset; four-seed statistics are reported separately
where available.""",
r"""\subsection{Evaluation}
We report the root-mean-square error (RMSE) of the predicted trajectory over stable test
scenarios, decomposed into the fault-on ($t<t_c$) and post-fault ($t\ge t_c$) regions; the
stability classification accuracy under the max-over-time $\pi$-separation criterion, with the
confusion matrix and precision, recall, and $F_1$ of the stable class; the conformal band
half-width and its empirical coverage with counts and binomial intervals; the CCT error against
the event-aligned TDS binary search; and the wall-clock speedup over the RK4 reference. ``RMSE''
refers to the \emph{pooled} RMSE over all stable scenarios, machines, and time steps in the
relevant region, computed in one pass; it is dominated by near-boundary scenarios and is not
comparable to a per-scenario time-averaged mean absolute error. All main-table numbers are
exported by a single evaluation script from one model and one dataset; four-seed statistics are
reported separately where available."""))

R.append(('s-5.3-hyper-del', r"""The results are not artifacts of the hyperparameters: halving the
Fourier harmonics to $K{=}8$ degrades the pooled RMSE to $45.9^\circ$ ($94.3\%$ accuracy), and
reducing the loss saturation to $\pm2\pi$ yields $94.7\%$ accuracy with a slightly better pooled
RMSE of $34.3^\circ$; we retain $\pm3\pi$ as the default because it is selected on
validation data and gives more stable fits to diverging unstable trajectories.

""", r""""""))

R.append(('s-5.3-speed', r"""The speedup is reported under two protocols that are kept separate. \emph{Throughput}: both
methods are single-threaded vectorized implementations, measured after warm-up over repeated
batches of $64$ scenarios on the same CPU, with BLAS/torch thread count pinned to one;
the operator evaluates a $401$-point trajectory at \timCPU{} ms per scenario against
\timRK{} ms for the RK4 reference ($0.5$ ms step, $4$ s horizon), a $\approx\timSU{}\times$
gap on identical CPU hardware. \emph{End-to-end latency}: a single fault scenario, including the
severity encoding (Kron reduction of the fault-on network, $\approx0.13$ ms per bus on the
39-bus system) and the sequential $16$-iteration CCT binary search ($16$ dependent operator
inferences, not $16\times$ the throughput time), takes \latEnd{} ms versus \latRK{} ms for the
same brute-force CCT search with TDS, a \latSU{}$\times$ end-to-end gap; a $100$-point
clearing-time sweep per fault costs \latSweep{} ms end-to-end including the encoding, versus
$2.3$ s for RK4. On the 118-bus system the encoding costs $\approx4.1$ ms per fault and is the
dominant term for a single scenario. Both protocols are reported with warm-up, repetition
counts, and median/P95 statistics in Table~\ref{tab:runtime} and the repository. Production TDS
tools typically use variable step sizes, multithreading, and richer models, so absolute timings
vary by implementation, but the gap persists under like-for-like settings (Section~\ref{sec:setup}).
The RK4 cost scales inversely with the reference step size, whereas the operator's cost depends
only on the number of output time points at which the trunk is evaluated ($401$ per trajectory
here), not on the integration step, so the speedup grows for finer reference steps. Hardware
models, software versions, and the number of timing repetitions are listed in the code
repository (\url{https://github.com/La0Fe1/operator-transient-stability}).""",
r"""The speedup is reported under two separate protocols on identical CPU hardware, with
single-threaded implementations, warm-up, and median/P95 statistics (details in the repository).
\emph{Throughput}: over repeated batches of $64$ scenarios the operator evaluates a $401$-point
trajectory at \timCPU{} ms per scenario against \timRK{} ms for the RK4 reference ($0.5$ ms step,
$4$ s horizon), a $\approx\timSU{}\times$ gap. \emph{End-to-end latency}: a single fault
scenario, including the severity encoding (Kron reduction, $\approx0.13$ ms per bus on the 39-bus
system) and the sequential $16$-iteration CCT binary search, takes \latEnd{} ms versus
\latRK{} ms for the same brute-force search with TDS, a \latSU{}$\times$ gap; a $100$-point
clearing-time sweep costs \latSweep{} ms end-to-end versus $2.3$ s for RK4. On the 118-bus system
the encoding ($\approx4.1$ ms per fault) dominates a single scenario. The RK4 cost scales
inversely with the reference step size, whereas the operator's cost depends only on the number of
output time points ($401$ per trajectory), so the speedup grows for finer reference steps.
Hardware models, software versions, and timing repetitions are listed in the repository
(\url{https://github.com/La0Fe1/operator-transient-stability})."""))

R.append(('tab-runtime-del', r"""\begin{table}[t]
\centering
\caption{Runtime summary (single-threaded, vectorized implementations; mean over repeated runs
after warm-up).}
\label{tab:runtime}
\begin{tabular}{lccc}
\toprule
Method & Device & Per-scenario time & Speedup \\
\midrule
Operator (trajectory inference, throughput) & CPU & \timCPU{} ms & $\approx\timSU{}\times$ \\
Operator (trajectory inference, latency B=1) & CPU & $1.46$ ms & -- \\
RK4 TDS reference ($0.5$ ms step, $4$ s) & CPU & \timRK{} ms & $1\times$ \\
CCT search (16 seq.\ steps, operator$+$encoding) & CPU & \latEnd{} ms & $\approx\latSU{}\times$ \\
CCT search (16 seq.\ steps, TDS) & CPU & \latRK{} ms & $1\times$ \\
Severity encoding (Kron reduction) & CPU & $0.13$ ms (39-bus) / $4.1$ ms (118-bus) & per fault \\
\bottomrule
\end{tabular}
\end{table}

""", r""""""))

# ---------------- results ----------------
R.append(('r-6.1', r"""\subsection{Trajectory prediction and classification}
Table~\ref{tab:main} summarizes the main results. The physics-informed operator predicts the
fault-on segment of the trajectory to \rmseFault{}$^\circ$ pooled RMSE---the segment most
sensitive to fault-specific dynamics---and the post-fault segment to \rmsePost{}$^\circ$ pooled
RMSE (the full-horizon pooled RMSE is \rmseAll{}$^\circ$; the pooled RMSE is not comparable to a
per-scenario time-averaged MAE, Section~\ref{sec:setup}). The post-fault error is
concentrated near the stability boundary, where the trajectory response to small parameter
changes is sharpest; this sensitivity is exactly what the conformal band is designed to capture.
Decomposing the post-fault error: the first-swing window (up to the latest per-machine first-swing
peak) is predicted to $10.5^\circ$ pooled RMSE, while the subsequent oscillation carries
$41.8^\circ$. The operator classifies stability under the max-over-time $\pi$-separation
criterion with \pctACC{} accuracy and confusion matrix (TP, FP, FN, TN) $=$\confmat{} ($F_1$
for the stable class \fone{}), and estimates the CCT to within \ccterr{} ms on average
(Table~\ref{tab:cct} lists each held-out bus), at a $\approx\timSU{}\times$ throughput speedup
over the RK4 reference on identical CPU hardware. In the scenarios studied, the CCT is dominated
by the first-swing peak angle, which the operator predicts more reliably than the subsequent
oscillation; the conformal band quantifies the residual error rather than claiming it is small.

For reference, the transient-energy-function (PEBS) approach of Athay et
al.~\cite{athay1979transient}, implemented with the conductance path-integral terms, attains a
mean CCT error of $18.7$ ms with a conservative (under-estimating) bias on the same held-out
buses, and an IMEAC-style individual-machine criterion~\cite{wang2018individual}---judging
stability per machine by the existence of a dynamic stationary point on the post-fault
trajectory, evaluated on the same $12$ held-out buses with the same $16$-iteration binary search
($0.5$ ms step) as the TDS reference---yields a mean CCT error of $9.6$ ms. The operator's
\ccterr{} ms mean absolute error is larger than both classical criteria on these buses
(IMEAC-style $9.6$ ms, PEBS $18.7$ ms) and carries a conservative bias ($-21$ ms mean signed
error), dominated by bus 12 (reference CCT $453$ ms, estimated $248$ ms, $-204.9$ ms), the bus
with the largest CCT, where the binary search lands between multiple local stable-to-unstable
crossings of the operator's criterion. Classical direct methods remain the more accurate CCT
estimators on this system; the operator's value is that it additionally returns the full
trajectory and a conformal band without any fault-on or post-fault integration.
Fig.~\ref{fig:trajectory} shows predicted versus true COI
trajectories for a representative near-boundary scenario, and Fig.~\ref{fig:cct} plots predicted
against true CCT across the held-out fault buses.""",
r"""\subsection{Trajectory prediction and classification}
Table~\ref{tab:main} summarizes the main results. The physics-informed operator predicts the
fault-on segment of the trajectory to \rmseFault{}$^\circ$ pooled RMSE---the segment most
sensitive to fault-specific dynamics---and the post-fault segment to \rmsePost{}$^\circ$ (the
full-horizon pooled RMSE is \rmseAll{}$^\circ$). The post-fault error is concentrated near the
stability boundary, where the trajectory response to small parameter changes is sharpest---the
sensitivity the conformal band is designed to capture. Decomposing the post-fault error, the
first-swing window is predicted to $10.5^\circ$ pooled RMSE while the subsequent oscillation
carries $41.8^\circ$. The operator classifies stability under the max-over-time $\pi$-separation
criterion with \pctACC{} accuracy and confusion matrix (TP, FP, FN, TN) $=$\confmat{} ($F_1$
for the stable class \fone{}), and estimates the CCT to within \ccterr{} ms on average (max
$204.9$ ms on bus 12), at a $\approx\timSU{}\times$ throughput speedup over the RK4 reference.

For reference, the transient-energy-function (PEBS) approach of Athay et
al.~\cite{athay1979transient} with the conductance path-integral terms attains a mean CCT error
of $18.7$ ms with a conservative bias, and an IMEAC-style individual-machine
criterion~\cite{wang2018individual} evaluated on the same held-out buses with the same
$16$-iteration binary search yields $9.6$ ms. The operator's \ccterr{} ms mean absolute error is
larger than both classical criteria and carries a conservative bias ($-21$ ms mean signed error),
dominated by bus 12 (reference CCT $453$ ms, estimated $248$ ms), the bus with the largest CCT,
where the binary search lands between multiple local stable-to-unstable crossings of the
operator's criterion. Classical direct methods remain the more accurate CCT estimators; the
operator's value is that it additionally returns the full trajectory and a conformal band without
any fault-on or post-fault integration. Fig.~\ref{fig:trajectory} shows predicted versus true COI
trajectories for a representative near-boundary scenario."""))

R.append(('tab-cct-del', r"""\begin{table}[t]
\centering
\caption{Per-bus CCT comparison on the $12$ held-out fault buses: event-aligned RK4 reference
($0.5$ ms step, $16$-iteration search), operator estimate, and signed error (ms).}
\label{tab:cct}
\small
\begin{tabular}{lcccccccccccc}
\toprule
Fault bus & 3 & 4 & 5 & 11 & 12 & 18 & 21 & 22 & 23 & 24 & 37 & 38 \\
\midrule
Reference CCT (ms) & \cctbusref \\
Operator CCT (ms) & \cctbuspred \\
Signed error (ms) & \cctbuserr \\
\bottomrule
\end{tabular}
\end{table}

""", r""""""))

R.append(('fig-cct-del', r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_cct.pdf}
\caption{Predicted versus true critical clearing time for the $12$ held-out fault buses (bus
numbers annotated; dashed line: diagonal; shaded band: $\pm$\ccterr{} ms mean absolute error,
a reference band from the observed MAE, not a prediction interval). The operator recovers the
CCT to within \ccterr{} ms on average; the largest per-bus error is explained in the text.}
\label{fig:cct}
\end{figure}

""", r""""""))

R.append(('r-6.2', r"""\subsection{Effect of the physics-informed encoding}
Table~\ref{tab:ablation} reports the ablation on the fault encoding. Replacing the
physics-informed acceleration vector with a one-hot bus index degrades the stable-trajectory
RMSE from \rmseAll{}$^\circ$ to $141.4^\circ$ and the classification accuracy from \pctACC{} to
$88.0\%$, because a one-hot encoding provides no signal for unseen fault buses. The one-hot
baseline also overfits at $p{=}512$: its RMSE exceeds the \randRMSE{}$^\circ$ RMSE that a
uniformly random predictor in $[-180^\circ,180^\circ]$ attains on the same stable test
trajectories (an actual baseline computed on the test set, not an analytic shortcut), and
reducing its capacity does not recover meaningful generalization. This indicates that the
physics-informed encoding is the mechanism enabling generalization across fault locations
(Fig.~\ref{fig:ablation}).

A GCN classifier---a standard topology-aware TSA baseline---operating on raw node features
(voltage magnitudes and angles, loads, and a fault flag) achieves only $76.0\%$ classification
accuracy, versus \pctACC{} for the severity-encoded operator. This comparison is not
architecture-controlled: both the input features and the task differ (the GCN outputs a label,
the operator a trajectory). It nevertheless shows that the severity encoding carries more
information than raw observables for this task, in addition to enabling generalization across
fault locations. A stronger graph baseline with three residual graph layers and concatenated
mean/max/sum pooling attains $77.3\%\pm3.9\%$ (five seeds) on the same raw features, indicating
that the gap is not an artifact of a weak graph architecture. A second graph baseline, a GCN that
regresses the CCT directly from the same raw node features (trained on the $27$ training fault
buses, evaluated on the $12$ held-out buses, five seeds), attains a mean CCT error of $48.0$ ms
(max $224.4$ ms), versus \ccterr{} ms (max $204.9$ ms) for the severity-encoded operator. To
separate the input representation from the architecture, a \emph{severity-MLP} receiving exactly
the same encoding as the operator attains \sevMLPacc{} classification accuracy and
\sevMLPcct{} ms mean CCT error (Table~\ref{tab:ablation}): the encoding, not the DeepONet
architecture, is the dominant factor for generalization to unseen fault locations, while the
operator retains the advantages of continuous-time evaluation and the low-rank structure
(Sections~\ref{sec:sampleeff} and \ref{sec:discussion}). Neither graph baseline produces a
trajectory or a conformal error band. A recurrent (GRU) sequence-to-sequence baseline, mapping
the same severity encoding to the trajectory through a sequential decoder rather than an operator
contraction, attains $57.3^\circ$ RMSE and $93.9\%$ accuracy (Table~\ref{tab:ablation}), worse than
the operator's \rmseAll{}$^\circ$: the low-rank operator structure outperforms a generic recurrent
decoder on this problem.""",
r"""\subsection{Effect of the physics-informed encoding}
Table~\ref{tab:ablation} reports the ablation on the fault encoding. Replacing the
physics-informed acceleration vector with a one-hot bus index degrades the stable-trajectory
RMSE from \rmseAll{}$^\circ$ to $141.4^\circ$ and the classification accuracy from \pctACC{} to
$88.0\%$, because a one-hot encoding provides no signal for unseen fault buses; the one-hot
baseline also overfits at $p{=}512$---its RMSE exceeds the \randRMSE{}$^\circ$ attained by a
uniformly random predictor in $[-180^\circ,180^\circ]$ on the same stable test trajectories, an
actual baseline computed on the test set---and reducing its capacity does not recover meaningful
generalization. The physics-informed encoding is therefore the mechanism enabling generalization
across fault locations.

A GCN classifier operating on raw node features (voltage magnitudes and angles, loads, and a
fault flag) achieves only $76.0\%$ accuracy versus \pctACC{} for the severity-encoded operator;
a stronger graph baseline with three residual layers and concatenated mean/max/sum pooling attains
$77.3\%\pm3.9\%$ (five seeds), so the gap is not an artifact of a weak graph architecture, and a
GCN regressing the CCT from the same raw features attains a mean CCT error of $48.0$ ms (max
$224.4$ ms) versus \ccterr{} ms (max $204.9$ ms). These comparisons are not
architecture-controlled (the GCNs output a label or a scalar, the operator a trajectory), but
they show the severity encoding carries more information than raw observables. To separate the
input representation from the architecture, a \emph{severity-MLP} receiving exactly the same
encoding attains \sevMLPacc{} accuracy and \sevMLPcct{} ms mean CCT error
(Table~\ref{tab:ablation}): the encoding, not the DeepONet architecture, is the dominant factor
for generalization to unseen fault locations, while the operator retains continuous-time
evaluation and the low-rank structure (Sections~\ref{sec:sampleeff} and \ref{sec:discussion}). A
recurrent (GRU) sequence-to-sequence baseline mapping the same encoding through a sequential
decoder attains $57.3^\circ$ RMSE and $93.9\%$ accuracy, worse than the operator's
\rmseAll{}$^\circ$: the low-rank operator structure outperforms a generic recurrent decoder."""))

R.append(('fig-ablation-del', r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_ablation.pdf}
\caption{Ablation and baseline comparison (bars, left axis: pooled stable-trajectory RMSE; line
with markers, right axis: classification accuracy; single seed-0 models, $p{=}512$ for the
DeepONet variants, physics-residual bars at the $\lambda$ values printed in the figure). The
physics-informed encoding is essential: a one-hot encoding cannot generalize to unseen fault
buses, while the pointwise MLP baseline is competitive with the operator on raw accuracy
($96.9\%$ vs.\ \pctACC{}); the MLP is pointwise in time and does not exploit the low-rank
operator structure, but its CCT and conformal numbers are evaluated under the same protocol in
Table~\ref{tab:ablation} rather than declared impossible.}
\label{fig:ablation}
\end{figure}

""", r""""""))

R.append(('r-6.3', r"""\subsection{Generalization to $N{-}1$ and $N{-}2$ topologies}
We train a single operator ($p{=}512$) on a mix of $N{-}0$ and $N{-}1$ scenarios ($9990$
training scenarios: $3240$ $N{-}0$ and $6750$ $N{-}1$, with line trips drawn from $25$ training
lines) using the extended encoding of Section~\ref{sec:method}, and evaluate generalization along
two held-out axes: unseen fault buses and unseen trip lines ($10$ lines). Table~\ref{tab:n1}
reports the results. On held-out line trips the operator achieves $33.6^\circ$ stable-trajectory
RMSE and $95.9\%$ classification accuracy, demonstrating that the post-fault severity encoding
lets the operator generalize to topologies never seen in training. On held-out fault buses the
RMSE is $64.3^\circ$ and the accuracy is $93.5\%$, reflecting the harder combined task; for
reference, the $N{-}0$-only operator attains $36.2^\circ$ on the corresponding held-out fault
buses, so the multi-topology operator trades some $N{-}0$ accuracy for topology
generalization.

The same encoding extends to $N{-}2$ contingencies (two lines tripped): on held-out line pairs
the operator attains $43.4^\circ$ stable-trajectory RMSE and $92.9\%$ classification accuracy
(Table~\ref{tab:n1}), with no architecture change, because the post-fault severity vector is
simply evaluated from the admittance matrix with both lines removed. The multi-topology operator
also estimates the CCT on held-out combinations under the corrected protocol: a mean absolute
error of $19.4$ ms on unseen trip lines ($10$ lines $\times$ $2$ fault buses), $31.9$ ms on
held-out fault buses (Table~\ref{tab:n1}), and \cctNtwo{} ms on held-out $N{-}2$ line pairs
(sampled). The $N{-}2$ evaluation comprises $972$ scenarios over $27$ training fault buses and
$150$ held-out line pairs, giving $863$ unique bus--pair combinations; split-conformal coverage
on these splits is reported in Section~\ref{sec:results}.""",
r"""\subsection{Generalization to $N{-}1$ and $N{-}2$ topologies}
We train a single operator ($p{=}512$) on a mix of $N{-}0$ and $N{-}1$ scenarios ($9990$
training scenarios: $3240$ $N{-}0$ and $6750$ $N{-}1$, line trips drawn from $25$ training lines)
using the extended encoding of Section~\ref{sec:method}, and evaluate on two held-out axes:
unseen fault buses and unseen trip lines ($10$ lines). On held-out line trips the operator
achieves $33.6^\circ$ stable-trajectory RMSE and $95.9\%$ accuracy, demonstrating that the
post-fault severity encoding lets the operator generalize to topologies never seen in training;
on held-out fault buses the RMSE is $64.3^\circ$ and the accuracy $93.5\%$ (Table~\ref{tab:n1}),
reflecting the harder combined task. The same encoding extends to $N{-}2$ contingencies with no
architecture change: on held-out line pairs the operator attains $43.4^\circ$ RMSE and $92.9\%$
accuracy, because the post-fault severity vector is simply evaluated from the admittance matrix
with both lines removed. The multi-topology operator also estimates the CCT on held-out
combinations: $19.4$ ms mean absolute error on unseen trip lines, $31.9$ ms on held-out fault
buses, and \cctNtwo{} ms on held-out $N{-}2$ line pairs (sampled); split-conformal coverage on
these splits is reported in Section~\ref{sec:results}."""))

R.append(('r-6.4', r"""\subsection{Sample efficiency of the operator}
\label{sec:sampleeff}
A motivation for the operator architecture is its low-rank inductive bias. We verify this with a
data-efficiency experiment, training the DeepONet and the pointwise MLP baseline on $10\%$,
$25\%$, $50\%$, and $100\%$ of the training scenarios (three seeds each) and evaluating on the
full held-out test set. At $10\%$ of the data the operator attains $56.1^\circ$ RMSE versus
$65.2^\circ$ for the MLP (a $\approx9^\circ$ advantage; across-seed standard deviations
$14.0^\circ$ and $2.6^\circ$); at $25\%$ the gap is $\approx8^\circ$ ($46.4^\circ$ versus
$54.8^\circ$; $\pm5.7^\circ$ and $\pm2.1^\circ$); at $50\%$ and $100\%$ the gap narrows to
$\approx2$--$3^\circ$ ($49.9^\circ$ versus $51.6^\circ$, and $38.8^\circ$ versus $41.5^\circ$;
the non-monotonicity at intermediate fractions is within the across-seed standard deviation).
The low-rank structure therefore confers a modest sample-efficiency advantage in the low-data
regime, where fast TSA surrogates are most valuable; with three seeds per fraction and a
$14.0^\circ$ across-seed spread at $10\%$ data, the low-data advantage is indicative rather than
statistically established, and we do not claim it transfers beyond this system and protocol. The
sample-efficiency numbers use the base $p{=}128$ architecture averaged over three seeds, whereas
Table~\ref{tab:ablation} reports the $p{=}512$ architecture, which explains the different
full-data values ($38.8^\circ$/$41.5^\circ$ here versus \rmseAll{}$^\circ$/$43.1^\circ$ in the main
experiment).""",
r"""\subsection{Sample efficiency of the operator}
\label{sec:sampleeff}
A motivation for the operator architecture is its low-rank inductive bias. We train the DeepONet
and the pointwise MLP baseline on $10\%$, $25\%$, $50\%$, and $100\%$ of the training scenarios
(three seeds each) and evaluate on the full held-out test set. At $10\%$ of the data the operator
attains $56.1^\circ$ RMSE versus $65.2^\circ$ for the MLP (a $\approx9^\circ$ advantage); at
$25\%$ the gap is $\approx8^\circ$ ($46.4^\circ$ versus $54.8^\circ$); at $50\%$ and $100\%$ it
narrows to $\approx2$--$3^\circ$. The low-rank structure therefore confers a modest
sample-efficiency advantage in the low-data regime, where fast TSA surrogates are most valuable;
with three seeds per fraction and a $14.0^\circ$ across-seed spread at $10\%$ data, the low-data
advantage is indicative rather than statistically established. These numbers use the base
$p{=}128$ architecture averaged over three seeds, whereas Table~\ref{tab:ablation} reports
$p{=}512$, which explains the different full-data values."""))

R.append(('r-6.5', r"""\subsection{Extensions to higher-fidelity generator models and damping}
The classical model treats each machine as a constant voltage behind a transient reactance. To
verify that the approach carries over to a more realistic generator representation, we repeat the
experiment with the one-axis (flux-decay) model, in which the q-axis transient EMF $E'_q$ obeys
dynamics driven by the field voltage and the d-axis armature reaction (a third-order per-machine
model~\cite{sauer1998dynamics}, Ch.~8): the architecture, encoding, and training protocol are
unchanged, the operator is \emph{retrained} on data from the one-axis simulator, and the data
protocol is identical to the classical case---the same $12$ held-out fault buses, the same
clearing-time range, and the same $p{=}512$ architecture---so only the ground-truth trajectories
change. The severity encoding needs no modification; the one-axis model yields lower critical
clearing times than the classical model, as expected from flux decay. On this model the retrained
operator attains $38.6^\circ$ pooled stable-trajectory RMSE
($1.8^\circ$ fault-on, $39.2^\circ$ post-fault), $94.0\%$ classification accuracy under the
max-over-time criterion, a mean CCT error of $23.3$ ms, and a conformal band of $121.3^\circ$
half-width with $0.903$ coverage ($\alpha{=}0.1$).

Adding a first-order automatic voltage regulator (a simplified IEEE DC1A exciter with $K_A{=}20$,
$T_A{=}0.2$ s and rate limits) yields a fourth-order per-machine model whose critical clearing
times are lower than under constant field voltage (a known negative-damping effect). On this
exciter model the operator is likewise retrained with the same encoding; it attains
$25.4^\circ$ pooled stable-trajectory RMSE ($2.2^\circ$ fault-on, $25.5^\circ$ post-fault),
$94.0\%$ classification accuracy, and conformal coverage $0.896$ with half-width $127.7^\circ$---
the lowest absolute RMSE of the three generator models, largely because voltage regulation
smooths the post-fault oscillation. The exciter model has sparser stable data, so the larger
$p{=}512$ network overfits it slightly; at $p{=}128$ it attains $19.7^\circ$. The residual in
these extensions remains the simplified swing-equation residual of \eqref{eq:physloss}: it is not
the full higher-order DAE residual, and no claim of residual-constrained higher-order dynamics is
made.

We also probe damped dynamics: adding uniform damping $D{=}1$ to the classical swing equations
leaves the severity encoding unchanged---it is evaluated at the pre-fault equilibrium, where the
damping term vanishes---so the same architecture is retrained on damped data without encoding
changes. The fault-on region improves to $1.8^\circ$ pooled RMSE, but the post-fault pooled RMSE
rises to $46.4^\circ$ (versus \rmsePost{}$^\circ$ undamped) and classification drops to $93.4\%$
(CCT error $30.8$ ms). Near-boundary damped trajectories decay slowly and linger at high angle
spreads, so small errors in the predicted decay rate accumulate over the $4$ s window and produce
a heavier error tail (maximum worst-case error $638^\circ$). The split-conformal band remains
approximately valid (mean coverage $0.90\pm0.02$ over five calibration splits), though the
heavier tail makes it more seed-sensitive than in the undamped case. Damping therefore does not
simplify the near-boundary prediction task.""",
r"""\subsection{Extensions to higher-fidelity generator models and damping}
To verify that the approach carries over to more realistic generator representations, we retrain
the operator---architecture, encoding, and protocol unchanged, with the same $12$ held-out fault
buses---on data from higher-fidelity simulators, so only the ground-truth trajectories change.
On the one-axis (flux-decay) model (a third-order per-machine model~\cite{sauer1998dynamics},
Ch.~8) the retrained operator attains $38.6^\circ$ pooled stable-trajectory RMSE ($1.8^\circ$
fault-on, $39.2^\circ$ post-fault), $94.0\%$ classification accuracy, a mean CCT error of
$23.3$ ms, and a conformal band of $121.3^\circ$ half-width with $0.903$ coverage
($\alpha{=}0.1$). Adding a first-order automatic voltage regulator (a simplified IEEE DC1A
exciter with $K_A{=}20$, $T_A{=}0.2$ s and rate limits) yields a fourth-order model on which the
operator attains $25.4^\circ$ pooled RMSE ($2.2^\circ$ fault-on, $25.5^\circ$ post-fault),
$94.0\%$ accuracy, and conformal coverage $0.896$ with half-width $127.7^\circ$---the lowest
absolute RMSE of the three generator models, largely because voltage regulation smooths the
post-fault oscillation.

Adding uniform damping $D{=}1$ leaves the severity encoding unchanged---it is evaluated at the
pre-fault equilibrium, where the damping term vanishes---so the same architecture is retrained on
damped data. The fault-on region improves to $1.8^\circ$ pooled RMSE, but the post-fault RMSE
rises to $46.4^\circ$ (versus \rmsePost{}$^\circ$ undamped) and classification drops to $93.4\%$
(CCT error $30.8$ ms): near-boundary damped trajectories decay slowly and linger at high angle
spreads, so small errors in the predicted decay rate accumulate over the $4$ s window and produce
a heavier error tail (maximum worst-case error $638^\circ$). The split-conformal band remains
approximately valid (mean coverage $0.90\pm0.02$ over five calibration splits). Damping therefore
does not simplify the near-boundary prediction task."""))

R.append(('r-6.6', r"""\subsection{Conformal bands}
The split-conformal band achieves empirical coverage close to nominal at every level, computed
with the order-statistic quantile of Section~\ref{sec:method} on the $N_{\mathrm{eva}}{=}178$
evaluation scenarios: $91.6\%$ ($163/178$, $95\%$ CI $87$--$95\%$) at the $95\%$ target,
$88.2\%$ ($157/178$, CI $83$--$93\%$) at $90\%$, $67.4\%$ ($120/178$) at $70\%$, and $43.8\%$
($78/178$) at $50\%$; these counts are consistent with the nominal levels and the bound is not
anticonservative (Fig.~\ref{fig:conformal}). The same split calibration on the multi-topology
operators yields empirical coverage of $0.915$ on both held-out $N{-}1$ fault buses and line
trips (calibration/evaluation sizes $553/553$ and $494/495$; $95\%$ CIs $\pm0.02$--$0.03$), and
$0.872$ and $0.852$ on held-out $N{-}2$ buses and line pairs (sizes $133/133$ and $141/142$; CIs
$\pm0.04$--$0.06$); interval half-widths are $121$--$181^\circ$. A Mondrian (grouped)
calibration~\cite{vovk2005algorithmic} by the post-fault peak of the pairwise spread---defined as
$\max_{t\ge t_c}\bigl(\max_i \tilde\delta_i(t)-\min_i \tilde\delta_i(t)\bigr)$ of the
\emph{true} trajectory; an offline diagnostic because the grouping uses ground-truth labels and
cannot be replicated at inference time---shows, at $\alpha{=}0.1$, that the band half-width grows
toward the stability boundary, from \mondA{}$^\circ$ for scenarios peaking below $45^\circ$
(group $48/51$ calibration/evaluation, coverage $0.863$), through $94.2^\circ$ for $45$--$90^\circ$
($44/52$, $0.923$), to \mondC{}$^\circ$ for those peaking above $90^\circ$ ($86/75$, $0.800$;
annotated in Fig.~\ref{fig:conformal}): coverage is approximately nominal away from the boundary
and below nominal for the near-boundary group. The band is therefore wide precisely where the
stability decision is most critical; it is a conservative screening envelope, not a tight
predictive interval.

The directly calibrated pairwise-difference score $R_{\mathrm{pw}}$ bounds the pairwise angle
difference---the exact quantity appearing in the stability criterion---to \pairq{}$^\circ$ at
$\alpha{=}0.1$ (coverage \paircov{}), below the $\pi$ threshold. Making a stability decision from
this bound still requires the \emph{predicted} spread to satisfy
$\mathrm{spread} + \hat q_{\mathrm{pw}} < \pi$; on the evaluation half this rule resolves
\decfrac{} of the predicted-stable scenarios at $\alpha{=}0.1$ (rising to $31\%$ at
$\alpha{=}0.3$ and $67\%$ at $\alpha{=}0.5$ as $\hat q_{\mathrm{pw}}$ tightens from
$90.5^\circ$ to $53.3^\circ$), and the empirical wrong-release rate of the
predicted-stable gate (fraction of scenarios predicted stable that are in fact unstable) is
\wrrate{} ($\wrnum{}$ of \wrdem{}), so any deployment must retain a TDS fallback for the
unresolved fraction. By contrast, per-machine calibration with Bonferroni/\v{S}id\'ak
multiplicity corrections does not tighten the bound (pairwise $734^\circ$): each machine's
worst-over-time score has heavy tails, so the $\alpha/n$ correction pushes the per-machine
quantiles deep into the tail. Calibrating instead on the validation set (training-bus scenarios)
and evaluating coverage on the held-out buses yields only $0.61$ at the nominal $0.90$ target:
the held-out-bus distribution is genuinely harder, so calibrating on held-out scenarios is
necessary rather than a leakage concern, provided future scenarios are exchangeable with the
calibration set. For deployment, rolling calibration windows refreshed on recent contingencies
and domain-shift-robust conformal variants are natural directions; the calibration does not
transfer across operating points (the $N{-}0$-calibrated interval covers only $0.51$ of stable
$0.8$-load scenarios and $0.72$ at $0.9$ load), consistent with the operating-point limitation of
Section~\ref{sec:discussion}---the binding obstacle is the operating-point shift itself, not the
calibration split.

We also examined alternative scores (details in the code repository): an amplitude-normalized
(relative) score bounds the error to $2.3\%$ of each scenario's peak amplitude (coverage $0.854$);
a first-swing-window score---the worst error within each machine's first swing, an offline
diagnostic because the swing window is defined by the ground-truth trajectory---bounds
the first swing to $70.8^\circ$ at $\alpha{=}0.1$ (coverage $0.865$), tighter than the
full-trajectory \qhatdeg{}$^\circ$; a time-averaged score yields $47.7^\circ$ with $0.893$ coverage
at the cost of bounding the time average rather than the uniform-in-time error. In the gated
regime where the band is applied only after a predicted-stable decision, we report the
protocol-correct numbers on the \emph{disjoint evaluation half only}: among its predicted-stable
scenarios the gate purity (true stable among predicted stable) is \gatepurity{}, the coverage
over all predicted-stable evaluation scenarios is \gatecovAll{}, and the coverage over the
true-stable-and-predicted-stable subset is \gatecovTS{} (\gatenum{} scenarios)---the last
quantity is an oracle-conditioned diagnostic (it conditions on ground-truth labels) and is not a
deployment guarantee, since conditioning on the prediction breaks exchangeability. Adaptive and
locally adaptive conformal methods~\cite{gibbs2021adaptive, lei2014distribution} and conformalized
quantile regression~\cite{romano2019conformalized} are candidates for further reducing the
near-boundary conservatism and are left as future work rather than evaluated here; time-percentile
scores are also left to future work.""",
r"""\subsection{Conformal bands}
The split-conformal band achieves empirical coverage close to nominal at every level on the
$N_{\mathrm{eva}}{=}178$ evaluation scenarios: $91.6\%$ ($163/178$) at the $95\%$ target,
$88.2\%$ ($157/178$, CI $83$--$93\%$) at $90\%$, $67.4\%$ ($120/178$) at $70\%$, and $43.8\%$
($78/178$) at $50\%$ (Fig.~\ref{fig:conformal}); the same split calibration on the multi-topology
operators yields coverage of $0.915$ on both held-out $N{-}1$ fault buses and line trips and
$0.872$/$0.852$ on held-out $N{-}2$ buses and line pairs, with interval half-widths
$121$--$181^\circ$. A Mondrian (grouped)
calibration~\cite{vovk2005algorithmic} by the post-fault peak of the pairwise spread---defined as
$\max_{t\ge t_c}\bigl(\max_i \tilde\delta_i(t)-\min_i \tilde\delta_i(t)\bigr)$ of the
\emph{true} trajectory, an offline diagnostic because the grouping uses ground-truth labels and
cannot be replicated at inference time---shows, at $\alpha{=}0.1$, that the band half-width grows
toward the stability boundary, from \mondA{}$^\circ$ for scenarios peaking below $45^\circ$
(group $48/51$ calibration/evaluation, coverage $0.863$), through $94.2^\circ$ for $45$--$90^\circ$
($44/52$, $0.923$), to \mondC{}$^\circ$ above $90^\circ$ ($86/75$, $0.800$): coverage is
approximately nominal away from the boundary and below nominal for the near-boundary group. The
band is wide precisely where the stability decision is most critical; it is a conservative
screening envelope, not a tight predictive interval.

The directly calibrated pairwise-difference score $R_{\mathrm{pw}}$ bounds the pairwise angle
difference---the exact quantity in the stability criterion---to \pairq{}$^\circ$ at
$\alpha{=}0.1$ (coverage \paircov{}), below the $\pi$ threshold. Making a decision from this
bound still requires the \emph{predicted} spread to satisfy
$\mathrm{spread} + \hat q_{\mathrm{pw}} < \pi$; on the evaluation half this rule resolves
\decfrac{} of the predicted-stable scenarios at $\alpha{=}0.1$ (rising to $31\%$ at
$\alpha{=}0.3$ and $67\%$ at $\alpha{=}0.5$ as $\hat q_{\mathrm{pw}}$ tightens from $90.5^\circ$
to $53.3^\circ$), and the empirical wrong-release rate of the predicted-stable gate is \wrrate{}
($\wrnum{}$ of \wrdem{}), so any deployment must retain a TDS fallback for the unresolved
fraction. Per-machine calibration with Bonferroni/\v{S}id\'ak corrections does not tighten the
bound (pairwise $734^\circ$), and calibrating on the validation set instead of held-out scenarios
yields only $0.61$ coverage at the nominal $0.90$ target: the held-out-bus distribution is
genuinely harder, so calibrating on held-out scenarios is necessary rather than a leakage
concern, provided future scenarios are exchangeable with the calibration set. The calibration
does not transfer across operating points (the $N{-}0$-calibrated interval covers only $0.51$ of
stable $0.8$-load scenarios and $0.72$ at $0.9$ load), consistent with the operating-point
limitation of Section~\ref{sec:discussion}. In the gated regime (band applied only after a
predicted-stable decision, evaluated on the disjoint half): gate purity \gatepurity{}, coverage
over predicted-stable scenarios \gatecovAll{}, and oracle-conditioned coverage \gatecovTS{}
(\gatenum{} scenarios; the last conditions on ground-truth labels and is not a deployment
guarantee, since conditioning on the prediction breaks exchangeability). Alternative scores
(amplitude-normalized $2.3\%$, first-swing-window $70.8^\circ$, time-averaged $47.7^\circ$; see
the repository) and adaptive/locally adaptive conformal
methods~\cite{gibbs2021adaptive, lei2014distribution} and conformalized quantile
regression~\cite{romano2019conformalized} are candidates for reducing near-boundary conservatism
and are left as future work."""))

R.append(('r-6.7', r"""\subsection{The negative result: input encoding versus residual loss}
The auxiliary physics-residual loss does not help in this setting. The residual reported here is
the correctly scaled form of \eqref{eq:physloss} (a scaling check confirmed that omitting the
synchronous-speed factor $\omega_s$ on the power-mismatch terms is a misspecification); with it,
the pooled stable-trajectory RMSE is $39.2^\circ$ at $\lambda{=}0.001$ (versus \rmseAll{}$^\circ$
for the data-only operator) and $76.5^\circ$ at $\lambda{=}0.01$ (Table~\ref{tab:ablation}): no
improvement at the mildest weight and harmful at larger weights. Denser collocation (every output
time point instead of every fifth) likewise does not help ($48.9^\circ$ pooled, $92.9\%$
accuracy), so the negative result is not an artifact of the collocation stride. We attribute this
to the fact that, without the correct post-fault initial condition, the swing-equation residual
is a weak constraint satisfied by many trajectories, so it adds regularization without
information. This conclusion is scoped to the present system, protocol, loss weights, and
architecture; it does not rule out residual-based physics regularization with hard state chaining
or adaptive weights. Here, physics is injected more effectively through the input encoding
derived from the swing equation than through a residual loss.""",
r"""\subsection{The negative result: input encoding versus residual loss}
The auxiliary physics-residual loss does not help in this setting: with the correctly scaled
residual of \eqref{eq:physloss}, the pooled stable-trajectory RMSE is $39.2^\circ$ at
$\lambda{=}0.001$ (versus \rmseAll{}$^\circ$ for the data-only operator) and $76.5^\circ$ at
$\lambda{=}0.01$ (Table~\ref{tab:ablation})---no improvement at the mildest weight and harmful at
larger weights---and denser collocation does not help either ($48.9^\circ$ pooled, $92.9\%$
accuracy). Without the correct post-fault initial condition the swing-equation residual is a weak
constraint satisfied by many trajectories, so it adds regularization without information. This
conclusion is scoped to the present system, protocol, weights, and architecture; here, physics is
injected more effectively through the input encoding derived from the swing equation than through
a residual loss."""))

R.append(('r-6.8', r"""\subsection{Near-boundary sensitivity and architecture ablations}
To quantify how much of the trajectory error is tied to the stability boundary, we train on
boundary-aware data (clearing times sampled within $[0.4,1.2]\times$CCT per fault bus) and find
the pooled post-fault RMSE rises from \rmsePost{}$^\circ$ (uniform) to $70.8^\circ$
(boundary-aware): near-boundary trajectories respond sharply to small parameter changes at the
stability bifurcation, which is the regime the conformal band is designed to flag (the rise can
reflect both intrinsic difficulty and the shifted training/test sampling; we report the
observation without attributing a single cause). Stratifying the test scenarios by their distance
to the TDS CCT quantifies this: more than $100$ ms inside the boundary the stable pooled RMSE is
$19.9^\circ$ with $8.8\%$ misclassification over the whole set; within $30$ ms of the boundary
the misclassification rate rises to $20.0\%$ ($n{=}60$); and within $10$ ms it reaches $46.2\%$
($n{=}13$; the exact $95\%$ binomial interval is approximately $19$--$75\%$, so this last figure
is descriptive). The sharpest decisions are the least reliable, which is precisely the regime the
conformal band is designed to flag. Replacing the branch network with self-attention over the
machines does not improve accuracy: at $p{=}512$ the attention branch attains $116.1^\circ$ RMSE
and $92.8\%$ accuracy (versus \rmseAll{}$^\circ$ and \pctACC{}), consistent with the severity
encoding already capturing the machine-interaction structure; a Fourier-neural-operator trunk
likewise gives no improvement ($47.4^\circ$ RMSE, $92.6\%$ accuracy;
Table~\ref{tab:ablation}). This supports the choice of a
simple DeepONet with a physics-informed encoding for this problem.""",
r"""\subsection{Near-boundary sensitivity and architecture ablations}
Training on boundary-aware data (clearing times sampled within $[0.4,1.2]\times$CCT per fault
bus) raises the pooled post-fault RMSE from \rmsePost{}$^\circ$ (uniform) to $70.8^\circ$
(boundary-aware): near-boundary trajectories respond sharply to small parameter changes at the
stability bifurcation, the regime the conformal band is designed to flag (the rise can reflect
both intrinsic difficulty and the shifted sampling; we report the observation without attributing
a single cause). Stratifying the test scenarios by their distance to the TDS CCT quantifies this:
more than $100$ ms inside the boundary the misclassification rate is $8.8\%$; within $30$ ms it
rises to $20.0\%$ ($n{=}60$); and within $10$ ms it reaches $46.2\%$ ($n{=}13$, descriptive). The
sharpest decisions are the least reliable. Replacing the branch network with self-attention does
not improve accuracy ($116.1^\circ$ RMSE, $92.8\%$ accuracy at $p{=}512$, versus
\rmseAll{}$^\circ$ and \pctACC{}), consistent with the severity encoding already capturing the
machine-interaction structure, and a Fourier-neural-operator trunk likewise gives no improvement
($47.4^\circ$ RMSE, $92.6\%$ accuracy; Table~\ref{tab:ablation}). This supports the choice of a
simple DeepONet with a physics-informed encoding."""))

R.append(('r-6.9', r"""\subsection{Robustness to measurement noise}
The physics-informed encoding $a_i(k)=(P_{mi}-P_{ei}^{\mathrm{f}})/M_i$ is itself computed from
estimates of mechanical power, inertia, and fault-on electrical power, which carry measurement
error. To assess robustness, we perturb the encoded acceleration vector with zero-mean Gaussian
noise (standard deviation proportional to each machine's scale) and re-evaluate the operator. At
$5\%$ noise the pooled stable RMSE rises from \rmseAll{}$^\circ$ to $115.4^\circ$ and the
classification accuracy drops from \pctACC{} to $89.1\%$; at $10\%$ noise, RMSE is $152.3^\circ$
and accuracy $82.9\%$; at $20\%$ noise, $213.6^\circ$ and $73.0\%$. The max-over-time criterion
is sensitive to noise-induced trajectory distortions near the boundary, so the operator is only
moderately robust to \emph{feature} noise in its input encoding; this does not cover
correlated or biased parameter-estimation errors (e.g., a systematic inertia error) or
errors in the fault location itself, which remain to be tested, and the conformal coverage under
input noise is likewise left to future work.""",
r"""\subsection{Robustness to measurement noise}
The encoding $a_i(k)=(P_{mi}-P_{ei}^{\mathrm{f}})/M_i$ is itself computed from estimates of
mechanical power, inertia, and fault-on electrical power, which carry measurement error.
Perturbing the encoded acceleration vector with zero-mean Gaussian noise, the pooled stable RMSE
rises from \rmseAll{}$^\circ$ to $115.4^\circ$ and accuracy drops from \pctACC{} to $89.1\%$ at
$5\%$ noise ($152.3^\circ$/$82.9\%$ at $10\%$, $213.6^\circ$/$73.0\%$ at $20\%$). The operator is
therefore only moderately robust to \emph{feature} noise in its input encoding; correlated or
biased parameter-estimation errors and conformal coverage under input noise remain to be tested."""))

R.append(('r-6.10', r"""\subsection{Scalability to the IEEE 118-bus system}
To probe scalability beyond a single test system, we evaluate the operator on a custom dynamic
test case built on the IEEE 118-bus static network: the $54$ machine parameters ($H$, $x'_{d}$)
are taken from the grouped tables of the Demetriou et al.~\cite{demetriou2017dynamic} modified
test system and assigned to pandapower's case118 generators by rated capacity. No per-machine
bus mapping and no per-unit base conversion are applied, so this is a synthetic dynamic case on
the IEEE 118-bus network rather than a reproduction of the Demetriou system; results on it probe
scalability of the \emph{method}, not of any benchmark system. The architecture is retrained on
118-bus data; with $24$ fault buses held out entirely from training ($3760$ training and $960$
test scenarios, clearing times in $[0.005,0.2]$ s and the system-adapted CCT search interval of
Section~\ref{sec:method}), the retrained operator attains \pctACCb{} classification accuracy and
$121.9^\circ$ stable-trajectory RMSE. The test set is $25.4\%$ stable, so a majority-class
(unstable) baseline would score $74.6\%$; the operator's \pctACCb{} therefore reflects real
discrimination, and it is close to the 39-bus \pctACC{} despite a system with $5.4\times$ more
generators ($54$ vs.\ $10$) and $3.0\times$ more buses ($118$ vs.\ $39$). Classification
scalability does not imply trajectory-level scalability: the higher trajectory RMSE reflects the
larger state space and the much lower critical clearing times of the assigned 118-bus case
($0.008$--$0.135$ s, versus $0.13$--$0.58$ s for the 39-bus system), a consequence of the
low-inertia machine parameters in the assigned set. The CCT error on the $24$ held-out buses is
$13.3$ ms mean absolute (max $54.7$ ms), large relative to the $0.008$--$0.135$ s CCT scale.
The split-conformal interval attains $0.918$ coverage at $\alpha{=}0.1$, but with a half-width
of $596.4^\circ$---far wider than the 39-bus band and larger than the $180^\circ$ stability
threshold, so on the 118-bus case the interval validates the coverage mechanism but is vacuous
for the stability decision; the directly calibrated pairwise score and richer calibration data
are required before the bound is operationally useful at this scale.""",
r"""\subsection{Scalability to the IEEE 118-bus system}
To probe scalability beyond a single test system, we evaluate the operator on a custom dynamic
test case built on the IEEE 118-bus static network: the $54$ machine parameters ($H$, $x'_{d}$)
are taken from the grouped tables of Demetriou et al.~\cite{demetriou2017dynamic} and assigned to
pandapower's case118 generators by rated capacity. No per-machine bus mapping and no per-unit
base conversion are applied, so this is a synthetic dynamic case on the IEEE 118-bus network
rather than a reproduction of the Demetriou system; results on it probe scalability of the
\emph{method}, not of any benchmark system. With $24$ fault buses held out entirely from training
($3760$ training and $960$ test scenarios), the retrained operator attains \pctACCb{}
classification accuracy and $121.9^\circ$ stable-trajectory RMSE. The test set is $25.4\%$
stable, so a majority-class baseline would score $74.6\%$; \pctACCb{} therefore reflects real
discrimination, close to the 39-bus \pctACC{} despite $5.4\times$ more generators and $3.0\times$
more buses. The higher trajectory RMSE reflects the larger state space and the much lower CCTs of
the assigned 118-bus case ($0.008$--$0.135$ s), and the CCT error on the $24$ held-out buses is
$13.3$ ms mean absolute (max $54.7$ ms). The split-conformal interval attains $0.918$ coverage at
$\alpha{=}0.1$, but with a half-width of $596.4^\circ$---far wider than the 39-bus band and larger
than the $180^\circ$ stability threshold, so on the 118-bus case the interval validates the
coverage mechanism but is vacuous for the stability decision; the directly calibrated pairwise
score and richer calibration data are required before the bound is operationally useful at this
scale."""))

# ---------------- discussion & conclusion ----------------
R.append(('discussion', r"""\section{Discussion and Limitations}
\label{sec:discussion}
First, the post-fault trajectory accuracy (\rmsePost{}$^\circ$ pooled RMSE) is limited and
concentrated near the stability boundary, where the classical model response to small parameter
changes is sharpest. Second, the conformal band is calibrated on stable scenarios and covers the
finite output grid: it is a screening band for the stable subpopulation, not a stability
certificate, and the stability of near-boundary and unstable scenarios is conveyed through the
classification, the wrong-release statistics of Section~\ref{sec:results}, and the heuristic CCT
band rather than through a numeric trajectory bound; on the 118-bus case the per-machine band is
currently vacuous for decisions (Section~\ref{sec:results}). The $\pi$-in-$4$ s criterion is the
operational label of this study; whether the first swing governs multi-swing instability in every
scenario of richer models was not established here. Third, governors, power-system stabilizers,
and dynamic loads are not modeled, and extending to higher-fidelity differential-algebraic models
and inverter-based resources is natural future work; the one-axis, exciter, and damped extensions
of Section~\ref{sec:results} retrain the same architecture and encoding rather than transferring
weights, although the encoding itself needs no modification. Hybrid event-structured designs,
which separate the fault-on and post-fault branches or trunks while retaining the
physics-informed encoding, are another avenue for better capturing the discontinuity at
clearing; we leave this to future work. Fourth, the pointwise MLP baseline is slightly worse than
the operator on trajectory RMSE at full data ($43.1^\circ$ versus \rmseAll{}$^\circ$ pooled,
Table~\ref{tab:ablation}) and the low-data advantage is indicative rather than established
(Section~\ref{sec:sampleeff}). Under the \emph{same} conformal and CCT protocol, however, the MLP
is not worse: its pairwise bound is $86.9^\circ$ (coverage $0.837$) against the operator's
\pairq{}$^\circ$, and its mean CCT error is \mlpcct{} ms against \ccterr{} ms
(Table~\ref{tab:ablation}). The operator's remaining advantages are the lower trajectory RMSE,
continuous-time evaluation through the trunk, and the low-rank structure; certification tightness
is not one of them, and we do not claim it is. Fifth, the operator is trained at a fixed
operating point and does not generalize to substantially different load levels: classification
accuracy drops to $75\%$ at $20\%$ lower load, while at $20\%$ higher load all $1080$ sampled
fault scenarios are unstable (loads were scaled uniformly with the slack bus absorbing the
imbalance; generator redispatch was not re-optimized, so this extreme operating point is not
claimed to be a feasible dispatch). Training across multiple load levels and including the load
level in the encoding does not resolve this (a model trained on four load levels attains only
$65\%$ on a held-out level). The deployment scope of the current model is therefore a screening
tool \emph{within} its calibrated operating domain; detecting and rejecting out-of-domain
operating points (e.g., via the encoding norm or an OOD detector) is left as required future
work, and operating-point generalization remains the binding obstacle to broader deployment.
Sixth, we model only bolted three-phase faults at buses; parametric fault impedance
and faults along line corridors are left to future work, although the severity encoding is
continuous in the fault admittance and can in principle represent non-bolted faults.""",
r"""\section{Discussion and Limitations}
\label{sec:discussion}
First, the post-fault trajectory accuracy (\rmsePost{}$^\circ$ pooled RMSE) is limited and
concentrated near the stability boundary, where the classical model response to small parameter
changes is sharpest. Second, the conformal band is calibrated on stable scenarios and covers the
finite output grid: it is a screening band for the stable subpopulation, not a stability
certificate; the stability of near-boundary and unstable scenarios is conveyed through the
classification, the wrong-release statistics of Section~\ref{sec:results}, and the heuristic CCT
band rather than through a numeric trajectory bound, and on the 118-bus case the per-machine band
is currently vacuous for decisions. The $\pi$-in-$4$ s criterion is the operational label of this
study; whether the first swing governs multi-swing instability in every scenario of richer models
was not established here. Third, governors, power-system stabilizers, and dynamic loads are not
modeled; the one-axis, exciter, and damped extensions of Section~\ref{sec:results} retrain the
same architecture and encoding rather than transferring weights, although the encoding itself
needs no modification. Fourth, the pointwise MLP baseline is slightly worse than the operator on
trajectory RMSE at full data ($43.1^\circ$ versus \rmseAll{}$^\circ$ pooled), and the low-data
advantage is indicative rather than established (Section~\ref{sec:sampleeff}); under the
\emph{same} conformal and CCT protocol the MLP is not worse, so the operator's remaining
advantages are the lower trajectory RMSE, continuous-time evaluation, and the low-rank structure,
and certification tightness is not one of them. Fifth, the operator is trained at a fixed
operating point and does not generalize to substantially different load levels: classification
accuracy drops to $75\%$ at $20\%$ lower load, and training across multiple load levels does not
resolve this (a model trained on four load levels attains only $65\%$ on a held-out level). The
deployment scope of the current model is therefore a screening tool \emph{within} its calibrated
operating domain; detecting and rejecting out-of-domain operating points (e.g., via the encoding
norm or an OOD detector) is left as required future work, and operating-point generalization
remains the binding obstacle to broader deployment. Sixth, we model only bolted three-phase
faults at buses; parametric fault impedance and faults along line corridors are left to future
work, although the severity encoding is continuous in the fault admittance and can in principle
represent non-bolted faults."""))

R.append(('conclusion', r"""\section{Conclusion}
We presented a physics-encoded neural operator for transient stability assessment that predicts
the full post-fault rotor-angle trajectory and the critical clearing time, with a split-conformal
layer providing a finite-sample, distribution-free trajectory band for the stable subpopulation
on the output grid. The key enabler is a physics-informed input encoding---the swing-equation
acceleration vector, extended by a post-fault severity vector---which lets the operator generalize
to unseen fault locations and to $N{-}1$ and $N{-}2$ line-trip topologies. On the IEEE 39-bus
system the operator runs $\approx\timSU{}\times$ faster than time-domain simulation on identical
hardware, classifies stability with \pctACC{} accuracy under the max-over-time $\pi$-separation
criterion, and estimates the CCT to within \ccterr{} ms against an event-aligned reference; the
directly calibrated pairwise bound (\pairq{}$^\circ$ at $\alpha{=}0.1$) is coverage-valid, with a
wrong-release rate and resolvable fraction reported in Section~\ref{sec:results} and a TDS
fallback for the remainder. A negative result shows that, for this system and protocol, the
physics-informed input encoding is more effective than a physics-residual loss. The architecture
retrained on one-axis, exciter, and damped generator models preserves the encoding and attains
comparable metrics, and retrained on a synthetic 118-bus dynamic case it reaches \pctACCb{}
classification accuracy, where trajectory-level bands remain vacuous for decisions. Together with
the operating-point limitation, these results define a fast, uncertainty-aware screening tool for
TSA within its calibrated domain, and a concrete agenda---out-of-domain detection, multi-swing
and multi-operating-point calibration, and tighter near-boundary scores---for its deployment.""",
r"""\section{Conclusion}
We presented a physics-encoded neural operator for transient stability assessment that predicts
the full post-fault rotor-angle trajectory and the critical clearing time, with a split-conformal
layer providing a finite-sample, distribution-free trajectory band for the stable subpopulation
on the output grid. The key enabler is a physics-informed input encoding---the swing-equation
acceleration vector, extended by a post-fault severity vector---which lets the operator generalize
to unseen fault locations and to $N{-}1$ and $N{-}2$ line-trip topologies. On the IEEE 39-bus
system the operator runs $\approx\timSU{}\times$ faster than time-domain simulation on identical
hardware, classifies stability with \pctACC{} accuracy under the max-over-time $\pi$-separation
criterion, and estimates the CCT to within \ccterr{} ms against an event-aligned reference; the
directly calibrated pairwise bound (\pairq{}$^\circ$ at $\alpha{=}0.1$) is coverage-valid, with a
wrong-release rate and resolvable fraction reported in Section~\ref{sec:results} and a TDS
fallback for the remainder. A negative result shows that, for this system and protocol, the
physics-informed input encoding is more effective than a physics-residual loss, and the encoding
carries over unchanged to one-axis, exciter, and damped generator models and to a synthetic
118-bus dynamic case (\pctACCb{} accuracy), where trajectory-level bands remain vacuous for
decisions. Together with the operating-point limitation, these results define a fast,
uncertainty-aware screening tool for TSA within its calibrated domain, and a concrete
agenda---out-of-domain detection, multi-swing and multi-operating-point calibration, and tighter
near-boundary scores---for its deployment."""))

# ---------------- frontmatter & declarations ----------------
R.append(('keywords', r"""\begin{keyword}
transient stability \sep neural operator \sep physics-informed machine learning \sep
conformal prediction \sep critical clearing time \sep power system dynamics
\end{keyword}""",
r"""\begin{keyword}
conformal prediction \sep critical clearing time \sep neural operator \sep physics-informed
machine learning \sep power system dynamics \sep transient stability
\end{keyword}"""))

R.append(('declarations', r"""\section*{Declarations}
\noindent\textbf{Funding.} The authors declare no specific funding for this work.\\
\textbf{Competing interests.} The authors declare no competing interests.\\""",
r"""\section*{Acknowledgments}
During the preparation of this work the authors used [TOOL NAME---e.g., Claude (Anthropic)] to
polish the language and improve the readability of the manuscript. After using this tool, the
authors reviewed and edited the content as needed and take full responsibility for the content
of the publication.

\section*{Declarations}
\noindent\textbf{Funding.} The authors declare no specific funding for this work.\\
\textbf{Conflict of interest.} The authors declare no competing interests.\\"""))

R.append(('data-avail', r"""vector sources for Figures~1--5 and Tables~1--6---is available at""",
r"""vector sources for Figures~1--3 and Tables~1--3---is available at"""))

R.append(('bios', r"""\bibliographystyle{elsarticle-num}
\bibliography{references}

\end{document}""",
r"""\bibliographystyle{elsarticle-num}
\bibliography{references}

\section*{Author biographies}
\noindent\textbf{Zhenyu Liu} received ... [PLACEHOLDER: education background and research
interests to be filled in by the authors].\\
\textbf{Jiale Wang} ... [PLACEHOLDER].\\
\textbf{Jiayong Liu} ... [PLACEHOLDER].\\
\textbf{Zhicheng Zeng} ... [PLACEHOLDER].

\end{document}"""))

# ---------------- apply ----------------
failed = []
for name, old, new in R:
    if old not in t:
        failed.append(name)
        print('FAIL:', name)
    else:
        t = t.replace(old, new, 1)
        print('ok :', name)

if failed:
    print('\nABORTED: %d replacements failed, file NOT written' % len(failed))
else:
    open(PATH, 'wb').write(t.replace('\n', '\r\n').encode('utf-8'))
    print('\nWRITTEN: all %d replacements applied' % len(R))
    import re
    words = len(re.findall(r'[A-Za-z][A-Za-z0-9\'-]*', t[:t.find(r'\bibliographystyle')]))
    print('body words (excl. refs):', words)
