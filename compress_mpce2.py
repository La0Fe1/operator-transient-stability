# -*- coding: utf-8 -*-
"""Round-2 compression: trim ~1.5 pages (11 -> 9-10)."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PATH = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
t = open(PATH, encoding='utf-8').read().replace('\r\n', '\n')

R = []

R.append(('rw-2.1-open', r"""Neural operators learn maps between function spaces rather than between fixed-size vectors.
The two dominant architectures are DeepONet~\cite{lu2021learning}, which represents an operator
as a contraction of a branch network (encoding the input function) with a trunk network
(encoding the output coordinates), and the Fourier Neural Operator (FNO)~\cite{li2021fourier},
which performs learned linear transforms in the Fourier domain. Physics-informed neural networks
(PINNs)~\cite{raissi2019physics} embed governing-equation residuals into the training loss to
regularize the model toward physically consistent solutions.""",
r"""Neural operators learn maps between function spaces rather than between fixed-size vectors; the
dominant architectures are DeepONet~\cite{lu2021learning} (branch--trunk contraction) and the
Fourier Neural Operator~\cite{li2021fourier}. Physics-informed neural networks
(PINNs)~\cite{raissi2019physics} embed governing-equation residuals into the training loss."""))

R.append(('rw-tef2', r"""Classical direct methods remain strong CCT baselines. The transient-energy-function (TEF/PEBS)
method~\cite{athay1979transient} and the individual-machine equal-area criterion
(IMEAC)~\cite{wang2018individual} extract the CCT from machine-level energy criteria;
Section~\ref{sec:results} compares our CCT against both. Our contribution is the combination of a
trajectory-level neural operator with a physics-informed fault encoding and a conformal error
band, which yields a CCT estimate alongside the trajectory; the derived CCT band is heuristic and
carries no formal coverage guarantee (Section~\ref{sec:method}).""",
r"""Classical direct methods remain strong CCT baselines: the transient-energy-function (TEF/PEBS)
method~\cite{athay1979transient} and the individual-machine equal-area criterion
(IMEAC)~\cite{wang2018individual} extract the CCT from machine-level energy criteria
(Section~\ref{sec:results}). Our contribution is the combination of a trajectory-level neural
operator with a physics-informed fault encoding and a conformal error band, which yields a CCT
estimate alongside the trajectory; the derived CCT band is heuristic and carries no formal
coverage guarantee (Section~\ref{sec:method})."""))

R.append(('m-4.1a', r"""where $P_{ei}^{\mathrm{f}}(k)$ is the electrical power of machine $i$ during the fault at bus
$k$, evaluated at the pre-fault angles. This vector is precisely the initial rate of change of
per-unit speed in \eqref{eq:swing2} (the angular acceleration is $\omega_s a_i$); it has
per-unit-per-second units and is divided by each machine's inertia $M_i$, which makes machines
of widely different inertia contribute comparably but is not a full non-dimensionalization. It is
the dominant driver of the fault-on transient. It is continuous in the fault admittance, so
physically similar faults map to nearby encodings. The branch input is the concatenation of
$\bm{a}(k)\in\mathbb{R}^n$ and the normalized clearing time. The encoding is not claimed to be
injective---two scenarios with similar encodings can still differ in later response near the
stability boundary---but Section~\ref{sec:results} shows it transfers the information needed for
this task, while the conformal band flags the scenarios where the prediction is least reliable.""",
r"""where $P_{ei}^{\mathrm{f}}(k)$ is the electrical power of machine $i$ during the fault at bus
$k$, evaluated at the pre-fault angles---precisely the initial rate of change of per-unit speed in
\eqref{eq:swing2} (the angular acceleration is $\omega_s a_i$). Dividing by $M_i$ makes machines
of widely different inertia contribute comparably, and continuity in the fault admittance maps
physically similar faults to nearby encodings. The branch input is the concatenation of
$\bm{a}(k)\in\mathbb{R}^n$ and the normalized clearing time. The encoding is not claimed to be
injective---two scenarios with similar encodings can still differ in later response near the
stability boundary---but Section~\ref{sec:results} shows it transfers the information needed for
this task, while the conformal band flags the scenarios where the prediction is least reliable."""))

R.append(('m-4.1b', r"""where $P_{ei}^{\mathrm{post}}(l)$ is the post-fault electrical power at the pre-fault angles with
line $l$ removed; for the intact network $\tilde a_i=0$. The full scenario encoding is the
concatenation of the fault-on severity \eqref{eq:severity}, the post-fault severity
\eqref{eq:postseverity}, and the clearing time. For $N{-}2$ contingencies the same vector is
evaluated from the admittance matrix with both lines removed: because $\tilde{\bm a}$ is always
an $n$-dimensional vector evaluated at the pre-fault angles, the encoding dimension is unchanged
for any number of tripped lines, and the same branch network applies without modification. Both
severity vectors are continuous in the fault admittance, so physically similar faults and
topologies map to nearby encodings.""",
r"""where $P_{ei}^{\mathrm{post}}(l)$ is the post-fault electrical power at the pre-fault angles with
line $l$ removed; for the intact network $\tilde a_i=0$. The full scenario encoding concatenates
the fault-on severity \eqref{eq:severity}, the post-fault severity \eqref{eq:postseverity}, and
the clearing time. For $N{-}2$ contingencies the same vector is evaluated with both lines removed:
$\tilde{\bm a}$ is always an $n$-dimensional vector evaluated at the pre-fault angles, so the
encoding dimension is unchanged and the same branch network applies without modification."""))

R.append(('m-4.2-fno', r"""(the FNO-trunk variant of Table~\ref{tab:ablation} is the closest comparison)""",
r"""(the FNO-trunk ablation of Section~\ref{sec:results} is the closest comparison)"""))

R.append(('m-4.4-band2', r"""A CCT band can be derived from the per-machine interval: clearing times whose predicted
trajectory, shifted by $\pm 2\hat q$ in spread space, cannot reach the $\pi$ threshold are
declared stable with margin, and clearing times whose shifted spread exceeds $\pi$ for \emph{all}
shifts are declared unstable; the remainder forms the band, whose endpoints are found by the same
binary search with the shifted criterion. Because the trajectory bound at one $t_c$ does not
transfer to a simultaneous guarantee across $t_c$ (the CCT is a nonlinear functional of the
trajectory), this CCT band is \emph{heuristic}: it carries no formal coverage guarantee and is
reported as a screening aid. Section~\ref{sec:results} gives its width and the fraction of
scenarios it resolves.""",
r"""A CCT band can be derived from the per-machine interval: clearing times whose predicted
trajectory, shifted by $\pm 2\hat q$ in spread space, cannot reach the $\pi$ threshold are
declared stable with margin, and clearing times whose shifted spread exceeds $\pi$ for \emph{all}
shifts are declared unstable; the remainder forms the band, found by the same binary search with
the shifted criterion. Because the trajectory bound at one $t_c$ does not transfer across $t_c$
(the CCT is a nonlinear functional of the trajectory), this CCT band is \emph{heuristic}: it
carries no formal coverage guarantee and is reported as a screening aid
(Section~\ref{sec:results})."""))

R.append(('s-5.1-trim', r"""The event-aligned labels differ from the pre-alignment reference (which kept the fault on
until the first grid point after $t_c$) on exactly one of $960$ test and one of $2160$ training
scenarios, both at the stability boundary, so the reported metrics are insensitive to this
correction. The CCT search range""",
r"""Event-aligned labels differ from the pre-alignment reference (which kept the fault on until
the first grid point after $t_c$) on exactly one of $960$ test and one of $2160$ training
scenarios, both at the stability boundary. The CCT search range"""))

R.append(('s-5.2-trim', r"""The DeepONet
variants, MLP, and GRU share the same dataset, optimizer, class-balanced mean-squared loss
(saturating beyond $\pm 3\pi$ so diverging unstable trajectories do not dominate the fit), and
$600$-epoch budget; the GCNs use binary cross-entropy (or MSE) and $300$ epochs on raw
observables, so the GCN comparison measures the value of the severity encoding rather than of the
operator architecture. A severity-MLP baseline receives exactly the same encoding as the operator
and predicts the stability label or the CCT, and the pointwise MLP and GRU are equipped with the
same conformal calibration and CCT search, so all trajectory-level models are compared under one
protocol. Parameter counts and training budgets are listed in the repository.""",
r"""The DeepONet variants, MLP, and GRU share the same dataset,
optimizer, class-balanced mean-squared loss (saturating beyond $\pm 3\pi$), and $600$-epoch
budget; the GCNs use binary cross-entropy (or MSE) and $300$ epochs on raw observables, so the
GCN comparison measures the value of the severity encoding. A severity-MLP baseline receives
exactly the same encoding as the operator, and the pointwise MLP and GRU are equipped with the
same conformal calibration and CCT search, so all trajectory-level models are compared under one
protocol; parameter counts and training budgets are in the repository."""))

R.append(('r-6.1-tef', r"""For reference, the transient-energy-function (PEBS) approach of Athay et
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
trajectories for a representative near-boundary scenario.""",
r"""For reference, the transient-energy-function (PEBS) approach of Athay et
al.~\cite{athay1979transient} with the conductance path-integral terms attains a mean CCT error
of $18.7$ ms with a conservative bias, and an IMEAC-style individual-machine
criterion~\cite{wang2018individual} evaluated on the same held-out buses yields $9.6$ ms. The
operator's \ccterr{} ms mean absolute error is larger than both and carries a conservative bias
($-21$ ms), dominated by bus 12 (reference CCT $453$ ms, estimated $248$ ms), where the binary
search lands between multiple local stable-to-unstable crossings of the operator's criterion.
Classical direct methods remain the more accurate CCT estimators; the operator's value is that it
additionally returns the full trajectory and a conformal band without any fault-on or post-fault
integration. Fig.~\ref{fig:trajectory} shows predicted versus true COI trajectories for a
representative near-boundary scenario."""))

R.append(('r-6.6-alt', r"""Alternative scores
(amplitude-normalized $2.3\%$, first-swing-window $70.8^\circ$, time-averaged $47.7^\circ$; see
the repository) and adaptive/locally adaptive conformal
methods~\cite{gibbs2021adaptive, lei2014distribution} and conformalized quantile
regression~\cite{romano2019conformalized} are candidates for reducing near-boundary conservatism
and are left as future work.""",
r"""Alternative scores
(amplitude-normalized $2.3\%$, first-swing $70.8^\circ$, time-averaged $47.7^\circ$; see the
repository), adaptive/locally adaptive conformal
methods~\cite{gibbs2021adaptive, lei2014distribution}, and conformalized quantile
regression~\cite{romano2019conformalized} are candidates for reducing near-boundary conservatism
and are left as future work."""))

R.append(('r-6.8-table', r"""and a Fourier-neural-operator trunk likewise gives no improvement
($47.4^\circ$ RMSE, $92.6\%$ accuracy; Table~\ref{tab:ablation}). This supports the choice of a
simple DeepONet with a physics-informed encoding.""",
r"""and a Fourier-neural-operator trunk likewise gives no improvement
($47.4^\circ$ RMSE, $92.6\%$ accuracy). This supports the choice of a
simple DeepONet with a physics-informed encoding."""))

R.append(('tab-ablation-rows', r"""Severity-MLP CCT regressor (same encoding) & $-$ & $-$ & \sevMLPcct{} ms \\
GCN classifier (raw features) & $-$ & $76.0\%$ & $-$ \\
GCN CCT regressor (raw features) & $-$ & $-$ & $48.0$ ms \\
Self-attention branch & $116.1^\circ$ & $92.8\%$ & $-$ \\
FNO trunk & $47.4^\circ$ & $92.6\%$ & $-$ \\
\bottomrule""",
r"""Severity-MLP CCT regressor (same encoding) & $-$ & $-$ & \sevMLPcct{} ms \\
GCN classifier (raw features) & $-$ & $76.0\%$ & $-$ \\
GCN CCT regressor (raw features) & $-$ & $-$ & $48.0$ ms \\
\bottomrule"""))

R.append(('cap-arch', r"""\caption{Overview of the proposed physics-informed neural operator ($N{-}0$ configuration shown;
$N{-}1$/$N{-}2$ configurations additionally use the post-fault severity vector). A fault scenario
is mapped through the physics-informed encoding (the swing-equation acceleration vector) to a
branch network, whose coefficients contract with a Fourier-feature trunk network to yield the
predicted rotor-angle trajectory $\hat{\bm\delta}$; split conformal prediction then attaches a
distribution-free error envelope.}""",
r"""\caption{Overview of the proposed physics-informed neural operator ($N{-}0$ configuration shown;
$N{-}1$/$N{-}2$ additionally use the post-fault severity vector): a fault scenario is mapped
through the physics-informed encoding to a branch network, whose coefficients contract with a
Fourier-feature trunk to yield the predicted rotor-angle trajectory $\hat{\bm\delta}$; split
conformal prediction then attaches a distribution-free error envelope.}"""))

R.append(('cap-traj', r"""\caption{Predicted (dashed) versus true (solid) COI rotor-angle trajectories for the two most
affected machines in a representative near-boundary stable scenario (fault bus \trajbus{},
$t_c=\trajtc{}$ s). The fault-on segment ($t<t_c$, marked) is reproduced accurately, while the
post-fault oscillation carries the bulk of the error; the shaded band shows the first-swing
conformal band ($\trajband{}$ at $\alpha{=}0.1$). The first-swing window is defined by the
\emph{true} trajectory (an offline diagnostic); the per-machine band does not by itself certify
the pairwise stability criterion, which the directly calibrated score of
Section~\ref{sec:method} targets.}""",
r"""\caption{Predicted (dashed) versus true (solid) COI rotor-angle trajectories for the two most
affected machines in a representative near-boundary stable scenario (fault bus \trajbus{},
$t_c=\trajtc{}$ s); the shaded band is the first-swing conformal band ($\trajband{}$ at
$\alpha{=}0.1$). The fault-on segment ($t<t_c$, marked) is reproduced accurately, while the
post-fault oscillation carries the bulk of the error. The first-swing window is defined by the
\emph{true} trajectory (an offline diagnostic); the per-machine band does not by itself certify
the pairwise stability criterion.}"""))

R.append(('cap-conf', r"""\caption{Conformal band validity and adaptivity. Left: empirical coverage (with binomial
intervals) against the target level $\alpha$; markers show the four reported levels and the
diagonal is the nominal target. Right: offline Mondrian diagnostic---the grouping uses the
ground-truth post-fault peak of each scenario---shows the band half-width growing toward the
stability boundary; bars are annotated with group size (calibration/evaluation) and per-group
coverage.}""",
r"""\caption{Conformal band validity and adaptivity. Left: empirical coverage (with binomial
intervals) against the target level $\alpha$. Right: offline Mondrian diagnostic---the grouping
uses the ground-truth post-fault peak of each scenario---shows the band half-width growing toward
the stability boundary; bars are annotated with group size (calibration/evaluation) and per-group
coverage.}"""))

R.append(('r-6.5-damped', r"""Adding uniform damping $D{=}1$ leaves the severity encoding unchanged---it is evaluated at the
pre-fault equilibrium, where the damping term vanishes---so the same architecture is retrained on
damped data. The fault-on region improves to $1.8^\circ$ pooled RMSE, but the post-fault RMSE
rises to $46.4^\circ$ (versus \rmsePost{}$^\circ$ undamped) and classification drops to $93.4\%$
(CCT error $30.8$ ms): near-boundary damped trajectories decay slowly and linger at high angle
spreads, so small errors in the predicted decay rate accumulate over the $4$ s window and produce
a heavier error tail (maximum worst-case error $638^\circ$). The split-conformal band remains
approximately valid (mean coverage $0.90\pm0.02$ over five calibration splits). Damping therefore
does not simplify the near-boundary prediction task.""",
r"""Adding uniform damping $D{=}1$ leaves the severity encoding unchanged---it is evaluated at the
pre-fault equilibrium, where the damping term vanishes---so the same architecture is retrained on
damped data. The fault-on region improves to $1.8^\circ$ pooled RMSE, but the post-fault RMSE
rises to $46.4^\circ$ (versus \rmsePost{}$^\circ$ undamped) and classification drops to $93.4\%$
(CCT error $30.8$ ms): near-boundary damped trajectories decay slowly, so small errors in the
predicted decay rate accumulate over the $4$ s window (maximum worst-case error $638^\circ$). The
split-conformal band remains approximately valid (mean coverage $0.90\pm0.02$ over five
calibration splits); damping does not simplify the near-boundary prediction task."""))

R.append(('discussion2', r"""\section{Discussion and Limitations}
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
represent non-bolted faults.""",
r"""\section{Discussion and Limitations}
\label{sec:discussion}
First, the post-fault trajectory accuracy (\rmsePost{}$^\circ$ pooled RMSE) is limited and
concentrated near the stability boundary, where the classical model response to small parameter
changes is sharpest. Second, the conformal band is calibrated on stable scenarios and covers the
finite output grid: it is a screening band for the stable subpopulation, not a stability
certificate; near-boundary and unstable scenarios are conveyed through the classification, the
wrong-release statistics of Section~\ref{sec:results}, and the heuristic CCT band rather than a
numeric trajectory bound, and on the 118-bus case the per-machine band is currently vacuous for
decisions. The $\pi$-in-$4$ s criterion is the operational label of this study; whether the first
swing governs multi-swing instability in every scenario of richer models was not established
here. Third, governors, power-system stabilizers, and dynamic loads are not modeled; the one-axis,
exciter, and damped extensions retrain the same architecture and encoding rather than
transferring weights, although the encoding itself needs no modification. Fourth, the pointwise
MLP baseline is slightly worse than the operator on trajectory RMSE at full data
($43.1^\circ$ versus \rmseAll{}$^\circ$ pooled), and the low-data advantage is indicative rather
than established (Section~\ref{sec:sampleeff}); under the \emph{same} conformal and CCT protocol
the MLP is not worse, so the operator's remaining advantages are the lower trajectory RMSE,
continuous-time evaluation, and the low-rank structure, and certification tightness is not one of
them. Fifth, the operator is trained at a fixed operating point and does not generalize to
substantially different load levels: classification accuracy drops to $75\%$ at $20\%$ lower
load, and training across multiple load levels does not resolve this (a model trained on four
load levels attains only $65\%$ on a held-out level). Deployment is therefore a screening tool
\emph{within} its calibrated operating domain; detecting and rejecting out-of-domain operating
points (e.g., via the encoding norm or an OOD detector) is left as required future work, and
operating-point generalization remains the binding obstacle to broader deployment. Sixth, we
model only bolted three-phase faults at buses; parametric fault impedance and faults along line
corridors are left to future work, although the severity encoding is continuous in the fault
admittance and can in principle represent non-bolted faults."""))

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
