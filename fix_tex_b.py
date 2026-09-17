# -*- coding: utf-8 -*-
"""Round-B text fixes: B2/B3/B5/B6 + b1-b8 + first-crossing protocol + 25.0 ms."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TEX = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
t = open(TEX, encoding='utf-8').read().replace('\r\n', '\n')
R = []

# ---- b1: mean absolute CCT error wording ----
R.append(('abstract-cct', r'''$\pi$-separation criterion, estimates the CCT to within \ccterr{} ms, and runs''',
r'''$\pi$-separation criterion, attains a mean absolute CCT error of \ccterr{} ms, and runs'''))
R.append(('conclusion-cct', r'''estimates the CCT to within \ccterr{} ms against an event-aligned reference''',
r'''attains a mean absolute CCT error of \ccterr{} ms against an event-aligned reference'''))
R.append(('r61-cct', r'''estimates the CCT to within \ccterr{} ms on average (max $204.9$ ms on bus 12), at''',
r'''attains a mean absolute CCT error of \ccterr{} ms (max $204.9$ ms on bus 12), at'''))

# ---- b3: pairwise ERROR bound wording ----
R.append(('abstract-pw', r'''score bounds the machine-angle spread to \pairq{}$^\circ$ at $\alpha{=}0.1$''',
r'''score bounds the machine-angle spread error to \pairq{}$^\circ$ at $\alpha{=}0.1$'''))
R.append(('conclusion-pw', r'''directly calibrated pairwise bound (\pairq{}$^\circ$ at $\alpha{=}0.1$) is coverage-valid''',
r'''directly calibrated pairwise-error bound (\pairq{}$^\circ$ at $\alpha{=}0.1$) is coverage-valid'''))
R.append(('m44-pw', r'''As shown in Section~\ref{sec:results}, $R_{\mathrm{pw}}$ bounds the pairwise difference to
\pairq{}$^\circ$ at $\alpha{=}0.1$''',
r'''As shown in Section~\ref{sec:results}, $R_{\mathrm{pw}}$ bounds the pairwise-difference
prediction error to \pairq{}$^\circ$ at $\alpha{=}0.1$'''))
R.append(('r66-pw', r'''bounds the pairwise angle
difference---the exact quantity in the stability criterion---to \pairq{}$^\circ$ at
$\alpha{=}0.1$ (coverage \paircov{})''',
r'''bounds the prediction error of the pairwise angle
difference---the exact quantity in the stability criterion---to \pairq{}$^\circ$ at
$\alpha{=}0.1$ (coverage \paircov{})'''))

# ---- b2: certification -> calibrated UQ ----
R.append(('rw-cert', r'''post-fault rotor-angle trajectory operator and couples it with conformal certification.''',
r'''post-fault rotor-angle trajectory operator and couples it with calibrated trajectory
uncertainty quantification.'''))

# ---- b5: raw time input ----
R.append(('trunk-raw', r'''we feed the trunk a Fourier feature embedding of
time, $\{\cos(2\pi k t/T),\sin(2\pi k t/T)\}_{k=1}^{K}$, before a small multilayer perceptron,''',
r'''we feed the trunk a Fourier feature embedding of
time, $\{\cos(2\pi k t/T),\sin(2\pi k t/T)\}_{k=1}^{K}$, alongside the normalized raw time,
before a small multilayer perceptron,'''))

# ---- B4: protocol description (4.5) ----
R.append(('prot45', r'''Given the operator, the CCT of a fault is estimated by binary search over the clearing time: for
each candidate $t_c$, the operator predicts the trajectory, and the scenario is declared unstable
if the maximum pairwise angle spread exceeds $\pi$ at \emph{any} output-grid point---the same
max-over-time criterion used for the reference labels. The search uses $16$ iterations over
$[0.01,0.8]$ s ($[0.005,0.2]$ s for the 118-bus system); the same protocol is applied to the TDS
reference, so the two CCT estimates are directly comparable. The $12\,\mu$s interval is the
\emph{search tolerance} of the bisection, not the physical accuracy of the CCT; the reference
itself uses the event-aligned RK4 scheme of Section~\ref{sec:problem}, whose convergence was
verified at a $0.25$ ms step, and both judge stability on the $0.01$ s output grid.''',
r'''Given the operator, the CCT of a fault is the first clearing time at which the criterion is
violated. Because neither the operator's criterion nor the reference is guaranteed monotone in
$t_c$, a $5$ ms sweep over $[0.01,0.8]$ s locates the first stable-to-unstable sign change, which
is then refined by local bisection to $0.1$ ms; the same first-crossing protocol (with the
system-dependent range $[0.005,0.2]$ s) is applied to the TDS reference, so the two CCT estimates
are directly comparable. The refinement tolerance is the search resolution, not the physical CCT
accuracy; the reference uses the event-aligned RK4 scheme of Section~\ref{sec:problem}
(convergence verified at a $0.25$ ms step), and both judge stability on the $0.01$ s grid.'''))

R.append(('s53-cct', r'''the CCT error against
the event-aligned TDS binary search; and the wall-clock speedup''',
r'''the CCT error against
the event-aligned TDS first-crossing search; and the wall-clock speedup'''))

# 6.1: bus-12 + protocol-insensitivity note
R.append(('r61-bus12', r'''dominated by bus 12 (reference CCT $453$ ms, estimated $248$ ms), where the binary
search lands between multiple local stable-to-unstable crossings of the operator's criterion.''',
r'''dominated by bus 12 (reference CCT $453$ ms, estimated $248$ ms), where the criterion
has multiple local stable-to-unstable crossings; the first-crossing search of
Section~\ref{sec:method} reproduces the previous binary-search CCTs within $3$ ms per bus.'''))

# ---- B2: training composition ----
R.append(('r63-open', r'''\subsection{Generalization to $N{-}1$ and $N{-}2$ topologies}
We train a single operator ($p{=}512$) on a mix of $N{-}0$ and $N{-}1$ scenarios ($9990$
scenarios, line trips drawn from $25$ training lines) using the extended encoding of
Section~\ref{sec:method}, and evaluate on two held-out axes:
unseen fault buses and unseen trip lines ($10$ lines).''',
r'''\subsection{Generalization to $N{-}1$ and $N{-}2$ topologies}
We train the multi-topology operator ($p{=}512$) with the extended encoding of
Section~\ref{sec:method}. The $N{-}1$ experiment trains on $9990$ $N{-}0$/$N{-}1$ scenarios
(line trips drawn from $25$ training lines) and evaluates on two held-out axes: unseen fault
buses and $10$ unseen trip lines. The $N{-}2$ experiment trains on $8748$ scenarios that
additionally include $N{-}2$ line-pair scenarios drawn from the training lines, and evaluates on
held-out line pairs.'''))
R.append(('r63-n2', r'''The same encoding extends to $N{-}2$ contingencies with no
architecture change: on held-out line pairs the operator attains $43.4^\circ$ RMSE and $92.9\%$
accuracy, because the post-fault severity vector is simply evaluated from the admittance matrix
with both lines removed.''',
r'''The same encoding and architecture apply to $N{-}2$ contingencies (two lines tripped)
without modification: on held-out line pairs the operator attains $43.4^\circ$ RMSE and $92.9\%$
accuracy, because the post-fault severity vector is simply evaluated from the admittance matrix
with both lines removed.'''))
R.append(('cap-n1', r'''($N{-}0$ + $N{-}1$ training): pooled''',
r'''(training mixes include $N{-}0$/$N{-}1$/$N{-}2$): pooled'''))
R.append(('bullet4', r'''We extend the encoding with a post-fault severity vector and show that a single operator
generalizes to $N{-}1$ and $N{-}2$ line-trip topologies not seen in training.''',
r'''We extend the encoding with a post-fault severity vector and show that the operator
generalizes to held-out $N{-}1$ line trips and $N{-}2$ line pairs.'''))
R.append(('conclusion-topo', r'''to unseen fault locations and to $N{-}1$ and $N{-}2$ line-trip topologies.''',
r'''to unseen fault locations and to held-out $N{-}1$/$N{-}2$ line-trip topologies.'''))

# ---- B3: per-alpha resolution counts ----
R.append(('r66-alpha', r'''this rule resolves
\decfrac{} of the predicted-stable scenarios at $\alpha{=}0.1$ (rising to $31\%$ at
$\alpha{=}0.3$ and $67\%$ at $\alpha{=}0.5$ as $\hat q_{\mathrm{pw}}$ tightens from $90.5^\circ$
to $53.3^\circ$)''',
r'''this rule resolves
\decfrac{} of the predicted-stable scenarios at $\alpha{=}0.1$ (rising to $31\%$ at
$\alpha{=}0.3$, $57\%$ at $\alpha{=}0.4$, and $67\%$ at $\alpha{=}0.5$ as $\hat q_{\mathrm{pw}}$
tightens)'''))

# ---- B5: per-unit convention ----
R.append(('b5-pu', r'''No per-machine bus mapping and no per-unit
base conversion are applied, so this is a synthetic dynamic case''',
r'''No per-machine bus mapping and no per-unit
base conversion are applied ($H$ and $x'_{d}$ are interpreted directly on the system MVA base,
a self-consistent convention); this is a synthetic dynamic case'''))

# ---- B6: MLP fairness sentence ----
R.append(('b6-mlp', r'''they show the severity encoding carries more information than raw observables. A \emph{severity-MLP}''',
r'''they show the severity encoding carries more information than raw observables. Note that the
pointwise MLP's higher classification accuracy does not carry over to trajectory quality
($43.1^\circ$ versus \rmseAll{}$^\circ$ pooled RMSE, without continuous-time evaluation;
Table~\ref{tab:ablation}). A \emph{severity-MLP}'''))

# ---- b7: Fig3 caption axis ----
R.append(('b7-cap', r'''Left: empirical coverage (with binomial
intervals) against the target level $\alpha$.''',
r'''Left: empirical coverage (with binomial
intervals) against the target coverage level $1-\alpha$.'''))

# ---- macro: 25.2 -> 25.0 ----
R.append(('ccterr', r'''\newcommand{\ccterr}{25.2}''',
r'''\newcommand{\ccterr}{25.0}'''))

# ================= compensating cuts =================
R.append(('cut-bullet2', r'''We show that a physics-informed \emph{input} encoding---the swing-equation acceleration
vector---is what enables generalization across unseen fault locations, and we quantify the
difference against a one-hot encoding and against graph baselines using raw observables.''',
r'''We show that a physics-informed \emph{input} encoding---the swing-equation acceleration
vector---enables generalization across unseen fault locations, and we quantify the difference
against one-hot and graph baselines.'''))
R.append(('cut-64', r'''These numbers use the base $p{=}128$ architecture (Table~\ref{tab:ablation}
reports $p{=}512$), which explains the different full-data values.''', ''))
R.append(('cut-69', r'''The operator is
therefore only moderately robust to \emph{feature} noise in its input encoding; correlated or
biased parameter-estimation errors and conformal coverage under input noise remain to be tested.''',
r'''The operator is therefore only moderately robust to
\emph{feature} noise; correlated or biased parameter errors and conformal coverage under noise
remain to be tested.'''))
R.append(('cut-65', r'''Adding uniform damping $D{=}1$ leaves the severity encoding unchanged---it is evaluated at the
pre-fault equilibrium, where the damping term vanishes---so the same architecture is retrained on
damped data.''',
r'''Adding uniform damping $D{=}1$ leaves the encoding unchanged (the damping term vanishes at
the pre-fault equilibrium), so the same architecture is retrained on damped data.'''))
R.append(('cut-62', r'''a stronger graph baseline with three residual layers and concatenated mean/max/sum pooling attains
$77.3\%\pm3.9\%$ (five seeds), so the gap is not an artifact of a weak graph architecture''',
r'''a stronger graph baseline with three residual layers attains
$77.3\%\pm3.9\%$ (five seeds), so the gap is not an artifact of a weak architecture'''))
R.append(('cut-66', r'''Alternative scores
(amplitude-normalized $2.3\%$, first-swing $70.8^\circ$, time-averaged $47.7^\circ$; see the
repository), adaptive/locally adaptive conformal
methods~\cite{gibbs2021adaptive, lei2014distribution}, and conformalized quantile
regression~\cite{romano2019conformalized} are candidates for reducing near-boundary conservatism
and are left as future work.''',
r'''Alternative scores (amplitude-normalized, first-swing, time-averaged; see the
repository), adaptive conformal
methods~\cite{gibbs2021adaptive, lei2014distribution}, and conformalized quantile
regression~\cite{romano2019conformalized} are candidates for reducing near-boundary conservatism.'''))

failed = []
for name, old, new in R:
    if old not in t:
        failed.append(name); print('FAIL:', name)
    else:
        t = t.replace(old, new, 1); print('ok :', name)
if failed:
    print('ABORTED, %d failed' % len(failed))
else:
    open(TEX, 'wb').write(t.replace('\n', '\r\n').encode('utf-8'))
    print('WRITTEN')
