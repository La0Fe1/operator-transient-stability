# -*- coding: utf-8 -*-
"""Produce paper_rev2.tex: R1-R11 revisions applied to a COPY of paper_mpce.tex,
each revision tagged with a literal ASCII marker [Rev. R#] that compiles into the PDF."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
SRC = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
DST = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_rev2.tex'
t = open(SRC, encoding='utf-8').read().replace('\r\n', '\n')

R = []

# ---- R1 (Q1): restore alternative-score numbers with trade-off ----
R.append(('R1', r'''Alternative scores (amplitude-normalized, first-swing, time-averaged; see the
repository), adaptive conformal
methods~\cite{gibbs2021adaptive, lei2014distribution}, and conformalized quantile
regression~\cite{romano2019conformalized} are candidates for reducing near-boundary conservatism.''',
r'''[Rev. R1] Alternative scores trade coverage semantics for tighter bands: an
amplitude-normalized (relative) score bounds the error to $2.3\%$ of each scenario's peak
amplitude (coverage $0.854$), a first-swing-window score bounds the first swing to
$70.8^\circ$ (coverage $0.865$; an offline diagnostic, since the window is defined by the true
trajectory), and a time-averaged score yields $47.7^\circ$ (coverage $0.893$) but bounds the
time average rather than the uniform-in-time error; none of these bounds the pairwise spread
directly, so they complement rather than replace $R_{\mathrm{pw}}$. Perturbation-scaled and
input-adaptive scores are candidates for tightening near-boundary bands and are left as future
work, together with adaptive conformal
methods~\cite{gibbs2021adaptive, lei2014distribution} and conformalized quantile
regression~\cite{romano2019conformalized}.'''))

# ---- R2 (Q2): grid-resolution check with real data ----
R.append(('R2', r'''it applies to the finite output grid (a denser-grid check found no missed
criterion crossings, Section~\ref{sec:results});''',
r'''it applies to the finite output grid ([Rev. R2] a $2$ ms dense-grid check on $200$ held-out
scenarios found no missed criterion crossings, i.e., no label flips relative to the $10$ ms
output grid);'''))

# ---- R3 (Q3+Q8): future work: physics-aware GNN baselines + TEF/IMEAC hybrid ----
R.append(('R3', r'''Sixth, only bolted three-phase faults are
modeled; parametric fault impedance and line-corridor faults are left to future work.''',
r'''Sixth, only bolted three-phase
bus faults are modeled; the encoding is continuous in the fault admittance, so finite-impedance
faults map to nearby encodings, and a fault along a line corridor can in principle be represented
by the admittance of the faulted, split line; neither case has been tested. [Rev. R3] Further
follow-ups include physically informed graph baselines (GNNs fed with $Y$-bus or PTDF/LDF
features) and hybrid schemes that use TEF/IMEAC to refine the operator's CCT near the boundary.'''))

# ---- R4 (Q4): practical calibration maintenance ----
R.append(('R4', r'''Per-machine calibration with Bonferroni/\v{S}id\'ak corrections does not tighten the
bound, and calibrating''',
r'''[Rev. R4] For deployment, rolling calibration windows refreshed on recent contingencies, and
grouping rules that depend only on observable inputs rather than ground-truth labels, are the
practical counterparts of the offline Mondrian diagnostic; these are left as future work.
Per-machine calibration with Bonferroni/\v{S}id\'ak corrections does not tighten the
bound, and calibrating'''))

# ---- R5 (Q5): 118 pairwise score (new experiment, real data) ----
R.append(('R5', r'''so on the 118-bus case the interval validates the coverage mechanism but is vacuous
for decisions; the pairwise score and richer calibration data
are required before the bound is operationally useful at this scale.''',
r'''so on the 118-bus case the interval validates the coverage mechanism but is vacuous
for decisions. [Rev. R5] The directly calibrated pairwise
score does not recover a nontrivial resolvable fraction there either: at $\alpha{=}0.1$ its
half-width is $1057.6^\circ$ (coverage $0.918$), and even at $\alpha{=}0.5$ ($91.7^\circ$,
coverage $0.582$) the rule $\mathrm{spread}+\hat q_{\mathrm{pw}}<\pi$ resolves $0$ of the $88$
predicted-stable evaluation scenarios, because the predicted spreads themselves sit near the
$\pi$ threshold; richer calibration data and system-specific scores remain required.'''))

# ---- R6 (Q6): CCT band vacuity statistics + monotone-regression future work ----
R.append(('R6', r'''In the gated regime (evaluated on the disjoint half): gate purity''',
r'''[Rev. R6] The derived CCT band is vacuous in the reported configuration: with
$2\hat q = 296.2^\circ$ at $\alpha{=}0.1$, the conservative endpoints land at the search lower
bound ($10$ ms) on all $12$ held-out buses, so no scenario is resolved; conformalized monotone
regression over $t_c$ is a candidate for stabilizing the band and is left as future work.
In the gated regime (evaluated on the disjoint half): gate purity'''))

# ---- R9 (Q9): restore K/p hyperparameter ablations ----
R.append(('R9', r'''All main-table numbers are
exported by a single evaluation script from one model and one dataset; four-seed statistics are
reported separately where available.''',
r'''[Rev. R9] The results are not artifacts of the hyperparameters: halving the Fourier harmonics
to $K{=}8$ degrades the pooled RMSE to $45.9^\circ$ ($94.3\%$ accuracy), and reducing the loss
saturation to $\pm2\pi$ yields $94.7\%$ accuracy with a slightly better pooled RMSE of
$34.3^\circ$; we retain $\pm3\pi$ because it is selected on validation data and gives more
stable fits to diverging unstable trajectories. The basis dimension $p$ is ablated in
Table~\ref{tab:ablation} and Section~\ref{sec:sampleeff}. All main-table numbers are
exported by a single evaluation script from one model and one dataset; four-seed statistics are
reported separately where available.'''))

# ---- R10 (Q10): concrete OOD mechanisms ----
R.append(('R10', r'''detecting and rejecting out-of-domain operating
points (e.g., via the encoding norm or an OOD detector) is left as required future work''',
r'''detecting and rejecting out-of-domain operating
points ([Rev. R10] e.g., thresholds on the encoding norm, statistics of the predicted spread,
or conformal-residual monitors) is left as required future work'''))

# ---- R11 (Q11): encoding/inference scaling with real measurements ----
R.append(('R11', r'''On the 118-bus system
the encoding ($\approx4.1$ ms per fault) dominates a single scenario.''',
r'''On the 118-bus system
the encoding ($\approx4.1$ ms per fault) dominates a single scenario. [Rev. R11] The encoding is
a Kron reduction whose measured cost is $0.13$ ms per fault on the 39-bus system and $4.1$ ms on
the 118-bus system---still a small fraction of one TDS run, but it grows with network size---and
the operator forward pass is linear in the number of output grid points and in the number of
machines (the branch input dimension is $n+1$).'''))

failed = []
for name, old, new in R:
    if old not in t:
        failed.append(name); print('FAIL:', name)
    else:
        t = t.replace(old, new, 1); print('ok :', name)

if failed:
    print('ABORTED, %d failed' % len(failed))
else:
    open(DST, 'wb').write(t.replace('\n', '\r\n').encode('utf-8'))
    print('WRITTEN paper_rev2.tex')
