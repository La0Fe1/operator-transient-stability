# -*- coding: utf-8 -*-
"""Round-3 compression: ~60 more lines to land at 9-10 pages."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PATH = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
t = open(PATH, encoding='utf-8').read().replace('\r\n', '\n')

R = []

R.append(('intro-2', r"""This cost has motivated a large body of machine-learning surrogates for TSA. Most learn a direct
map from operating-condition or fault features to a binary stable/unstable label using classifiers
such as support vector machines, decision trees, or deep networks~\cite{alimi2020review}. Fast as
they are, these surrogates have three limitations that matter for deployment. First, they discard
the rotor-angle trajectory and thus cannot provide a stability \emph{margin} or a CCT estimate.
Second, they do not generalize naturally across fault locations: representing a fault by a one-hot
bus index yields no signal for unseen buses. Third, and most importantly for a safety-critical
application, they output no uncertainty or error bound, so an operator cannot know when the
surrogate is trustworthy.""",
r"""This cost has motivated a large body of machine-learning surrogates for TSA: classifiers such as
support vector machines, decision trees, or deep networks~\cite{alimi2020review} that map
operating-condition or fault features to a binary stable/unstable label. Fast as they are, these
surrogates discard the rotor-angle trajectory (and thus any stability \emph{margin} or CCT
estimate), do not generalize naturally across fault locations (a one-hot bus index yields no
signal for unseen buses), and output no uncertainty or error bound, so an operator cannot know
when the surrogate is trustworthy."""))

R.append(('intro-3', r"""We address all three limitations with a physics-encoded neural operator equipped with conformal
trajectory bands. The operator is a Deep Operator Network (DeepONet)~\cite{lu2021learning}, whose
branch network encodes a fault scenario and whose Fourier-feature trunk network encodes time;
their contraction yields the full center-of-inertia (COI) rotor-angle trajectory rather than
merely a label. The fault scenario is encoded by a \emph{physics-informed} vector: the initial
acceleration $(P_{mi}-P_{ei}^{\mathrm{f}})/M_i$ that the fault induces on each machine, computed
directly from the swing equation. This continuous, physically meaningful encoding is what allows
the operator to generalize across unseen fault locations. Finally, we use split conformal
prediction~\cite{lei2018distribution} to attach a finite-sample, distribution-free error band to
every predicted trajectory, together with a tighter score that directly bounds the pairwise angle
difference entering the stability criterion. Throughout, ``physics-informed'' refers to the input
encoding derived from the swing equation, not to a residual-constrained model, and the conformal
output is a calibrated trajectory band, not a stability certificate.""",
r"""We address all three limitations with a physics-encoded neural operator equipped with conformal
trajectory bands. The operator is a Deep Operator Network (DeepONet)~\cite{lu2021learning}, whose
branch encodes a fault scenario and whose Fourier-feature trunk encodes time; their contraction
yields the full center-of-inertia (COI) rotor-angle trajectory rather than merely a label. The
fault scenario is encoded by a \emph{physics-informed} vector: the initial acceleration
$(P_{mi}-P_{ei}^{\mathrm{f}})/M_i$ that the fault induces on each machine, computed directly from
the swing equation, which allows the operator to generalize across unseen fault locations. Split
conformal prediction~\cite{lei2018distribution} attaches a finite-sample, distribution-free error
band to every predicted trajectory, together with a tighter score that directly bounds the
pairwise angle difference entering the stability criterion. Throughout, ``physics-informed''
refers to the input encoding derived from the swing equation, not to a residual-constrained model,
and the conformal output is a calibrated trajectory band, not a stability certificate."""))

R.append(('rw-es', r"""A few works predict the post-fault trajectory rather than only the label; these are closest in
spirit to ours, but they provide no error bounds. Event-structured PINNs model the pre-fault,
fault-on, and post-fault stages explicitly with hard state chaining and a differentiable CCT
boundary~\cite{hao2026event}; their error-propagation theorem is complementary to our
finite-sample conformal band. PINN-based acceleration of time-domain simulation
\cite{stiasny2021transient} trains a network with physics residuals; our operator instead learns
the scenario-to-trajectory map with a physics-informed input encoding, avoiding residual
minimization altogether. Fast CCT calculation for systems mixing synchronous and asynchronous
generation~\cite{wang2025fast} is complementary to the TDS-based reference we validate against.""",
r"""A few works predict the post-fault trajectory rather than only the label; these are closest in
spirit to ours, but they provide no error bounds. Event-structured PINNs model the pre-fault,
fault-on, and post-fault stages with hard state chaining and a differentiable CCT
boundary~\cite{hao2026event}; their error-propagation theorem is complementary to our
finite-sample conformal band. PINN-based acceleration of time-domain simulation
\cite{stiasny2021transient} trains a network with physics residuals, whereas we learn the
scenario-to-trajectory map with a physics-informed input encoding; fast CCT calculation for
mixed synchronous/asynchronous generation~\cite{wang2025fast} is complementary to our TDS-based
reference."""))

R.append(('pf-3.2-eq5', r"""We seek a map
\begin{equation}
\mathcal{G}: (\text{scenario}) \;\longmapsto\; \{\tilde\delta_i(t)\}_{i=1}^{n},\; t\in[0,T],
\end{equation}
returning the COI-referenced rotor-angle trajectory, together""",
r"""We seek the map
$\mathcal{G}\colon(\text{scenario})\mapsto\{\tilde\delta_i(t)\}_{i=1}^{n}$, $t\in[0,T]$,
returning the COI-referenced rotor-angle trajectory, together"""))

R.append(('m-4.4-open', r"""The trajectory operator alone gives no indication of its own error. We use split conformal
prediction~\cite{lei2018distribution} to attach a finite-sample, distribution-free band to the
single-machine COI angles. We calibrate on \emph{stable} scenarios: unstable trajectories diverge
and carry errors orders of magnitude beyond any useful bound, so the band is defined on the stable
subpopulation by design---a scope choice, not a claim that unstable dynamics are mathematically
ill-posed. For a scenario $s$, the nonconformity score is the worst-case prediction error over
machines and output-grid time points,""",
r"""The trajectory operator alone gives no indication of its own error. We use split conformal
prediction~\cite{lei2018distribution} to attach a finite-sample, distribution-free band to the
single-machine COI angles, calibrated on \emph{stable} scenarios: unstable trajectories diverge
and carry errors orders of magnitude beyond any useful bound, so the band is defined on the stable
subpopulation by design---a scope choice, not a claim that unstable dynamics are mathematically
ill-posed. For a scenario $s$, the nonconformity score is the worst-case prediction error over
machines and output-grid time points,"""))

R.append(('m-4.4-split', r"""We split the $N$ stable scenarios of the held-out test set (Section~\ref{sec:setup})---which the
operator never saw during training and which are divided at random from the same held-out-bus
sampling distribution, so that calibration and evaluation scores are exchangeable---into a
calibration half ($N_{\mathrm{cal}}$) and a disjoint evaluation half ($N_{\mathrm{eva}}$). For a
target miscoverage level $\alpha$, the bound $\hat q$ is the $k$-th order statistic of the
calibration scores with $k=\lceil(N_{\mathrm{cal}}+1)(1-\alpha)\rceil$, the standard finite-sample
split-conformal rule. For a new scenario exchangeable with the calibration scenarios and drawn
from the same distribution \emph{conditionally on being truly stable}, the whole predicted
trajectory is then covered simultaneously on the output grid:""",
r"""We split the $N$ stable scenarios of the held-out test set (Section~\ref{sec:setup})---which the
operator never saw during training and which are drawn at random from the same held-out-bus
sampling distribution, so calibration and evaluation scores are exchangeable---into a calibration
half ($N_{\mathrm{cal}}$) and a disjoint evaluation half ($N_{\mathrm{eva}}$). For target
miscoverage $\alpha$, the bound $\hat q$ is the $k$-th order statistic of the calibration scores
with $k=\lceil(N_{\mathrm{cal}}+1)(1-\alpha)\rceil$, the standard finite-sample split-conformal
rule. For a new scenario exchangeable with the calibration scenarios and drawn from the same
distribution \emph{conditionally on being truly stable}, the whole predicted trajectory is then
covered simultaneously on the output grid:"""))

R.append(('m-4.5-tail', r"""verified at a $0.25$ ms step, and both judge stability on the $0.01$ s output grid. The same
search with the shifted criterion of the previous subsection yields the heuristic CCT band.""",
r"""verified at a $0.25$ ms step, and both judge stability on the $0.01$ s output grid."""))

R.append(('s-5.2-enum', r"""We compare against (i) the same DeepONet with a one-hot fault encoding, isolating the effect of
the physics-informed encoding; (ii) a pointwise multilayer perceptron baseline mapping (scenario,
time) to the angle directly; (iii) the operator trained with the auxiliary physics-residual loss
at several weights; (iv) graph convolutional networks (GCNs) that predict the stability label or
the CCT from raw node features (voltage magnitudes and angles, loads, and a fault flag); (v) a
self-attention branch; (vi) a Fourier-neural-operator trunk; and (vii) a recurrent (GRU)
sequence-to-sequence baseline.""",
r"""We compare against the same DeepONet with (i) a one-hot fault encoding; (ii) a physics-residual
loss at several weights; (iii) a self-attention branch; and (iv) a Fourier-neural-operator trunk,
plus a pointwise multilayer perceptron (MLP) baseline mapping (scenario, time) to the angle
directly, graph convolutional networks (GCNs) that predict the stability label or the CCT from
raw node features (voltage magnitudes and angles, loads, and a fault flag), and a recurrent (GRU)
sequence-to-sequence baseline."""))

R.append(('s-5.3-speed-tail', r"""On the 118-bus system
the encoding ($\approx4.1$ ms per fault) dominates a single scenario. The RK4 cost scales
inversely with the reference step size, whereas the operator's cost depends only on the number of
output time points ($401$ per trajectory), so the speedup grows for finer reference steps.
Hardware models, software versions, and timing repetitions are listed in the repository
(\url{https://github.com/La0Fe1/operator-transient-stability}).""",
r"""On the 118-bus system
the encoding ($\approx4.1$ ms per fault) dominates a single scenario. The RK4 cost scales
inversely with the reference step size, whereas the operator's cost depends only on the number of
output time points ($401$ per trajectory), so the speedup grows for finer reference steps; all
timing details are in the repository (\url{https://github.com/La0Fe1/operator-transient-stability})."""))

R.append(('r-6.1-open', r"""\subsection{Trajectory prediction and classification}
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
$204.9$ ms on bus 12), at a $\approx\timSU{}\times$ throughput speedup over the RK4 reference.""",
r"""\subsection{Trajectory prediction and classification}
Table~\ref{tab:main} summarizes the main results. The physics-informed operator predicts the
fault-on segment of the trajectory to \rmseFault{}$^\circ$ pooled RMSE and the post-fault segment
to \rmsePost{}$^\circ$ (full-horizon \rmseAll{}$^\circ$); the post-fault error is concentrated
near the stability boundary---the sensitivity the conformal band is designed to capture---with
$10.5^\circ$ in the first-swing window and $41.8^\circ$ in the subsequent oscillation. The
operator classifies stability under the max-over-time $\pi$-separation criterion with \pctACC{}
accuracy and confusion matrix (TP, FP, FN, TN) $=$\confmat{} ($F_1$ for the stable class
\fone{}), and estimates the CCT to within \ccterr{} ms on average (max $204.9$ ms on bus 12), at
a $\approx\timSU{}\times$ throughput speedup over the RK4 reference."""))

R.append(('r-6.2-gcn', r"""A GCN classifier operating on raw node features (voltage magnitudes and angles, loads, and a
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
\rmseAll{}$^\circ$: the low-rank operator structure outperforms a generic recurrent decoder.""",
r"""A GCN classifier operating on raw node features (voltage magnitudes and angles, loads, and a
fault flag) achieves only $76.0\%$ accuracy versus \pctACC{} for the severity-encoded operator;
a stronger graph baseline with three residual layers and concatenated mean/max/sum pooling attains
$77.3\%\pm3.9\%$ (five seeds), so the gap is not an artifact of a weak graph architecture, and a
GCN regressing the CCT from the same raw features attains a mean CCT error of $48.0$ ms (max
$224.4$ ms) versus \ccterr{} ms (max $204.9$ ms). These comparisons are not
architecture-controlled (the GCNs output a label or a scalar, the operator a trajectory), but
they show the severity encoding carries more information than raw observables. A \emph{severity-MLP}
receiving exactly the same encoding attains \sevMLPacc{} accuracy and \sevMLPcct{} ms mean CCT
error (Table~\ref{tab:ablation}): the encoding, not the DeepONet architecture, is the dominant
factor for generalization to unseen fault locations. A recurrent (GRU) sequence-to-sequence
baseline mapping the same encoding through a sequential decoder attains $57.3^\circ$ RMSE and
$93.9\%$ accuracy, worse than the operator's \rmseAll{}$^\circ$: the low-rank operator structure
outperforms a generic recurrent decoder."""))

R.append(('r-6.4-trim', r"""A motivation for the operator architecture is its low-rank inductive bias. We train the DeepONet
and the pointwise MLP baseline on $10\%$, $25\%$, $50\%$, and $100\%$ of the training scenarios
(three seeds each) and evaluate on the full held-out test set. At $10\%$ of the data the operator
attains $56.1^\circ$ RMSE versus $65.2^\circ$ for the MLP (a $\approx9^\circ$ advantage); at
$25\%$ the gap is $\approx8^\circ$ ($46.4^\circ$ versus $54.8^\circ$); at $50\%$ and $100\%$ it
narrows to $\approx2$--$3^\circ$. The low-rank structure therefore confers a modest
sample-efficiency advantage in the low-data regime, where fast TSA surrogates are most valuable;
with three seeds per fraction and a $14.0^\circ$ across-seed spread at $10\%$ data, the low-data
advantage is indicative rather than statistically established. These numbers use the base
$p{=}128$ architecture averaged over three seeds, whereas Table~\ref{tab:ablation} reports
$p{=}512$, which explains the different full-data values.""",
r"""A motivation for the operator architecture is its low-rank inductive bias. Training the DeepONet
and the pointwise MLP baseline on $10\%$, $25\%$, $50\%$, and $100\%$ of the training scenarios
(three seeds each), the operator attains $56.1^\circ$ RMSE versus $65.2^\circ$ for the MLP at
$10\%$ of the data (a $\approx9^\circ$ advantage), $\approx8^\circ$ at $25\%$, and
$\approx2$--$3^\circ$ at $50\%$/$100\%$. The low-rank structure therefore confers a modest
sample-efficiency advantage in the low-data regime; with three seeds per fraction and a
$14.0^\circ$ across-seed spread at $10\%$ data, the advantage is indicative rather than
statistically established. These numbers use the base $p{=}128$ architecture (Table~\ref{tab:ablation}
reports $p{=}512$), which explains the different full-data values."""))

R.append(('r-6.6-open', r"""The split-conformal band achieves empirical coverage close to nominal at every level on the
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
screening envelope, not a tight predictive interval.""",
r"""The split-conformal band achieves empirical coverage close to nominal at every level on the
$N_{\mathrm{eva}}{=}178$ evaluation scenarios: $91.6\%$ ($163/178$) at the $95\%$ target,
$88.2\%$ ($157/178$, CI $83$--$93\%$) at $90\%$, $67.4\%$ ($120/178$) at $70\%$, and $43.8\%$
($78/178$) at $50\%$ (Fig.~\ref{fig:conformal}); the same calibration on the multi-topology
operators yields coverage of $0.915$ on both held-out $N{-}1$ splits and $0.872$/$0.852$ on
$N{-}2$ buses and line pairs (half-widths $121$--$181^\circ$). A Mondrian (grouped)
calibration~\cite{vovk2005algorithmic} by the post-fault peak of the pairwise
spread---$\max_{t\ge t_c}\bigl(\max_i \tilde\delta_i(t)-\min_i \tilde\delta_i(t)\bigr)$ of the
\emph{true} trajectory, an offline diagnostic because the grouping uses ground-truth labels---shows,
at $\alpha{=}0.1$, that the half-width grows toward the stability boundary, from
\mondA{}$^\circ$ for scenarios peaking below $45^\circ$ (group $48/51$ calibration/evaluation,
coverage $0.863$), through $94.2^\circ$ for $45$--$90^\circ$ ($44/52$, $0.923$), to
\mondC{}$^\circ$ above $90^\circ$ ($86/75$, $0.800$): coverage is approximately nominal away from
the boundary and below nominal near it. The band is wide precisely where the stability decision
is most critical; it is a conservative screening envelope, not a tight predictive interval."""))

R.append(('r-6.6-valset', r"""fraction. Per-machine calibration with Bonferroni/\v{S}id\'ak corrections does not tighten the
bound (pairwise $734^\circ$), and calibrating on the validation set instead of held-out scenarios
yields only $0.61$ coverage at the nominal $0.90$ target: the held-out-bus distribution is
genuinely harder, so calibrating on held-out scenarios is necessary rather than a leakage
concern, provided future scenarios are exchangeable with the calibration set. The calibration
does not transfer across operating points""",
r"""fraction. Per-machine calibration with Bonferroni/\v{S}id\'ak corrections does not tighten the
bound (pairwise $734^\circ$), and calibrating on the validation set instead of held-out scenarios
yields only $0.61$ coverage at the nominal $0.90$ target: the held-out-bus distribution is
genuinely harder, so calibrating on held-out scenarios is necessary rather than a leakage
concern. The calibration does not transfer across operating points"""))

R.append(('r-6.7-tail', r"""This
conclusion is scoped to the present system, protocol, weights, and architecture; here, physics is
injected more effectively through the input encoding derived from the swing equation than through
a residual loss.""",
r"""This conclusion is scoped to the present system, protocol, weights, and architecture."""))

R.append(('r-6.10-trim', r"""The higher trajectory RMSE reflects the larger state space and the much lower CCTs of
the assigned 118-bus case ($0.008$--$0.135$ s), and the CCT error on the $24$ held-out buses is
$13.3$ ms mean absolute (max $54.7$ ms). The split-conformal interval attains $0.918$ coverage at
$\alpha{=}0.1$, but with a half-width of $596.4^\circ$---far wider than the 39-bus band and larger
than the $180^\circ$ stability threshold, so on the 118-bus case the interval validates the
coverage mechanism but is vacuous for the stability decision; the directly calibrated pairwise
score and richer calibration data are required before the bound is operationally useful at this
scale.""",
r"""The higher trajectory RMSE reflects the larger state space and the much lower CCTs of
the assigned 118-bus case ($0.008$--$0.135$ s), with a CCT error of $13.3$ ms mean absolute (max
$54.7$ ms) on the $24$ held-out buses. The split-conformal interval attains $0.918$ coverage at
$\alpha{=}0.1$ with a half-width of $596.4^\circ$---larger than the $180^\circ$ stability
threshold, so on the 118-bus case the interval validates the coverage mechanism but is vacuous
for the stability decision; the directly calibrated pairwise score and richer calibration data
are required before the bound is operationally useful at this scale."""))

R.append(('data-avail-trim', r"""\textbf{Data availability.} The code reproducing all experiments---including scenario generation
(event-aligned RK4 reference simulator and Kron reduction for the IEEE 39- and 118-bus systems),
model training for all variants, conformal calibration, CCT binary search, and the scripts and
vector sources for Figures~1--3 and Tables~1--3---is available at
\url{https://github.com/La0Fe1/operator-transient-stability}, together with the random seeds,
software versions, and environment file.\\""",
r"""\textbf{Data availability.} The code reproducing all experiments---scenario generation,
model training for all variants, conformal calibration, CCT binary search, and the sources for
Figures~1--3 and Tables~1--3---is available at
\url{https://github.com/La0Fe1/operator-transient-stability}, together with the random seeds,
software versions, and environment file.\\"""))

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
