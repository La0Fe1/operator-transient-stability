# -*- coding: utf-8 -*-
"""Round-4 compression: ~20 more lines to land at 10 pages."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PATH = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
t = open(PATH, encoding='utf-8').read().replace('\r\n', '\n')

R = []

R.append(('eq11-inline', r"""covered simultaneously on the output grid:
\begin{equation}
\Pr\Bigl\{ \max_{i,t\in\mathcal{T}_{\mathrm{grid}}} \bigl| \hat{\tilde\delta}_i(t) - \tilde\delta_i(t) \bigr| \le \hat q \;\Big|\; \text{scenario stable} \Bigr\} \ge 1-\alpha,
\end{equation}
i.e., with probability at least $1-\alpha$ over exchangeable stable scenarios, every single-machine
COI angle at every \emph{output-grid} point lies within $\hat q$ of the truth. Three limits are""",
r"""covered simultaneously on the output grid: $\Pr\{ \max_{i,t\in\mathcal{T}_{\mathrm{grid}}}
\bigl| \hat{\tilde\delta}_i(t) - \tilde\delta_i(t) \bigr| \le \hat q \mid \text{scenario stable} \}
\ge 1-\alpha$, i.e., with probability at least $1-\alpha$ over exchangeable stable scenarios,
every single-machine COI angle at every \emph{output-grid} point lies within $\hat q$ of the
truth. Three limits are"""))

R.append(('m-4.2-trim', r"""so the trunk represents the natural oscillation modes more efficiently than a plain perceptron
(the FNO-trunk ablation of Section~\ref{sec:results} is the closest comparison). The trunk is smooth""",
r"""so the trunk represents the natural oscillation modes more efficiently than a plain perceptron.
The trunk is smooth"""))

R.append(('r-6.3-trim', r"""We train a single operator ($p{=}512$) on a mix of $N{-}0$ and $N{-}1$ scenarios ($9990$
training scenarios: $3240$ $N{-}0$ and $6750$ $N{-}1$, line trips drawn from $25$ training lines)
using the extended encoding of Section~\ref{sec:method}, and evaluate on two held-out axes:""",
r"""We train a single operator ($p{=}512$) on a mix of $N{-}0$ and $N{-}1$ scenarios ($9990$
scenarios, line trips drawn from $25$ training lines) using the extended encoding of
Section~\ref{sec:method}, and evaluate on two held-out axes:"""))

R.append(('r-6.1-tef2', r"""criterion~\cite{wang2018individual} evaluated on the same held-out buses yields $9.6$ ms. The
operator's \ccterr{} ms mean absolute error is larger than both and carries a conservative bias""",
r"""criterion~\cite{wang2018individual} yields $9.6$ ms. The operator's \ccterr{} ms mean absolute
error is larger than both and carries a conservative bias"""))

R.append(('disc-sixth', r"""Sixth, we
model only bolted three-phase faults at buses; parametric fault impedance and faults along line
corridors are left to future work, although the severity encoding is continuous in the fault
admittance and can in principle represent non-bolted faults.""",
r"""Sixth, only bolted three-phase
faults at buses are modeled; parametric fault impedance and faults along line corridors are left
to future work."""))

R.append(('cap-ablation', r"""\caption{Comparison of variants and baselines on the IEEE 39-bus system ($12$ held-out fault
buses, seed 0, max-over-time criterion). RMSE is the pooled stable-trajectory RMSE; ``$-$'' marks
a metric the variant does not produce. The DeepONet variants use $p{=}512$; the MLP, GRU, and GCN
models have no basis dimension. The pointwise MLP and GRU are evaluated under the same
max-over-time criterion, conformal calibration, and CCT search as the operator.}""",
r"""\caption{Comparison of variants and baselines on the IEEE 39-bus system ($12$ held-out fault
buses, seed 0). RMSE is the pooled stable-trajectory RMSE; ``$-$'' marks a metric the variant does
not produce. The pointwise MLP and GRU are evaluated under the same criterion, conformal
calibration, and CCT search as the operator.}"""))

R.append(('cap-n1', r"""\caption{Generalization of the multi-topology operator ($N{-}0$ + $N{-}1$ training). Pooled
stable-trajectory RMSE, classification accuracy, and CCT error (mean over the held-out
bus--line combinations).}""",
r"""\caption{Generalization of the multi-topology operator ($N{-}0$ + $N{-}1$ training): pooled
stable-trajectory RMSE, classification accuracy, and CCT error (mean over held-out
bus--line combinations).}"""))

R.append(('cap-main', r"""\caption{Main results on the IEEE 39-bus system ($12$ held-out fault buses).}""",
r"""\caption{Main results on the IEEE 39-bus system.}"""))

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
