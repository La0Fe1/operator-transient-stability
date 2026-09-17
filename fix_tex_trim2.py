# -*- coding: utf-8 -*-
"""Final ~8-line trim to land the LaTeX version at 10 pages."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TEX = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
t = open(TEX, encoding='utf-8').read().replace('\r\n', '\n')
R = []
R.append(('intro-throughout', r'''Throughout, ``physics-informed''
refers to the input encoding derived from the swing equation, not to a residual-constrained model,
and the conformal output is a calibrated trajectory band, not a stability certificate.''',
r'''Throughout, ``physics-informed''
refers to the input encoding, and the conformal output is a calibrated trajectory band, not a
stability certificate.'''))
R.append(('m4.5-trim', r'''The $12\,\mu$s interval is the
\emph{search tolerance} of the bisection, not the physical accuracy of the CCT; the reference
itself uses the event-aligned RK4 scheme of Section~\ref{sec:problem}, whose convergence was
verified at a $0.25$ ms step, and both judge stability on the $0.01$ s output grid.''',
r'''The $12\,\mu$s interval is the
\emph{search tolerance} of the bisection, not the physical CCT accuracy; the reference uses the
event-aligned RK4 scheme of Section~\ref{sec:problem} (convergence verified at a $0.25$ ms step),
and both judge stability on the $0.01$ s grid.'''))
R.append(('gru-lowrank', r'''worse than the operator's
\rmseAll{}$^\circ$: the low-rank operator structure outperforms a generic recurrent decoder.''',
r'''worse than the operator's
\rmseAll{}$^\circ$: the low-rank structure outperforms a generic recurrent decoder.'''))
R.append(('exciter-trim', r'''Adding a first-order automatic voltage regulator (a simplified IEEE DC1A
exciter with $K_A{=}20$, $T_A{=}0.2$ s and rate limits) yields a fourth-order model on which the''',
r'''Adding a first-order automatic voltage regulator (a simplified IEEE DC1A
exciter) yields a fourth-order model on which the'''))
R.append(('mondrian-n', r'''(group $48/51$ calibration/evaluation,
coverage $0.863$)''',
r'''(group $48/51$,
coverage $0.863$)'''))
R.append(('oracle-trim', r'''(\gatenum{} scenarios; the last conditions on ground-truth labels and is not a deployment
guarantee, since conditioning on the prediction breaks exchangeability)''',
r'''(\gatenum{} scenarios; the last is an
oracle-conditioned diagnostic, not a deployment guarantee)'''))
R.append(('speed-trim', r'''The RK4 cost scales
inversely with the reference step size, whereas the operator's cost depends only on the number of
output time points ($401$ per trajectory), so the speedup grows for finer reference steps; all
timing details are in the repository''',
r'''The RK4 cost scales
inversely with the reference step size, whereas the operator's cost depends only on the output
time points ($401$ per trajectory); all timing details are in the repository'''))
R.append(('disc-vacuous', r'''and on the 118-bus case the per-machine band
is currently vacuous for decisions.''',
r'''and on the 118-bus case the per-machine band
is currently vacuous.'''))
R.append(('118-trim', r'''so on the 118-bus case the interval validates the
coverage mechanism but is vacuous for the stability decision; the directly calibrated pairwise
score and richer calibration data are required before the bound is operationally useful at this
scale.''',
r'''so on the 118-bus case the interval validates the
coverage mechanism but is vacuous for decisions; the pairwise score and richer calibration data
are required before the bound is operationally useful.'''))
failed = []
for name, old, new in R:
    if old not in t:
        failed.append(name); print('FAIL:', name)
    else:
        t = t.replace(old, new, 1); print('ok :', name)
if failed:
    print('ABORTED')
else:
    open(TEX, 'wb').write(t.replace('\n', '\r\n').encode('utf-8'))
    print('WRITTEN')
