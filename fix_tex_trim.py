# -*- coding: utf-8 -*-
"""Reclaim ~9 lines to return to 10 pages; ~ spacing for seed stats; shorter bios."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TEX = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
t = open(TEX, encoding='utf-8').read().replace('\r\n', '\n')
R = []
R.append(('seed-mac', r'''\newcommand{\accSeed}{94.0\%\,$\pm$\,1.2\%}''',
r'''\newcommand{\accSeed}{94.0\%~$\pm$~1.2\%}'''))
R.append(('seed-rmse', r'''$20.2^\circ\,\pm\,1.4^\circ$''',
r'''$20.2^\circ~$\pm$~1.4^\circ$'''))
R.append(('baseline-feats', r'''raw node features (voltage magnitudes and angles, loads, and a fault flag)''',
r'''raw node features (voltages, angles, loads, and a fault flag)'''))
R.append(('gated-trim', r'''In the gated regime (band applied only after a
predicted-stable decision, evaluated on the disjoint half): gate purity''',
r'''In the gated regime (evaluated on the disjoint half): gate purity'''))
R.append(('mult-trim', r'''Multiplicity-corrected
per-machine bounds (Bonferroni/\v{S}id\'ak) do not tighten the pairwise bound
(Section~\ref{sec:results}); any joint guarantee would additionally require the structural
conditions of the chosen correction.''',
r'''Multiplicity-corrected
per-machine bounds (Bonferroni/\v{S}id\'ak) do not tighten the pairwise bound
(Section~\ref{sec:results}).'''))
R.append(('bonf-dup', r'''Per-machine calibration with Bonferroni/\v{S}id\'ak corrections does not tighten the
bound (pairwise $734^\circ$), and calibrating''',
r'''Per-machine calibration with Bonferroni/\v{S}id\'ak corrections does not tighten the
bound, and calibrating'''))
R.append(('sixth-trim', r'''Sixth, only bolted three-phase
faults at buses are modeled; parametric fault impedance and faults along line corridors are left
to future work.''',
r'''Sixth, only bolted three-phase faults are
modeled; parametric fault impedance and line-corridor faults are left to future work.'''))
R.append(('bios-short', r'''\section*{Author biographies}
\noindent\textbf{Zhenyu Liu} is currently pursuing the B.Eng. degree at the International
Energy College, Jinan University, Guangzhou, China. His research interests include power system
stability analysis, machine learning, and uncertainty quantification.\\
\textbf{Jiale Wang} is currently pursuing the B.Eng. degree at the International Energy
College, Jinan University, Guangzhou, China. His research interests include power system
simulation and artificial intelligence applications in power engineering.\\
\textbf{Jiayong Liu} is currently pursuing the B.Eng. degree at the International Energy
College, Jinan University, Guangzhou, China. His research interests include power system
dynamics and data-driven methods.\\
\textbf{Zhicheng Zeng} is currently pursuing the B.Eng. degree at the International Energy
College, Jinan University, Guangzhou, China. His research interests include power system
analysis and deep learning.''',
r'''\section*{Author biographies}
\noindent\textbf{Zhenyu Liu} is pursuing the B.Eng. degree at the International Energy College,
Jinan University, Guangzhou, China. His research interests include power system stability and
machine learning.\\
\textbf{Jiale Wang} is pursuing the B.Eng. degree at the International Energy College, Jinan
University, Guangzhou, China. His research interests include power system simulation and
artificial intelligence.\\
\textbf{Jiayong Liu} is pursuing the B.Eng. degree at the International Energy College, Jinan
University, Guangzhou, China. His research interests include power system dynamics and
data-driven methods.\\
\textbf{Zhicheng Zeng} is pursuing the B.Eng. degree at the International Energy College, Jinan
University, Guangzhou, China. His research interests include power system analysis and deep
learning.'''))
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
