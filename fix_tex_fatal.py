# -*- coding: utf-8 -*-
"""Fix the same fatal errors in the LaTeX version (paper_mpce.tex + references.bib)."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TEX = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
BIB = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\references.bib'

# ---------------- references.bib ----------------
b = open(BIB, encoding='utf-8').read()
b = b.replace('Pitt, Zachary', 'Pitt, David')
b = b.replace('Ma, Zhiyuan', 'Ma, Ziqi')
assert 'Pitt, David' in b and 'Ma, Ziqi' in b
open(BIB, 'w', encoding='utf-8', newline='\n').write(b)
print('bib: Pitt -> David, Ma -> Ziqi')

# ---------------- paper_mpce.tex ----------------
t = open(TEX, encoding='utf-8').read().replace('\r\n', '\n')
R = []

R.append(('preamble-roman', r'''\newcommand{\sgn}{\mathrm{sgn}}''',
r'''\renewcommand{\thetable}{\Roman{table}}
\newcommand{\sgn}{\mathrm{sgn}}'''))

R.append(('abstract-pct', r'''Retrained on the IEEE 118-bus system it reaches
\pctACCb{}$\%$ accuracy''',
r'''Retrained on the IEEE 118-bus system it reaches
\pctACCb{} accuracy'''))

R.append(('dataavail-roman', r'''Figures~1--3 and Tables~1--3---is available at''',
r'''Figures~1--3 and Tables~I--III---is available at'''))

R.append(('trajtc', r'''\newcommand{\trajtc}{0.1700}''',
r'''\newcommand{\trajtc}{0.170}'''))

R.append(('caption-deg', r'''the shaded band is the first-swing conformal band ($\trajband{}$ at
$\alpha{=}0.1$)''',
r'''the shaded band is the first-swing conformal band ($\trajband{}^\circ$ at
$\alpha{=}0.1$)'''))

R.append(('accseed', r'''\newcommand{\accSeed}{94.0\%$\pm$1.2\%}''',
r'''\newcommand{\accSeed}{94.0\%\,$\pm$\,1.2\%}'''))

R.append(('seed-rmse', r'''the per-scenario mean RMSE is
$20.2^\circ\pm1.4^\circ$''',
r'''the per-scenario mean RMSE is
$20.2^\circ\,\pm\,1.4^\circ$'''))

R.append(('confmat-space', r'''confusion matrix (TP, FP, FN, TN) $=$\confmat{} ($F_1$ for the stable class''',
r'''confusion matrix (TP, FP, FN, TN) $=$ \confmat{} ($F_1$ for the stable class'''))

R.append(('gru', r'''and a recurrent (GRU)
sequence-to-sequence baseline.''',
r'''and a gated-recurrent-unit (GRU)
sequence-to-sequence baseline.'''))

R.append(('mse', r'''the GCNs use binary cross-entropy (or MSE) and $300$ epochs''',
r'''the GCNs use binary cross-entropy (or mean squared error (MSE)) and $300$ epochs'''))

R.append(('pebs', r'''the transient-energy-function (TEF/PEBS)
method~\cite{athay1979transient}''',
r'''the transient-energy-function (TEF) / potential-energy-boundary-surface (PEBS)
method~\cite{athay1979transient}'''))

R.append(('bios', r'''\section*{Author biographies}
\noindent\textbf{Zhenyu Liu} received ... [PLACEHOLDER: education background and research
interests to be filled in by the authors].\\
\textbf{Jiale Wang} ... [PLACEHOLDER].\\
\textbf{Jiayong Liu} ... [PLACEHOLDER].\\
\textbf{Zhicheng Zeng} ... [PLACEHOLDER].''',
r'''\section*{Author biographies}
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
analysis and deep learning.'''))

failed = []
for name, old, new in R:
    if old not in t:
        failed.append(name)
        print('FAIL:', name)
    else:
        t = t.replace(old, new, 1)
        print('ok :', name)

if failed:
    print('ABORTED, %d failed' % len(failed))
else:
    open(TEX, 'wb').write(t.replace('\n', '\r\n').encode('utf-8'))
    print('TEX WRITTEN')
