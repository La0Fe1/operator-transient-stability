# -*- coding: utf-8 -*-
"""Update N-1/N-2/118 CCT numbers from the first-crossing recomputation."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
p = r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex'
t = open(p, encoding='utf-8').read().replace('\r\n', '\n')
R = [
 (r'$19.4$ ms mean absolute error on unseen trip lines, $31.9$ ms on held-out fault buses, and \cctNtwo{} ms on held-out $N{-}2$ line pairs (sampled)',
  r'$19.7$ ms mean absolute error on unseen trip lines (sampled), $34.1$ ms on held-out fault buses (sampled), and \cctNtwo{} ms on held-out $N{-}2$ line pairs (sampled)'),
 (r'with a CCT error of $13.3$ ms mean absolute (max $54.7$ ms) on the $24$ held-out buses.',
  r'with a CCT error of $9.9$ ms mean absolute (max $22.3$ ms) among the 11 buses with a defined first crossing.'),
]
failed = []
for old, new in R:
    if old not in t:
        failed.append(old[:60]); print('FAIL:', old[:70])
    else:
        t = t.replace(old, new, 1); print('ok')
open(p, 'wb').write(t.replace('\n', '\r\n').encode('utf-8'))
print('failed:', failed)
