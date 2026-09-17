# -*- coding: utf-8 -*-
"""Post-compression verification of paper_mpce.tex."""
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

new = open(r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex',
           encoding='utf-8').read().replace('\r\n', '\n')

print('figure envs:', new.count(r'\begin{figure}'))
print('table envs:', new.count(r'\begin{table}'))
eqs = sorted(set(re.findall(r'\\label\{(eq:[a-z0-9]+)\}', new)))
print('equation labels (%d):' % len(eqs), eqs)
m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', new, re.S)
body = re.sub(r'\\[A-Za-z]+(\[[^\]]*\])?(\{[^}]*\})?', ' ', m.group(1))
print('abstract words:', len(re.findall(r'[A-Za-z0-9][A-Za-z0-9%\-]*', body)))
m2 = re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', new, re.S)
print('keywords:', ' / '.join(m2.group(1).split()))
print('sections:', re.findall(r'\\section\*?\{([^}]+)\}', new))
print('subsections:', len(re.findall(r'\\subsection\{', new)))
print('cites:', len(re.findall(r'\\cite\{', new)))
print('fig refs:', re.findall(r'\\ref\{(fig:[a-z0-9]+)\}', new))
print('tab refs:', re.findall(r'\\ref\{(tab:[a-z0-9]+)\}', new))
# sanity: no leftover placeholder macro references that were only in deleted content
for name in ['\\cctbusref', '\\cctbuspred', '\\cctbuserr', '\\trajband']:
    print(name, 'used:', new.count(name), 'times')
# acknowledgments + declarations + bios present
for kw in ['Acknowledgments', 'Conflict of interest', 'Author biographies', 'CRediT authorship', '[TOOL NAME']:
    print(kw, '->', kw in new)
