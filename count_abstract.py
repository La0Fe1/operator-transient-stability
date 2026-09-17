# -*- coding: utf-8 -*-
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
t = open(r'C:\Users\佬肥\PycharmProjects\PythonProject\operator-transient-stability\paper_mpce.tex',
         encoding='utf-8').read().replace('\r\n', '\n')
m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S)
body = m.group(1)
# remove math
body = re.sub(r'\$[^$]*\$', ' ', body)
# remove commands with up to one arg
body = re.sub(r'\\[A-Za-z]+', ' ', body)
# remove braces
body = body.replace('{', ' ').replace('}', ' ')
words = re.findall(r"[A-Za-z][A-Za-z0-9'%\-]*", body)
print('tex abstract words:', len(words))
print('---')
print(' '.join(words[:40]))
