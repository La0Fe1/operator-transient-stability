# -*- coding: utf-8 -*-
"""Final docx round-B: token-aware paragraph rebuilds and regex replacements."""
import io, sys, re, zipfile, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from lxml import etree

DOCX = r'C:\Users\佬肥\Desktop\算子\A Physics-Encoded Neural Operator with Conformal Trajectory Bands for Fast Transient Stability Screening_Liu_Wang_Liu_Zeng_MPCE_revised.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
XMLSP = '{http://www.w3.org/XML/1998/namespace}space'
TOK = re.compile(r'\x00M(\d+)\x00')

zin = zipfile.ZipFile(DOCX)
tree = etree.fromstring(zin.read('word/document.xml'))
body = tree

def para_tok(p):
    xml = etree.tostring(p).decode('utf-8')
    maths = re.findall(r'<m:oMathPara>.*?</m:oMathPara>|<m:oMath>.*?</m:oMath>', xml, re.S)
    for i, mblk in enumerate(maths):
        xml = xml.replace(mblk, '\x00M%d\x00' % i)
    text = re.sub(r'<w:[^>]*>', '', xml)
    text = re.sub(r'</w:[^>]*>', '', text)
    text = (text.replace('&amp;', '&').replace('&#8212;', chr(8212))
                .replace('&#8217;', chr(8217)).replace('&#160;', ' ')
                .replace('&#8203;', '').replace('&#8211;', chr(8211)))
    return text, maths

def rebuild(p, text, maths):
    pPr = p.find('./w:pPr', NS)
    for child in list(p):
        p.remove(child)
    if pPr is not None:
        p.append(pPr)
    for seg in TOK.split(text):
        if seg == '':
            continue
        if TOK.fullmatch('\x00M' + seg + '\x00') or (seg.isdigit()):
            # token case: seg is the group digits
            pass
    # simpler: iterate with finditer over tokens
    pos = 0
    for m in TOK.finditer(text):
        if m.start() > pos:
            seg = text[pos:m.start()]
            r = etree.SubElement(p, W + 'r')
            t = etree.SubElement(r, W + 't'); t.set(XMLSP, 'preserve'); t.text = seg
        idx = int(m.group(1))
        wrap = ('<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
                + maths[idx] + '</w:r>')
        p.append(etree.fromstring(wrap)[0])
        pos = m.end()
    if pos < len(text):
        seg = text[pos:]
        r = etree.SubElement(p, W + 'r')
        t = etree.SubElement(r, W + 't'); t.set(XMLSP, 'preserve'); t.text = seg

def tok_sub(p, old_re, new, count=1):
    text, maths = para_tok(p)
    if not re.search(old_re, text):
        return False
    text2 = re.sub(old_re, new, text, count=count)
    rebuild(p, text2, maths)
    return True

MTOK = r'\x00M\d+\x00'

done = []

# --- 6.3 paragraph: full rebuild ---
for p in body.findall('.//w:p', NS):
    text, maths = para_tok(p)
    if 'We train a single operator (' in text:
        new = ('We train the multi-topology operator (\x00M0\x00) with the extended encoding of '
               'Section\u00a04. The N\u22121 experiment trains on \x00M3\x00 \x00M1\x00/\x00M2\x00 '
               'scenarios (line trips drawn from \x00M4\x00 training lines) and evaluates on two '
               'held-out axes: unseen fault buses and \x00M5\x00 unseen trip lines. The N\u22122 '
               'experiment trains on 8748 scenarios that additionally include N\u22122 line-pair '
               'scenarios drawn from the training lines, and evaluates on held-out line pairs. '
               'On held-out line trips the operator achieves \x00M6\x00 stable-trajectory RMSE and '
               '\x00M7\x00 accuracy, demonstrating that the post-fault severity encoding lets the '
               'operator generalize to topologies never seen in training; on held-out fault buses '
               'the RMSE is \x00M8\x00 and the accuracy \x00M9\x00 (Table\u00a0III), reflecting the '
               'harder combined task. The same encoding and architecture apply to \x00M10\x00 '
               'contingencies (two lines tripped) without modification: on held-out line pairs the '
               'operator attains \x00M11\x00 RMSE and \x00M12\x00 accuracy, because the post-fault '
               'severity vector is simply evaluated from the admittance matrix with both lines '
               'removed. The multi-topology operator also estimates the CCT on held-out '
               'combinations: \x00M13\x00 ms mean absolute error on unseen trip lines, '
               '\x00M14\x00 ms on held-out fault buses, and 18.7 ms on held-out \x00M10\x00 line '
               'pairs (sampled); split-conformal coverage on these splits is reported in '
               'Section\u00a06.')
        rebuild(p, new, maths)
        done.append('6.3 rebuild')
        break

# --- 4.5 protocol paragraph ---
for p in body.findall('.//w:p', NS):
    text, maths = para_tok(p)
    if 'Given the operator, the CCT of a fault is estimated by binary search' in text:
        new = ('Given the operator, the CCT of a fault is the first clearing time at which the '
               'criterion is violated. Because neither the operator\u2019s criterion nor the '
               'reference is guaranteed monotone in \x00M0\x00, a 5 ms sweep over \x00M3\x00 s '
               'locates the first stable-to-unstable sign change, which is then refined by local '
               'bisection to 0.1 ms; the same first-crossing protocol (with the system-dependent '
               'range \x00M4\x00 s) is applied to the TDS reference, so the two CCT estimates are '
               'directly comparable. The refinement tolerance is the search resolution, not the '
               'physical CCT accuracy; the reference uses the event-aligned RK4 scheme of '
               'Section\u00a03 (convergence verified at a \x00M6\x00 ms step), and both judge '
               'stability on the \x00M7\x00 s grid.')
        rebuild(p, new, maths)
        done.append('4.5 protocol')
        break

# --- intro bullet 4 ---
for p in body.findall('.//w:p', NS):
    text, maths = para_tok(p)
    if 'a single operator generalizes to ' in text:
        new = ('We extend the encoding with a post-fault severity vector and show that the '
               'operator generalizes to held-out \x00M0\x00 line trips and \x00M1\x00 line pairs.')
        rebuild(p, new, maths)
        done.append('bullet4')
        break

# --- piece-wise / regex replacements ---
edits = [
 (r'where the binary search lands between multiple local stable-to-unstable crossings of the operator\u2019s criterion\.',
  'where the criterion has multiple local stable-to-unstable crossings; the first-crossing search of Section\u00a04 reproduces the previous binary-search CCTs within 3 ms per bus.'),
 (r'The operator\u2019s 25\.2 ms mean absolute error',
  'The operator\u2019s 25.0 ms mean absolute error'),
 (r'against the target level ' + MTOK,
  'against the target coverage level 1\u2212' + '\x00M0\x00'),
 (r'rising to ' + MTOK + r' at ' + MTOK + r' and ' + MTOK + r' at ' + MTOK + r' as ',
  'rising to \x00M6\x00 at \x00M7\x00, 57% at \u03b1=0.4, and \x00M8\x00 at \x00M9\x00 as '),
 (r'These numbers use the base ' + MTOK + r' architecture \(Table\u00a0II reports ' + MTOK + r'\), which explains the different full-data values\.',
  ''),
 (r'leaves the severity encoding unchanged\u2014it is evaluated at the pre-fault equilibrium, where the damping term vanishes\u2014so',
  'leaves the encoding unchanged (the damping term vanishes at the pre-fault equilibrium), so'),
 (r'a stronger graph baseline with three residual layers and concatenated mean/max/sum pooling attains',
  'a stronger graph baseline with three residual layers attains'),
 (r'not an artifact of a weak graph architecture',
  'not an artifact of a weak architecture'),
 (r'Alternative scores \(amplitude-normalized ' + MTOK + r', first-swing ' + MTOK + r', time-averaged ' + MTOK + r'; see the repository\), adaptive/locally adaptive',
  'Alternative scores (amplitude-normalized, first-swing, time-averaged; see the repository), adaptive'),
]
for old_re, new in edits:
    hit = False
    for p in body.findall('.//w:p', NS):
        if tok_sub(p, old_re, new):
            done.append(old_re[:40])
            hit = True
            break
    if not hit:
        print('MISS:', old_re[:60])

# --- remove remaining '&' math runs (swing equations) ---
removed = 0
for om in body.findall('.//m:oMath', NS):
    for r in list(om.findall('.//m:r', NS)):
        rt = ''.join(t.text or '' for t in r.findall('.//m:t', NS))
        if rt.strip() == '&':
            r.getparent().remove(r)
            removed += 1
print('& removed:', removed)

tmp = DOCX + '.tmp'
with zipfile.ZipFile(DOCX) as zr, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zw:
    for item in zr.namelist():
        data = zr.read(item)
        if item == 'word/document.xml':
            data = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
        zw.writestr(item, data)
shutil.move(tmp, DOCX)
print('DONE:', done)
print('SAVED')
