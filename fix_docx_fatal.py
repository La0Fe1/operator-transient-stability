# -*- coding: utf-8 -*-
"""Fix all fatal errors from 论文1 投稿评估报告 in the revised MPCE docx."""
import io, sys, re, zipfile, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from lxml import etree

DOCX = r'C:\Users\佬肥\Desktop\算子\A Physics-Encoded Neural Operator with Conformal Trajectory Bands for Fast Transient Stability Screening_Liu_Wang_Liu_Zeng_MPCE_revised.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
WNS = NS['w']

zin = zipfile.ZipFile(DOCX)
doc = zin.read('word/document.xml')

# ---------------- 1. headers ----------------
JOURNAL_HDR = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
 '<w:p><w:pPr><w:jc w:val="center"/><w:pBdr><w:bottom w:val="single" w:sz="4" w:space="1" w:color="auto"/></w:pBdr>'
 '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="16"/></w:rPr></w:pPr>'
 '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="16"/></w:rPr>'
 '<w:t>JOURNAL OF MODERN POWER SYSTEMS AND CLEAN ENERGY</w:t></w:r></w:p></w:hdr>')
RUNHEAD = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
 '<w:p><w:pPr><w:jc w:val="center"/><w:rPr><w:i/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="16"/></w:rPr></w:pPr>'
 '<w:r><w:rPr><w:i/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="16"/></w:rPr>'
 '<w:fldSimple w:instr=" PAGE "><w:r><w:rPr><w:i/></w:rPr><w:t>7</w:t></w:r></w:fldSimple></w:r>'
 '<w:r><w:rPr><w:i/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="16"/></w:rPr>'
 '<w:t xml:space="preserve">  |  </w:t></w:r>'
 '<w:r><w:rPr><w:i/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="16"/></w:rPr>'
 '<w:t>LIU et al.: A PHYSICS-ENCODED NEURAL OPERATOR WITH CONFORMAL TRAJECTORY BANDS</w:t></w:r>'
 '</w:p></w:hdr>')

tree = etree.fromstring(doc)
body = tree

# helper: all w:t nodes (body only)
def all_t(body):
    return body.findall('.//w:t', NS)

# ---------------- 2. table numbering: arabic -> roman (in-text) ----------------
tmap = {'Tables 1–3': 'Tables I–III', 'Tables 1-3': 'Tables I–III',
        'Table 3': 'Table III', 'Table 2': 'Table II', 'Table 1': 'Table I',
        'Tables 1': 'Tables I'}
nrep = 0
for node in all_t(body):
    if node.text:
        for a, b in tmap.items():
            if a in node.text:
                node.text = node.text.replace(a, b)
                nrep += 1
print('table-number replacements:', nrep)

# ---------------- 3. eq labels -> numbers ----------------
eqmap = {'[eq:swing1]': '(1)', '[eq:swing2]': '(2)',
         '[eq:severity]': '(5)', '[eq:postseverity]': '(6)',
         '[eq:physloss]': '(8)'}
nrep = 0
for node in all_t(body):
    if node.text:
        for a, b in eqmap.items():
            if a in node.text:
                node.text = node.text.replace(a, b)
                nrep += 1
print('eq-label replacements:', nrep)

# ---------------- 4. reference [20]: Z. Pitt -> D. Pitt ----------------
for node in all_t(body):
    if node.text and 'Z. Pitt' in node.text:
        node.text = node.text.replace('Z. Pitt', 'D. Pitt')
        print('Pitt fixed:', node.text[:60])

# ---------------- 5. abstract double percent ----------------
fixed_pct = 0
for p in body.findall('.//w:p', NS):
    texts = [n.text or '' for n in p.findall('.//w:t', NS)]
    joined = ''.join(texts)
    if '92.9%' in joined and joined.count('%') >= 2 and 'Retrained' in joined:
        # remove the standalone % run(s) after the 92.9% run
        prev_pct = False
        for node in p.findall('.//w:t', NS):
            t = node.text or ''
            if t.strip() == '%' and prev_pct:
                node.getparent().remove(node.getparent() if node.getparent().tag == W+'r' else node)
                fixed_pct += 1
            prev_pct = ('92.9%' in t)
print('abstract % fixes:', fixed_pct)

# ---------------- 6. caption pollution before TABLE II/III ----------------
for p in body.findall('.//w:p', NS):
    texts = [n.text or '' for n in p.findall('.//w:t', NS)]
    joined = ''.join(texts)
    if 'TABLE II' in joined or 'TABLE III' in joined:
        # find the first w:t containing 'TABLE', delete all preceding sibling nodes in the paragraph
        first = None
        for node in p.findall('.//w:t', NS):
            if 'TABLE' in (node.text or ''):
                first = node
                break
        if first is not None:
            run = first.getparent()  # w:r
            # walk back through paragraph children before this run, remove
            for child in list(p):
                if child is run:
                    break
                p.remove(child)
            if first.text.startswith(' '):
                pass
            print('cleaned caption:', repr(first.text[:30]))

# ---------------- 7. equations: split eqArr, number (1)-(10) ----------------
def first_tokens(omathpara):
    return ''.join(t.text or '' for t in omathpara.findall('.//m:t', NS))[:50]

# collect display equation paragraphs in order
eqparas = []
for p in body.findall('.//w:p', NS):
    op = p.find('./m:oMathPara', NS)
    if op is not None:
        eqparas.append((p, op))

# split the eqArr paragraph (should be the first one)
p0, op0 = eqparas[0]
omath0 = op0.find('./m:oMath', NS)
eqarr = omath0.find('./m:eqArr', NS)
assert eqarr is not None, 'first display equation is not eqArr'
rows = eqarr.findall('./m:e', NS)
assert len(rows) == 2, 'expected 2 rows'

def make_eq_para(row, num):
    p = etree.SubElement(p0.getparent(), W + 'p') if False else None
    p = etree.Element(W + 'p')
    pPr = etree.SubElement(p, W + 'pPr')
    ps = etree.SubElement(pPr, W + 'pStyle'); ps.set(W + 'val', 'BodyText')
    tabs = etree.SubElement(pPr, W + 'tabs')
    tab = etree.SubElement(tabs, W + 'tab'); tab.set(W + 'val', 'right'); tab.set(W + 'pos', '4860')
    op = etree.SubElement(p, M + 'oMathPara')
    opPr = etree.SubElement(op, M + 'oMathParaPr')
    jc = etree.SubElement(opPr, M + 'jc'); jc.set(M + 'val', 'center')
    om = etree.SubElement(op, M + 'oMath')
    om.append(row)
    r = etree.SubElement(p, W + 'r')
    t = etree.SubElement(r, W + 't'); t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = '(%d)' % num
    return p

pA = make_eq_para(rows[0], 1)
pB = make_eq_para(rows[1], 2)
p0.addnext(pB)
p0.addnext(pA)
# remove original eqArr paragraph
p0.getparent().remove(p0)
eqparas = eqparas[1:]

# number remaining display equations (3..10) in document order
num = 3
for p, op in eqparas:
    pPr = p.find('./w:pPr', NS)
    if pPr is None:
        pPr = etree.Element(W + 'pPr'); p.insert(0, pPr)
    tabs = etree.SubElement(pPr, W + 'tabs')
    tab = etree.SubElement(tabs, W + 'tab'); tab.set(W + 'val', 'right'); tab.set(W + 'pos', '4860')
    r = etree.SubElement(p, W + 'r')
    t = etree.SubElement(r, W + 't'); t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = '(%d)' % num
    num += 1
print('equations numbered: 1..%d' % (num - 1))

# ---------------- 8. zero-width space cleanup ----------------
nzw = 0
for node in all_t(body):
    if node.text and '\u200b' in node.text:
        node.text = node.text.replace('\u200b', '')
        nzw += 1
print('zero-width spaces cleaned:', nzw)

# ---------------- 9. caption: degree symbol + tc decimals ----------------
for node in all_t(body):
    if node.text:
        if '70.8 at' in node.text:
            node.text = node.text.replace('70.8 at', '70.8° at')
            print('deg fixed in caption')
        if '0.1700' in node.text:
            node.text = node.text.replace('0.1700', '0.170')
            print('tc decimals fixed')

# ---------------- 10. data availability spacing ----------------
for p in body.findall('.//w:p', NS):
    texts = [n.text or '' for n in p.findall('.//w:t', NS)]
    joined = ''.join(texts)
    if 'available at' in joined and 'together with the random seeds' in joined:
        for node in p.findall('.//w:t', NS):
            if node.text == ' ,' or node.text == ',' and False:
                pass
        # normalize: remove space-before-comma nodes
        prev = None
        for node in p.findall('.//w:t', NS):
            if (node.text or '').strip() == ',' and prev is not None and (prev.text or '').endswith(' '):
                prev.text = prev.text.rstrip()
            if node.text:
                prev = node
        print('data availability comma cleaned')

# ---------------- 11. GRU / MSE / PEBS expansions ----------------
for node in all_t(body):
    if node.text:
        if '(GRU)' in node.text and 'recurrent' in node.text:
            node.text = node.text.replace('a recurrent (GRU)', 'a gated-recurrent-unit (GRU)')
            print('GRU expanded')
        if '(or MSE)' in node.text:
            node.text = node.text.replace('(or MSE)', '(or mean squared error (MSE))')
            print('MSE expanded')
        if '(TEF/PEBS)' in node.text:
            node.text = node.text.replace('transient-energy-function (TEF/PEBS)',
                                          'transient-energy-function (TEF) / potential-energy-boundary-surface (PEBS)')
            print('PEBS expanded')

# ---------------- 12. confusion matrix spacing (if present) ----------------
for node in all_t(body):
    if node.text and 'TN)' in node.text and '=318' in node.text:
        node.text = node.text.replace('=318', '= 318')
        print('confusion matrix spacing fixed')

# ---------------- 13. seed statistics +/- spacing (math runs) ----------------
# locate oMath containing 94.0%±1.2% and 20.2°±1.4°; replace with plain text
for om in body.findall('.//m:oMath', NS):
    s = ''.join(t.text or '' for t in om.findall('.//m:t', NS))
    if '94.0' in s and '1.2' in s and '±' in s:
        r = etree.Element(W + 'r')
        t = etree.SubElement(r, W + 't'); t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        t.text = '94.0% ± 1.2%'
        om.addnext(r); om.getparent().remove(om)
        print('seed accuracy spacing normalized')
    elif '20.2' in s and '1.4' in s and '±' in s:
        r = etree.Element(W + 'r')
        t = etree.SubElement(r, W + 't'); t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        t.text = '20.2° ± 1.4°'
        om.addnext(r); om.getparent().remove(om)
        print('seed RMSE spacing normalized')

# ---------------- 14. author biographies ----------------
BIOS = [
 ('Zhenyu Liu', ' is currently pursuing the B.Eng. degree at the International Energy College, '
  'Jinan University, Guangzhou, China. His research interests include power system stability '
  'analysis, machine learning, and uncertainty quantification.'),
 ('Jiale Wang', ' is currently pursuing the B.Eng. degree at the International Energy College, '
  'Jinan University, Guangzhou, China. His research interests include power system simulation '
  'and artificial intelligence applications in power engineering.'),
 ('Jiayong Liu', ' is currently pursuing the B.Eng. degree at the International Energy College, '
  'Jinan University, Guangzhou, China. His research interests include power system dynamics and '
  'data-driven methods.'),
 ('Zhicheng Zeng', ' is currently pursuing the B.Eng. degree at the International Energy College, '
  'Jinan University, Guangzhou, China. His research interests include power system analysis and '
  'deep learning.'),
]
bio_idx = 0
for p in body.findall('.//w:p', NS):
    joined = ''.join(n.text or '' for n in p.findall('.//w:t', NS))
    if 'PLACEHOLDER' in joined and bio_idx < 4:
        name, rest = BIOS[bio_idx]
        for child in list(p):
            p.remove(child)
        r1 = etree.SubElement(p, W + 'r')
        rPr = etree.SubElement(r1, W + 'rPr'); etree.SubElement(rPr, W + 'b')
        t1 = etree.SubElement(r1, W + 't'); t1.text = name
        r2 = etree.SubElement(p, W + 'r')
        t2 = etree.SubElement(r2, W + 't'); t2.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        t2.text = rest
        bio_idx += 1
        print('bio filled:', name)
print('bios filled:', bio_idx)

# ---------------- save ----------------
tmp = DOCX + '.tmp'
with zipfile.ZipFile(DOCX) as zr, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zw:
    for item in zr.namelist():
        data = zr.read(item)
        if item == 'word/document.xml':
            data = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
        elif item in ('word/header1.xml', 'word/header3.xml'):
            data = JOURNAL_HDR.encode('utf-8')
        elif item == 'word/header2.xml':
            data = RUNHEAD.encode('utf-8')
        zw.writestr(item, data)
shutil.move(tmp, DOCX)
print('SAVED:', DOCX)
