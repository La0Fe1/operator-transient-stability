# -*- coding: utf-8 -*-
"""Token-aware CCT number updates in the docx (with lambda replacements)."""
import io, sys, zipfile, shutil, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from lxml import etree

DOCX = r'C:\Users\佬肥\Desktop\算子\A Physics-Encoded Neural Operator with Conformal Trajectory Bands for Fast Transient Stability Screening_Liu_Wang_Liu_Zeng_MPCE_revised.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
XMLSP = '{http://www.w3.org/XML/1998/namespace}space'
TOKEN_RE = re.compile(r'\x00M(\d+)\x00')

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
                .replace('&#8217;', chr(8217)).replace('&#160;', ' ').replace('&#8203;', ''))
    return text, maths

def rebuild(p, text, maths):
    pPr = p.find('./w:pPr', NS)
    for child in list(p):
        p.remove(child)
    if pPr is not None:
        p.append(pPr)
    pos = 0
    for m in TOKEN_RE.finditer(text):
        if m.start() > pos:
            r = etree.SubElement(p, W + 'r')
            t = etree.SubElement(r, W + 't'); t.set(XMLSP, 'preserve'); t.text = text[pos:m.start()]
        idx = int(m.group(1))
        wrap = ('<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
                + maths[idx] + '</w:r>')
        p.append(etree.fromstring(wrap)[0])
        pos = m.end()
    if pos < len(text):
        r = etree.SubElement(p, W + 'r')
        t = etree.SubElement(r, W + 't'); t.set(XMLSP, 'preserve'); t.text = text[pos:]

def tok_sub(p, old_re, repl_fn):
    text, maths = para_tok(p)
    if not re.search(old_re, text):
        return False
    rebuild(p, re.sub(old_re, repl_fn, text, count=1), maths)
    return True

NUL = chr(0)
TOK = NUL + r'M\d+' + NUL
edits = [
 # N-1/N-2 CCT sentence: keep token indices M13/M14/M10 (from the 6.3 rebuild)
 (NUL + r'M13' + NUL + r' ms mean absolute error on unseen trip lines, '
  + NUL + r'M14' + NUL + r' ms on held-out fault buses, and 18\.7 ms on held-out',
  lambda m: NUL + 'M13' + NUL + ' ms mean absolute error on unseen trip lines (sampled), '
            + NUL + 'M14' + NUL + ' ms on held-out fault buses (sampled), and 39.7 ms on held-out'),
 # 118 CCT sentence
 (r'with a CCT error of (' + TOK + r') ms mean absolute \(max (' + TOK
  + r') ms\) on the (' + TOK + r') held-out buses\.',
  lambda m: 'with a CCT error of ' + m.group(1) + ' ms mean absolute (max ' + m.group(2)
            + ' ms) among the 11 buses with a defined first crossing.'),
]
for old_re, repl_fn in edits:
    hit = False
    for p in body.findall('.//w:p', NS):
        if tok_sub(p, old_re, repl_fn):
            hit = True; print('ok:', old_re[:40].encode('unicode_escape').decode()[:60])
            break
    if not hit:
        print('MISS')

tmp = DOCX + '.tmp'
with zipfile.ZipFile(DOCX) as zr, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zw:
    for item in zr.namelist():
        data = zr.read(item)
        if item == 'word/document.xml':
            data = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
        zw.writestr(item, data)
shutil.move(tmp, DOCX)
print('SAVED')
# verify
z2 = zipfile.ZipFile(DOCX)
t2 = re.sub(r'<[^>]+>', ' ', z2.read('word/document.xml').decode('utf-8'))
flat = re.sub(r'\s+', ' ', t2)
for k in ['19.7', '34.1', '39.7', '9.9', '22.3', 'defined first crossing', '18.7 ms', '13.3', '54.7']:
    print(k, '->', k in flat)
