# -*- coding: utf-8 -*-
"""Third pass: math-node fixes (abstract %, caption deg/tc, +/- normalization)."""
import io, sys, re, zipfile, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from lxml import etree

DOCX = r'C:\Users\佬肥\Desktop\算子\A Physics-Encoded Neural Operator with Conformal Trajectory Bands for Fast Transient Stability Screening_Liu_Wang_Liu_Zeng_MPCE_revised.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
XMLSP = '{http://www.w3.org/XML/1998/namespace}space'

zin = zipfile.ZipFile(DOCX)
tree = etree.fromstring(zin.read('word/document.xml'))
body = tree

def omath_text(om):
    return ''.join(t.text or '' for t in om.findall('.//m:t', NS))

# 1. remove the stray % math right after '92.9%' in the abstract
for p in body.findall('.//w:p', NS):
    joined = ''.join(t.text or '' for t in p.findall('.//w:t', NS))
    if '92.9%' in joined and 'Retrained' in joined:
        # iterate children of paragraph; find oMath whose text is '%' following a '92.9%' run
        seen_929 = False
        for child in list(p):
            if child.tag == W + 'r':
                for t in child.findall('.//w:t', NS):
                    if '92.9%' in (t.text or ''):
                        seen_929 = True
            elif child.tag == M + 'oMath' and seen_929:
                if omath_text(child).strip() == '%':
                    p.remove(child)
                    print('abstract math %% removed')
                    seen_929 = False
                    break

# 2. m:t fixes: degree in caption, tc decimals
for node in body.findall('.//m:t', NS):
    if node.text == '70.8':
        node.text = '70.8°'
        print('caption degree added')
    if node.text == '0.1700':
        node.text = '0.170'
        print('caption tc decimals fixed')

# 3. normalize inline +/- statistics to spaced plain text
for om in list(body.findall('.//m:oMath', NS)):
    s = omath_text(om)
    if s == '77.3%±3.9%' or s == '0.90±0.02':
        r = etree.Element(W + 'r')
        t = etree.SubElement(r, W + 't'); t.set(XMLSP, 'preserve')
        t.text = s.replace('±', ' ± ')
        om.addnext(r)
        om.getparent().remove(om)
        print('normalized:', s)

# 4. remaining zero-width spaces (incl. math)
n = 0
for node in body.findall('.//w:t', NS) + body.findall('.//m:t', NS):
    if node.text and '\u200b' in node.text:
        node.text = node.text.replace('\u200b', '')
        n += 1
print('remaining zero-width spaces cleaned:', n)

tmp = DOCX + '.tmp'
with zipfile.ZipFile(DOCX) as zr, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zw:
    for item in zr.namelist():
        data = zr.read(item)
        if item == 'word/document.xml':
            data = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
        zw.writestr(item, data)
shutil.move(tmp, DOCX)
print('SAVED')

# verification
z = zipfile.ZipFile(DOCX)
doc2 = z.read('word/document.xml').decode('utf-8')
def txt(x): return re.sub(r'<[^>]+>', ' ', x)
t2 = txt(doc2)
i = t2.find('Retrained on the IEEE')
print('abstract now:', repr(t2[i:i+130]))
i = t2.find('first-swing conformal band')
print('caption now:', repr(t2[i:i+110]))
for pat in ['70.8°', '0.170 ', '77.3% ± 3.9%', '0.90 ± 0.02', '94.0% ± 1.2%', '20.2° ± 1.4°']:
    print(pat, '->', pat in t2)
