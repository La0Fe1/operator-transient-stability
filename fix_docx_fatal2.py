# -*- coding: utf-8 -*-
"""Second pass: cross-node table numbering, abstract %, bios, bare +/- math, header2 XML."""
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
doc = zin.read('word/document.xml')
tree = etree.fromstring(doc)
body = tree

def all_t(el):
    return el.findall('.//w:t', NS)

# ---------------- 1. table numbering (cross-node) ----------------
num2roman = {'1': 'I', '2': 'II', '3': 'III', '1–3': 'I–III', '1-3': 'I–III'}
fixed = 0
for p in body.findall('.//w:p', NS):
    nodes = p.findall('.//w:t', NS)
    for idx, node in enumerate(nodes):
        t = node.text or ''
        # single-node cases
        for a, b in [('Tables\xa01–3', 'Tables\xa0I–III'), ('Tables 1–3', 'Tables I–III'),
                     ('Tables\xa01-3', 'Tables\xa0I–III')]:
            if a in t:
                node.text = t.replace(a, b); fixed += 1
        # cross-node: node ends with 'Table'/'Tables' (with trailing nbsp or space),
        # next node is a bare number
        if re.search(r'(Tables?)(\xa0|\s)$', t):
            if idx + 1 < len(nodes):
                nxt = nodes[idx + 1].text or ''
                if nxt in num2roman:
                    nodes[idx + 1].text = num2roman[nxt]
                    fixed += 1
print('table refs fixed:', fixed)

# ---------------- 2. abstract double percent ----------------
for p in body.findall('.//w:p', NS):
    joined = ''.join(n.text or '' for n in p.findall('.//w:t', NS))
    if '92.9%' in joined and 'Retrained' in joined:
        cnt = 0
        for node in list(p.findall('.//w:t', NS)):
            if (node.text or '').strip() == '%':
                # remove the run containing this node
                r = node.getparent()
                r.getparent().remove(r)
                cnt += 1
        print('abstract % runs removed:', cnt)

# ---------------- 3. bios: append the other three ----------------
BIOS_EXTRA = [
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
bio_paras = []
for p in body.findall('.//w:p', NS):
    joined = ''.join(n.text or '' for n in p.findall('.//w:t', NS))
    if joined.startswith('Zhenyu Liu') and 'is currently pursuing' in joined:
        bio_paras.append(p)
assert len(bio_paras) == 1, 'expected one Zhenyu bio paragraph, got %d' % len(bio_paras)
zp = bio_paras[0]
prev = zp
for name, rest in BIOS_EXTRA:
    np = etree.Element(W + 'p')
    # copy pPr from Zhenyu's paragraph if present
    pPr = zp.find('./w:pPr', NS)
    if pPr is not None:
        np.append(etree.fromstring(etree.tostring(pPr)))
    r1 = etree.SubElement(np, W + 'r')
    rPr = etree.SubElement(r1, W + 'rPr'); etree.SubElement(rPr, W + 'b')
    t1 = etree.SubElement(r1, W + 't'); t1.text = name
    r2 = etree.SubElement(np, W + 'r')
    t2 = etree.SubElement(r2, W + 't'); t2.set(XMLSP, 'preserve'); t2.text = rest
    prev.addnext(np)
    prev = np
    print('bio appended:', name)

# ---------------- 4. bare +/- math -> plain spaced text ----------------
for om in list(body.findall('.//m:oMath', NS)):
    s = ''.join(t.text or '' for t in om.findall('.//m:t', NS))
    if s.strip() == '±':
        r = etree.Element(W + 'r')
        t = etree.SubElement(r, W + 't'); t.set(XMLSP, 'preserve')
        t.text = ' ± '
        om.addnext(r)
        om.getparent().remove(om)
        print('bare +/- normalized')

# ---------------- 5. save ----------------
tmp = DOCX + '.tmp'
RUNHEAD = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
 '<w:p><w:pPr><w:jc w:val="center"/><w:rPr><w:i/></w:rPr></w:pPr>'
 '<w:fldSimple w:instr=" PAGE "><w:r><w:rPr><w:i/></w:rPr><w:t>7</w:t></w:r></w:fldSimple>'
 '<w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">  |  </w:t></w:r>'
 '<w:r><w:rPr><w:i/></w:rPr><w:t>LIU et al.: A PHYSICS-ENCODED NEURAL OPERATOR WITH CONFORMAL '
 'TRAJECTORY BANDS</w:t></w:r>'
 '</w:p></w:hdr>')
with zipfile.ZipFile(DOCX) as zr, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zw:
    for item in zr.namelist():
        data = zr.read(item)
        if item == 'word/document.xml':
            data = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
        elif item == 'word/header2.xml':
            data = RUNHEAD.encode('utf-8')
        zw.writestr(item, data)
shutil.move(tmp, DOCX)
print('SAVED')

# ---------------- verification scan ----------------
z = zipfile.ZipFile(DOCX)
doc2 = z.read('word/document.xml').decode('utf-8')
def txt(x): return re.sub(r'<[^>]+>', ' ', x)
print('--- remaining issues ---')
print('Table N left:', re.findall(r'Table[s]?\s*\d', txt(doc2))[:10])
print('eq labels left:', re.findall(r'\[eq:[a-z0-9]+\]', txt(doc2)))
print('PLACEHOLDER left:', txt(doc2).count('PLACEHOLDER'))
print('92.9%% left:', '92.9% %' in doc2 or '92.9%  %' in doc2)
print('AUTHOR et al left:', 'AUTHOR et al' in z.read('word/header2.xml').decode('utf-8'))
print('VOL. XX left:', 'VOL. XX' in z.read('word/header1.xml').decode('utf-8') or 'VOL. XX' in z.read('word/header3.xml').decode('utf-8'))
print('Z. Pitt left:', 'Z. Pitt' in doc2)
print('0.1700 left:', '0.1700' in doc2)
print('bios:', txt(doc2)[txt(doc2).find('Author biographies'):txt(doc2).find('Author biographies')+380].replace('  ',' '))
