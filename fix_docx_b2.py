# -*- coding: utf-8 -*-
"""Robust paragraph-level replacements for the docx (math-preserving) + save."""
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

def para_plain(p):
    """Paragraph text with oMath blocks replaced by \x00i\x00 tokens."""
    xml = etree.tostring(p).decode('utf-8')
    maths = re.findall(r'<m:oMathPara>.*?</m:oMathPara>|<m:oMath>.*?</m:oMath>', xml, re.S)
    for i, mblk in enumerate(maths):
        xml = xml.replace(mblk, '\x00M%d\x00' % i)
    xml = re.sub(r'<w:br[^>]*/>', ' ', xml)
    text = re.sub(r'<w:[^>]*>', '', xml)
    text = re.sub(r'</w:[^>]*>', '', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text, maths

def para_apply(p, old, new):
    text, maths = para_plain(p)
    if old not in text:
        return False
    text = text.replace(old, new, 1)
    # rebuild paragraph: keep pPr, then runs split by math tokens
    pPr = p.find('./w:pPr', NS)
    for child in list(p):
        p.remove(child)
    if pPr is not None:
        p.append(pPr)
    for seg in re.split(r'\x00(M\d+)\x00', text):
        if seg == '':
            continue
        if seg.startswith('M') and seg[1:].isdigit() and int(seg[1:]) < len(maths):
            wrap = '<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">' + maths[int(seg[1:])] + '</w:r>'
            blk = etree.fromstring(wrap)[0]
            p.append(blk)
        else:
            r = etree.SubElement(p, W + 'r')
            t = etree.SubElement(r, W + 't')
            t.set(XMLSP, 'preserve')
            t.text = seg
    return True

reps = [
 ('estimates the CCT to within 25.2 ms, and runs',
  'attains a mean absolute CCT error of 25.0 ms, and runs'),
 ('estimates the CCT to within 25.2 ms against an event-aligned reference',
  'attains a mean absolute CCT error of 25.0 ms against an event-aligned reference'),
 ('estimates the CCT to within 25.2 ms on average (max',
  'attains a mean absolute CCT error of 25.0 ms (max'),
 ('bounds the machine-angle spread to 148',
  'bounds the machine-angle spread error to 148'),
 ('directly calibrated pairwise bound (148',
  'directly calibrated pairwise-error bound (148'),
 ('bounds the pairwise difference to 148',
  'bounds the pairwise-difference prediction error to 148'),
 ('bounds the pairwise angle difference—the exact quantity in the stability criterion—to 148',
  'bounds the prediction error of the pairwise angle difference—the exact quantity in the stability criterion—to 148'),
 ('couples it with conformal certification',
  'couples it with calibrated trajectory uncertainty quantification'),
 ('before a small multilayer perceptron',
  'alongside the normalized raw time, before a small multilayer perceptron'),
 ('the event-aligned TDS binary search',
  'the event-aligned TDS first-crossing search'),
 ('where the binary search lands between multiple local stable-to-unstable crossings of the operator’s criterion.',
  'where the criterion has multiple local stable-to-unstable crossings; the first-crossing search of Section 4 reproduces the previous binary-search CCTs within 3 ms per bus.'),
 ('where the binary search lands between multiple local stable-to-unstable crossings of the operator\u2019s criterion.',
  'where the criterion has multiple local stable-to-unstable crossings; the first-crossing search of Section 4 reproduces the previous binary-search CCTs within 3 ms per bus.'),
 ('against the target level α.',
  'against the target coverage level 1\u2212α.'),
 ('the last conditions on ground-truth labels and is not a deployment guarantee, since conditioning on the prediction breaks exchangeability',
  'the last is an oracle-conditioned diagnostic over the truly stable population, not a deployment guarantee'),
 ('(rising to 31% at α=0.3 and 67% at α=0.5 as',
  '(rising to 31% at α=0.3, 57% at α=0.4, and 67% at α=0.5 as'),
 ('No per-machine bus mapping and no per-unit base conversion are applied, so this is a synthetic dynamic case',
  'No per-machine bus mapping and no per-unit base conversion are applied (H and x′d are interpreted directly on the system MVA base, a self-consistent convention); this is a synthetic dynamic case'),
 ('they show the severity encoding carries more information than raw observables. A severity-MLP',
  'they show the severity encoding carries more information than raw observables. Note that the pointwise MLP’s higher classification accuracy does not carry over to trajectory quality (43.1° versus 36.3° pooled RMSE, without continuous-time evaluation; Table II). A severity-MLP'),
 ('We train a single operator (p=512) on a mix of N−0 and N−1 scenarios (9990 scenarios, line trips drawn from 25 training lines) using the extended encoding of Section 4, and evaluate on two held-out axes: unseen fault buses and unseen trip lines (10 lines).',
  'We train the multi-topology operator (p=512) with the extended encoding of Section 4. The N−1 experiment trains on 9990 N−0/N−1 scenarios (line trips drawn from 25 training lines) and evaluates on two held-out axes: unseen fault buses and 10 unseen trip lines. The N−2 experiment trains on 8748 scenarios that additionally include N−2 line-pair scenarios drawn from the training lines, and evaluates on held-out line pairs.'),
 ('The same encoding extends to N−2 contingencies with no architecture change: on held-out line pairs',
  'The same encoding and architecture apply to N−2 contingencies (two lines tripped) without modification: on held-out line pairs'),
 ('Generalization of the multi-topology operator (N−0 + N−1 training):',
  'Generalization of the multi-topology operator (training mixes include N−0/N−1/N−2):'),
 ('a single operator generalizes to N−1 and N−2 line-trip topologies not seen in training.',
  'the operator generalizes to held-out N−1 line trips and N−2 line pairs.'),
 ('to unseen fault locations and to N−1 and N−2 line-trip topologies.',
  'to unseen fault locations and to held-out N−1/N−2 line-trip topologies.'),
 ('Given the operator, the CCT of a fault is estimated by binary search over the clearing time: for each candidate tc, the operator predicts the trajectory, and the scenario is declared unstable if the maximum pairwise angle spread exceeds π at any output-grid point—the same max-over-time criterion used for the reference labels. The search uses 16 iterations over [0.01,0.8] s ([0.005,0.2] s for the 118-bus system); the same protocol is applied to the TDS reference, so the two CCT estimates are directly comparable. The 12 μs interval is the search tolerance of the bisection, not the physical accuracy of the CCT; the reference itself uses the event-aligned RK4 scheme of Section 3, whose convergence was verified at a 0.25 ms step, and both judge stability on the 0.01 s output grid.',
  'Given the operator, the CCT of a fault is the first clearing time at which the criterion is violated. Because neither the operator’s criterion nor the reference is guaranteed monotone in tc, a 5 ms sweep over [0.01,0.8] s locates the first stable-to-unstable sign change, which is then refined by local bisection to 0.1 ms; the same first-crossing protocol (with the system-dependent range [0.005,0.2] s) is applied to the TDS reference, so the two CCT estimates are directly comparable. The refinement tolerance is the search resolution, not the physical CCT accuracy; the reference uses the event-aligned RK4 scheme of Section 3 (convergence verified at a 0.25 ms step), and both judge stability on the 0.01 s grid.'),
 ('These numbers use the base p=128 architecture (Table II reports p=512), which explains the different full-data values.',
  ''),
 ('The operator is therefore only moderately robust to feature noise in its input encoding; correlated or biased parameter-estimation errors and conformal coverage under input noise remain to be tested.',
  'The operator is therefore only moderately robust to feature noise; correlated or biased parameter errors and conformal coverage under noise remain to be tested.'),
 ('Adding uniform damping D=1 leaves the severity encoding unchanged—it is evaluated at the pre-fault equilibrium, where the damping term vanishes—so the same architecture is retrained on damped data.',
  'Adding uniform damping D=1 leaves the encoding unchanged (the damping term vanishes at the pre-fault equilibrium), so the same architecture is retrained on damped data.'),
 ('a stronger graph baseline with three residual layers and concatenated mean/max/sum pooling attains 77.3%±3.9% (five seeds), so the gap is not an artifact of a weak graph architecture',
  'a stronger graph baseline with three residual layers attains 77.3%±3.9% (five seeds), so the gap is not an artifact of a weak architecture'),
 ('Alternative scores (amplitude-normalized 2.3%, first-swing 70.8°, time-averaged 47.7°; see the repository), adaptive/locally adaptive conformal methods [24], [25], and conformalized quantile regression [26] are candidates for reducing near-boundary conservatism and are left as future work.',
  'Alternative scores (amplitude-normalized, first-swing, time-averaged; see the repository), adaptive conformal methods [24], [25], and conformalized quantile regression [26] are candidates for reducing near-boundary conservatism.'),
]

done, skipped = 0, []
for old, new in reps:
    hit = False
    for p in body.findall('.//w:p', NS):
        if para_apply(p, old, new):
            done += 1
            hit = True
            break
    if not hit:
        skipped.append(old[:55])
print('paragraph replacements:', done)
for s in skipped:
    print('SKIP:', s)

# ---------------- remove '&' from the two split swing equations ----------------
removed = 0
for om in body.findall('.//m:oMath', NS):
    txt = ''.join(t.text or '' for t in om.findall('.//m:t', NS))
    if '&' in txt and txt.startswith('dδ'):
        for r in list(om.findall('.//m:r', NS)):
            rt = ''.join(t.text or '' for t in r.findall('.//m:t', NS))
            if rt.strip() == '&':
                r.getparent().remove(r)
                removed += 1
print('& removed from equations:', removed)

# ---------------- save ----------------
tmp = DOCX + '.tmp'
with zipfile.ZipFile(DOCX) as zr, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zw:
    for item in zr.namelist():
        data = zr.read(item)
        if item == 'word/document.xml':
            data = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
        zw.writestr(item, data)
shutil.move(tmp, DOCX)
print('SAVED')
