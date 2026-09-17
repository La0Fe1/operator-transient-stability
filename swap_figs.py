# -*- coding: utf-8 -*-
"""Swap Fig2/Fig3 images in the revised MPCE docx with the real-data versions."""
import io, sys, re, zipfile, shutil, struct
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DOCX = r'C:\Users\佬肥\Desktop\算子\A Physics-Encoded Neural Operator with Conformal Trajectory Bands for Fast Transient Stability Screening_Liu_Wang_Liu_Zeng_MPCE_revised.docx'
FIG2 = r'C:\Users\佬肥\Desktop\算子\figures\fig2_trajectory.png'
FIG3 = r'C:\Users\佬肥\Desktop\算子\figures\fig3_conformal.png'

zin = zipfile.ZipFile(DOCX)
doc = zin.read('word/document.xml').decode('utf-8')
rels = zin.read('word/_rels/document.xml.rels').decode('utf-8')

# drawings in document order -> r:embed ids
embeds = re.findall(r'<a:blip[^>]*r:embed="(rId\d+)"', doc)
print('drawings in doc order:', embeds)
relmap = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="media/([^"]+)"', rels))
print('media mapping:', {e: relmap.get(e) for e in embeds})

# sanity: first drawing is Fig1 architecture, second Fig2 trajectory, third Fig3 conformal
assert len(embeds) == 3, 'expected 3 drawings'
fig2_media = 'word/media/' + relmap[embeds[1]]
fig3_media = 'word/media/' + relmap[embeds[2]]

# check old dims
for mname in [fig2_media, fig3_media]:
    data = zin.read(mname)
    w, h = struct.unpack('>II', data[16:24])
    print(mname, 'old dims:', w, 'x', h)

new2 = open(FIG2, 'rb').read()
new3 = open(FIG3, 'rb').read()
for f in [FIG2, FIG3]:
    w, h = struct.unpack('>II', open(f, 'rb').read(26)[16:24])
    print(f, 'new dims:', w, 'x', h)

tmp = DOCX + '.tmp'
with zipfile.ZipFile(DOCX) as zr, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zw:
    for item in zr.namelist():
        data = zr.read(item)
        if item == fig2_media:
            data = new2
            print('replaced', item)
        elif item == fig3_media:
            data = new3
            print('replaced', item)
        zw.writestr(item, data)
shutil.move(tmp, DOCX)
print('SWAPPED:', DOCX)
