"""GMA-207 fixed OCR test set. Deterministic (seed 20261012). Writes PNG pages + gold JSON."""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
R = random.Random(20261012)
W, H = 1275, 1650  # US letter @150dpi
def font(sz, bold=False):
    for p in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else []) + ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()
F, FB, FS, FT = font(26), font(40, True), font(20), font(24)

BODY = [
 "The tenant shall pay rent on the first business day of each month to the address listed above.",
 "Late payments accrue a fee of 5% of the outstanding balance after a grace period of ten days.",
 "Either party may terminate this agreement with sixty days written notice delivered by certified mail.",
 "All maintenance requests must be submitted through the owner portal and will be answered within 48 hours.",
 "The security deposit of $2,400.00 will be returned within thirty days of move-out, less documented damages.",
 "Invoice totals include applicable sales tax at the rate in effect on the date of shipment.",
 "Questions about this statement should be directed to accounts@northwind.example or (303) 555-0142.",
 "Section 4.2 supersedes any prior oral or written understanding between the parties on this subject.",
]

def wrap(d, text, f, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > maxw and cur: lines.append(cur); cur = w
        else: cur = t
    lines.append(cur); return lines

def page(header, title, paras, footer, pnum, table=None):
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im)
    gold = {"header": header, "footer": footer, "page_number": pnum, "title": title, "paragraphs": [], "table": table}
    blocks = []
    if header:
        d.text((100, 50), header, font=FS, fill=0); blocks.append(("header", header, [100,50,100+d.textlength(header,font=FS),72]))
    y = 130
    if title:
        d.text((100, y), title, font=FB, fill=0); blocks.append(("title", title, [100,y,100+d.textlength(title,font=FB),y+44])); y += 80
    for p in paras:
        ls = wrap(d, p, F, W-200); y0 = y
        for l in ls: d.text((100, y), l, font=F, fill=0); y += 36
        blocks.append(("paragraph", p, [100,y0,W-100,y])); gold["paragraphs"].append(p); y += 24
    if table:
        cw = (W-200)//len(table[0]); y0 = y
        for r, row in enumerate(table):
            for c, cell in enumerate(row):
                x = 100 + c*cw; d.rectangle([x, y, x+cw, y+44], outline=0, width=2)
                d.text((x+10, y+10), cell, font=FT if r else font(24, True), fill=0)
            y += 44
        blocks.append(("table", "\n".join(" ".join(c for c in r if c) for r in table), [100,y0,100+cw*len(table[0]),y])); y += 30
    if footer:
        d.text((100, H-80), footer, font=FS, fill=0); blocks.append(("footer", footer, [100,H-80,100+d.textlength(footer,font=FS),H-58]))
    if pnum:
        tw = d.textlength(pnum, font=FS); d.text((W-100-tw, H-80), pnum, font=FS, fill=0); blocks.append(("page_number", pnum, [W-100-tw,H-80,W-100,H-58]))
    gold["blocks"] = [{"kind": k, "text": t, "bbox_px": b} for k, t, b in blocks]
    gold["full_text"] = "\n".join(t for k, t, b in blocks if t)
    return im, gold

def scanify(im, seed):
    rr = random.Random(seed)
    im = im.rotate(rr.uniform(-1.5, 1.5), resample=Image.BICUBIC, fillcolor=245, expand=False)
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(im).astype(np.float32)
    a = a*0.85 + 25 + np.random.default_rng(seed).normal(0, 12, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

pages, manifest = [], []
def add(name, im, gold, kind, doc=None, scan=False):
    if scan: im = scanify(im, len(manifest)+7)
    p = OUT / f"{name}.png"; im.convert("RGB").save(p)
    gold.update(name=name, kind=kind, scan=scan, doc=doc); (OUT / f"{name}.json").write_text(json.dumps(gold, indent=1))
    manifest.append({"name": name, "kind": kind, "scan": scan, "doc": doc}); pages.append(im.convert("RGB"))

T1 = [["Item", "Qty", "Unit price", "Amount"], ["Widget A", "4", "$12.50", "$50.00"], ["Gadget B", "2", "$7.25", "$14.50"], ["Service fee", "1", "$30.00", "$30.00"], ["Total", "", "", "$94.50"]]
T2 = [["Unit", "Bedrooms", "Rent", "Status"], ["101", "2", "$1,850", "Occupied"], ["102", "1", "$1,420", "Vacant"], ["201", "3", "$2,300", "Occupied"], ["202", "2", "$1,900", "Notice"]]
# Multi-doc packet: doc0 invoice 2pp, doc1 lease 3pp, doc2 statement 1pp. Page numbers restart; furniture differs.
add("md-00", *page("Northwind Traders | Invoice #4471", "Northwind Invoice", BODY[5:7], "Northwind Traders, 10 Market St, Denver CO", "Page 1 of 2", T1), "multidoc", doc=0, scan=True)
add("md-01", *page("Northwind Traders | Invoice #4471", None, [BODY[1], "Thank you for your business."], "Northwind Traders, 10 Market St, Denver CO", "Page 2 of 2"), "multidoc", doc=0, scan=True)
add("md-02", *page("Contoso Property Lease", "Residential Lease Agreement", BODY[0:3], "Contoso Leasing LLC - Confidential", "1", None), "multidoc", doc=1)
add("md-03", *page("Contoso Property Lease", "Section 4. Maintenance", [BODY[3], BODY[7]], "Contoso Leasing LLC - Confidential", "2", T2), "multidoc", doc=1, scan=True)
add("md-04", *page("Contoso Property Lease", None, [BODY[4], "Signatures follow on the next page."], "Contoso Leasing LLC - Confidential", "3"), "multidoc", doc=1)
add("md-05", *page("Fabrikam Bank", "Monthly Account Statement", [BODY[6], BODY[5]], "Member FDIC", "Page 1 of 1", [["Date", "Description", "Amount"], ["10/01", "Opening balance", "$5,210.33"], ["10/05", "Rent deposit", "$1,850.00"], ["10/31", "Closing balance", "$7,060.33"]]), "multidoc", doc=2, scan=True)
pages[0].save(OUT / "multidoc.pdf", save_all=True, append_images=pages[1:6], resolution=150)
# hard-scan variants: 100 dpi-equivalent downsample, 3 deg skew, heavy noise, JPEG q=30
import io
def hard(im, seed):
    rr = random.Random(seed); im = im.rotate(rr.choice([-3, 3]), resample=Image.BICUBIC, fillcolor=235)
    im = im.resize((W*2//3, H*2//3), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.0))
    a = np.asarray(im).astype(np.float32)*0.8 + 30 + np.random.default_rng(seed).normal(0, 22, (H*2//3, W*2//3))
    b = io.BytesIO(); Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(b, "JPEG", quality=30); return Image.open(io.BytesIO(b.getvalue()))
for src, (args, nm) in {"md-00": ((None,), "hs-00"), "md-03": ((None,), "hs-03"), "md-05": ((None,), "hs-05")}.items():
    pass
HARD = [("hs-00", page("Northwind Traders | Invoice #4471", "Northwind Invoice", BODY[5:7], "Northwind Traders, 10 Market St, Denver CO", "Page 1 of 2", T1)),
        ("hs-03", page("Contoso Property Lease", "Section 4. Maintenance", [BODY[3], BODY[7]], "Contoso Leasing LLC - Confidential", "2", T2)),
        ("hs-05", page("Fabrikam Bank", "Monthly Account Statement", [BODY[6], BODY[5]], "Member FDIC", "Page 1 of 1", None))]
for i, (nm, (im, g)) in enumerate(HARD):
    p = OUT / f"{nm}.png"; hard(im, 100+i).convert("RGB").save(p)
    g.update(name=nm, kind="hard-scan", scan=True, doc=None); (OUT / f"{nm}.json").write_text(json.dumps(g, indent=1)); manifest.append({"name": nm, "kind": "hard-scan", "scan": True})
(OUT / "manifest.json").write_text(json.dumps({"seed": 20261012, "dpi": 150, "pages": manifest, "multidoc_boundaries": [0, 2, 5]}, indent=1))
print(len(manifest), "pages")
