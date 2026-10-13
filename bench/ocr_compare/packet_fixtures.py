"""Render Packet's builtin eval fixtures (packet.eval.fixtures) to PNG at 150 dpi.
Gold text is the PDF text layer (blank scan pages stay empty) for score.py."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root; testset/ and src/ live here
sys.path.insert(0, str(ROOT / "src"))
import pypdfium2 as pdfium
from packet.eval.fixtures import build_builtin
out = ROOT / "testset" / "packet"; out.mkdir(parents=True, exist_ok=True)
man = []
for fx in build_builtin(out / ".src"):
    pdf = pdfium.PdfDocument(str(fx.pdf))
    for i in range(len(pdf)):
        name = f"pk-{fx.name}-{i}"; page = pdf[i]
        page.render(scale=150/72).to_pil().convert("RGB").save(out / f"{name}.png")
        tp = page.get_textpage()
        text = ((tp.get_text_bounded() if tp else "") or "").replace("\r\n", "\n").replace("\r", "\n")
        (out / f"{name}.json").write_text(json.dumps({"name": name, "full_text": text}, indent=1))
        man.append({"name": name, "fixture": fx.name, "page": i, "condition": fx.condition, "gold_docs": [(d.page_start, d.page_end) for d in fx.gold.documents]})
(out / "manifest.json").write_text(json.dumps(man, indent=1)); print(len(man))
