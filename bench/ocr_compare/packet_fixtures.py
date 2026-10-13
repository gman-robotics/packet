"""Render Packet's builtin eval fixtures (packet.eval.fixtures) to PNG at 150 dpi."""
import sys, json; sys.path.insert(0, "packet/src")
from pathlib import Path
import pypdfium2 as pdfium
from packet.eval.fixtures import build_builtin
out = Path("testset/packet"); out.mkdir(parents=True, exist_ok=True)
man = []
for fx in build_builtin(Path("testset/packet/.src")):
    pdf = pdfium.PdfDocument(str(fx.pdf))
    for i in range(len(pdf)):
        name = f"pk-{fx.name}-{i}"; pdf[i].render(scale=150/72).to_pil().convert("RGB").save(out / f"{name}.png")
        man.append({"name": name, "fixture": fx.name, "page": i, "condition": fx.condition, "gold_docs": [(d.page_start, d.page_end) for d in fx.gold.documents]})
(out / "manifest.json").write_text(json.dumps(man, indent=1)); print(len(man))
