from pathlib import Path

from packet.eval.doclaynet import DocRecord, materialize_placeholder_packet, plan_packets
from packet.eval.fixtures import build_builtin
from packet.eval.metrics import boundary_score
from packet.eval.pdfutil import write_text_pdf
from packet.pipeline import Pipeline
from packet.source import PdfPageSource


def test_builtin_pdfs_are_readable(tmp_path: Path):
    fixtures = build_builtin(tmp_path)
    names = {fx.name for fx in fixtures}
    assert names == {"text-layer", "multi-doc", "shared-header", "scan"}
    for fx in fixtures:
        with PdfPageSource(fx.pdf) as src:
            assert src.page_count == fx.gold.origin.page_count


def test_text_pdf_extracts(tmp_path: Path):
    pdf = write_text_pdf(tmp_path / "one.pdf", [["Hello Packet", "Page 1 of 1"]])
    with PdfPageSource(pdf) as src:
        refs = list(src.pages())
        assert len(refs) == 1
        text = src.text_for(0)
        assert "Hello Packet" in text


def test_pipeline_on_text_layer(tmp_path: Path):
    fx = next(f for f in build_builtin(tmp_path) if f.name == "text-layer")
    pred = Pipeline().parse(fx.pdf)
    assert pred.origin.page_count == 3
    score = boundary_score(fx.gold, pred)
    assert score.recall == 1.0


def test_doclaynet_planner_placeholder(tmp_path: Path):
    records = [
        DocRecord("a.pdf", "val", "patents", page_paths=[Path("1"), Path("2")], page_nos=[1, 2]),
        DocRecord("b.pdf", "val", "manuals", page_paths=[Path("3")], page_nos=[1]),
        DocRecord("c.pdf", "val", "patents", page_paths=[Path("4"), Path("5")], page_nos=[1, 2]),
        DocRecord("d.pdf", "val", "laws_and_regulations", page_paths=[Path("6")], page_nos=[1]),
    ]
    specs = plan_packets(records, slice_name="poly-seq", split="val", packet_count=2)
    assert specs
    pdf, gold = materialize_placeholder_packet(specs[0], tmp_path / "synth")
    assert pdf.exists()
    assert gold.document_count() == len(specs[0].documents)
    with PdfPageSource(pdf) as src:
        assert src.page_count == gold.origin.page_count
