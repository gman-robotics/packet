"""GMA-270: PoliTax-Split eval set (manifest, gold adapter, report-only gate)."""
import hashlib
import json
from pathlib import Path

import pytest

from packet.eval import politax
from packet.eval.metrics import document_boundaries
from packet.eval.pdfutil import write_text_pdf


def test_manifest_lists_https_urls_and_pins_every_pdf():
    m = politax.load_manifest()
    assert m.name == "packet-politax"
    assert "CC0" in m.license_note or "CC0" in m.annotations.license
    assert len(m.docs) == 30
    for d in m.docs:
        assert d.url.startswith("https://") and len(d.sha256) == 64 and d.pages >= 1
    assert len(m.annotations.sha256) == 64 and m.annotations.url.startswith("https://")


def _fake_manifest(tmp: Path, n_docs: int = 2) -> politax.Manifest:
    docs = []
    for i in range(n_docs):
        pages = 2 + i
        p = write_text_pdf(tmp / f"p{i}.pdf", [[f"p{i} pg{k}"] for k in range(pages)])
        docs.append(
            politax.PoliTaxDoc(
                "trump",
                p.name,
                p.name,
                f"https://example.invalid/{p.name}",
                pages,
                hashlib.sha256(p.read_bytes()).hexdigest(),
                "test",
            )
        )
    ann = politax.AnnotationsRef(
        "https://example.invalid/annotations.jsonl",
        "annotations.jsonl",
        "0" * 64,
        "CC0-1.0",
    )
    return politax.Manifest("t", 1, "CC0 test", ann, docs)


def _write_ann(tmp: Path, m: politax.Manifest) -> Path:
    lines = []
    for i, d in enumerate(m.docs):
        lines.append(
            {
                "file_name": d.file_name,
                "page_count": d.pages,
                "sub_documents": [
                    {"type": "form", "start_page": 1, "end_page": d.pages, "identifier": f"doc-{i}"}
                ],
            }
        )
    path = tmp / "annotations.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    m.annotations.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def test_fetch_rejects_checksum_mismatch(tmp_path):
    m = _fake_manifest(tmp_path)
    _write_ann(tmp_path, m)
    m.docs[0].sha256 = "0" * 64
    with pytest.raises(politax.FetchError):
        politax.fetch(m, cache=tmp_path / "cache", opener=lambda url: b"", pause=0)


def test_fetch_downloads_and_verifies(tmp_path):
    m = _fake_manifest(tmp_path)
    ann_path = _write_ann(tmp_path, m)
    blobs = {d.url: (tmp_path / d.file).read_bytes() for d in m.docs}
    blobs[m.annotations.url] = ann_path.read_bytes()
    cache = tmp_path / "cache"
    politax.fetch(m, cache=cache, opener=lambda url: blobs[url], pause=0)
    assert (cache / m.annotations.file).exists()
    assert all((cache / d.file).exists() for d in m.docs)


def test_adapter_rejects_overlap():
    subs = [politax.SubDocument(1, 2), politax.SubDocument(2, 3)]
    with pytest.raises(politax.GoldAdapterError, match="overlap"):
        politax.validate_sub_documents(subs, 3)


def test_adapter_rejects_gap():
    subs = [politax.SubDocument(1, 1), politax.SubDocument(3, 3)]
    with pytest.raises(politax.GoldAdapterError, match="gap"):
        politax.validate_sub_documents(subs, 3)


def test_adapter_rejects_incomplete_coverage():
    subs = [politax.SubDocument(1, 2)]
    with pytest.raises(politax.GoldAdapterError, match="expected 3"):
        politax.validate_sub_documents(subs, 3)


def test_gold_maps_one_indexed_pages_to_zero_indexed_and_last_end_not_boundary():
    ann = politax.FileAnnotation(
        "x.pdf",
        5,
        [
            politax.SubDocument(1, 2, "a"),
            politax.SubDocument(3, 5, "b"),
        ],
    )
    docs = politax.logical_documents_from_gold("x.pdf", ann, fixture_id="fx")
    assert [(d.page_start, d.page_end) for d in docs] == [(0, 1), (2, 4)]
    bounds = document_boundaries(docs)
    assert bounds == [1]
    assert 4 not in bounds


def test_build_fixtures_pairs_pdf_with_gold(tmp_path):
    m = _fake_manifest(tmp_path, 3)
    ann_path = _write_ann(tmp_path, m)
    cache = tmp_path / "cache"
    cache.mkdir()
    for d in m.docs:
        (cache / d.file).write_bytes((tmp_path / d.file).read_bytes())
    annotations = politax.load_annotations(ann_path)
    fixtures = politax.build_fixtures(m, cache, annotations)
    assert len(fixtures) == 3
    for fx, d in zip(fixtures, m.docs):
        assert fx.pdf.exists()
        assert fx.gold.origin.page_count == d.pages
        assert fx.gold.document_count() == 1


def test_cli_eval_politax_is_report_only(tmp_path, monkeypatch, capsys):
    from packet import cli

    m = _fake_manifest(tmp_path, 2)
    ann_path = _write_ann(tmp_path, m)
    cache = tmp_path / "cache"
    cache.mkdir()
    for d in m.docs:
        (cache / d.file).write_bytes((tmp_path / d.file).read_bytes())
    blobs = {d.url: (cache / d.file).read_bytes() for d in m.docs}
    blobs[m.annotations.url] = ann_path.read_bytes()

    def fake_load(path=None):
        return m

    real_fetch = politax.fetch

    def fake_fetch(manifest, cache, opener=politax._download, pause=0.5):
        return real_fetch(manifest, cache, opener=lambda url: blobs[url], pause=0)

    monkeypatch.setattr(politax, "load_manifest", fake_load)
    monkeypatch.setattr(politax, "fetch", fake_fetch)
    rc = cli.main(["eval", "--set", "politax", "--data-dir", str(cache), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["set"] == "politax"
    assert out["gate"]["min_f1"] == 0.0 and out["gate"]["min_macro_f1"] == 0.0
    assert out["gate"]["passed"] is True
    assert "micro_boundary_f1" in out
    assert rc == 0
