"""GMA-216: whole-document joined DocLayNet packets (downloaded, never vendored)."""
import hashlib
from pathlib import Path

import pytest

from packet.eval import wholedoc
from packet.eval.pdfutil import write_text_pdf


def test_manifest_records_license_and_pins_every_doc():
    m = wholedoc.load_manifest()
    assert m.name == "packet-dlnet-whole"
    assert "CDLA-Permissive-1.0" in m.license_note
    assert len(m.docs) >= 20
    for d in m.docs:
        assert d.url.startswith("https://") and d.license and len(d.sha256) == 64 and d.pages >= 2


def _fake(tmp: Path, n_docs: int = 6) -> wholedoc.Manifest:
    docs = []
    for i in range(n_docs):
        pages = 2 + i % 3
        p = write_text_pdf(tmp / f"d{i}.pdf", [[f"Doc {i} page {k}"] for k in range(pages)])
        docs.append(wholedoc.WholeDoc(name=f"d{i}", url=f"https://example.invalid/d{i}.pdf",
                                      file=p.name, pages=pages, license="test",
                                      sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return wholedoc.Manifest(name="t", version=1, seed=7, docs_per_packet=(2, 3),
                             license_note="CDLA-Permissive-1.0 test", docs=docs)


def test_plan_is_deterministic_and_uses_each_doc_once(tmp_path):
    m = _fake(tmp_path)
    a, b = wholedoc.plan(m), wholedoc.plan(m)
    assert a == b
    used = [i for packet in a for i in packet]
    assert sorted(used) == list(range(len(m.docs)))
    assert all(len(p) >= 2 for p in a)


def test_build_joins_whole_docs_with_gold_boundaries(tmp_path):
    m = _fake(tmp_path)
    fixtures = wholedoc.build_fixtures(m, cache=tmp_path, out=tmp_path / "out")
    for fx, idxs in zip(fixtures, wholedoc.plan(m)):
        sizes = [m.docs[i].pages for i in idxs]
        ends = [sum(sizes[: k + 1]) - 1 for k in range(len(sizes))]
        assert [d.page_end for d in fx.gold.documents] == ends
        assert fx.gold.origin.page_count == sum(sizes)
        assert fx.pdf.exists()


def test_fetch_rejects_checksum_mismatch(tmp_path):
    m = _fake(tmp_path, 2)
    m.docs[0].sha256 = "0" * 64
    with pytest.raises(wholedoc.FetchError):
        wholedoc.fetch(m, cache=tmp_path, opener=lambda url: b"", pause=0)


def test_fetch_downloads_missing_and_verifies(tmp_path):
    m = _fake(tmp_path, 2)
    blobs = {d.url: (tmp_path / d.file).read_bytes() for d in m.docs}
    cache = tmp_path / "cache"
    paths = wholedoc.fetch(m, cache=cache, opener=lambda url: blobs[url], pause=0)
    assert all(p.exists() and p.parent == cache for p in paths)


def test_cli_eval_dlnet_whole_uses_set_floors(tmp_path, monkeypatch, capsys):
    from packet import cli

    m = _fake(tmp_path)
    monkeypatch.setattr(wholedoc, "load_manifest", lambda path=None: m)
    rc = cli.main(["eval", "--set", "dlnet-whole", "--data-dir", str(tmp_path), "--json"])
    import json
    out = json.loads(capsys.readouterr().out)
    assert out["set"] == "dlnet-whole"
    assert out["gate"]["min_f1"] == 0.0 and out["gate"]["min_macro_f1"] == 0.6
    assert len(out["fixtures"]) == len(wholedoc.plan(m))
    assert rc == (0 if out["gate"]["passed"] else 1)
