"""GMA-446: pure helpers in the Jev eval scripts. No model load, no ~/.cache."""
import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("PIL")

from PIL import Image, ImageDraw

# Mutation workers put the mutated tree first on PYTHONPATH; prefer it over the checkout.
_REPO = Path(__file__).resolve().parents[1]
for _root in reversed([*map(Path, filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))), _REPO]):
    if (_root / "bench" / "model_eval").is_dir():
        sys.path.insert(0, str(_root / "bench" / "model_eval"))
jev_score = importlib.import_module("jev_score")
jev_a2 = importlib.import_module("jev_a2")


def test_tag_groups_and_regions_keep_negative_coords():
    raw = "![text+](-12,3,40,-5)\nhello\n![title](0,1,2,3)\nT"
    match = jev_score.TAG.search(raw)
    assert match.groups() == ("text", "-12", "3", "40", "-5")
    assert jev_score.regions(raw) == [
        ("text", (-12, 3, 40, -5), "hello"),
        ("title", (0, 1, 2, 3), "T"),
    ]
    assert jev_score.page_text(raw) == "hello\n\nT"


def test_drop_indexes_negative_coord_blocks_like_regions():
    raw = "pre\n![text](-10,0,10,10)\nkeep\n![text](0,-4,8,8)\ndrop"
    assert [bb for _, bb, _ in jev_score.regions(raw)] == [(-10, 0, 10, 10), (0, -4, 8, 8)]
    text, n = jev_a2._drop(raw, {0: 0.9, 1: 0.1}, 0.5)
    assert n == 1
    assert text.startswith("pre") and "keep" in text and "drop" not in text
    assert jev_a2._drop("plain", {0: 0.0}, 0.5) == ("plain", 0)


def test_drop_keeps_blocks_without_a_prob():
    raw = "![text](0,0,1,1)\nkeep\n![image](2,2,3,3)\npic"
    assert jev_a2._drop(raw, {0: 0.5}, 0.5) == (raw, 0)


def test_norm_bbox_clamps_and_orders_corners():
    assert jev_score.norm_bbox((-20, 1200, 40, -5)) == (0, 0, 40, 1000)
    assert jev_score.norm_bbox((900, 800, 100, 50)) == (100, 50, 900, 800)
    assert jev_score.norm_bbox((1, 2, 3, 4)) == (1, 2, 3, 4)


def test_crop_swapped_corners_is_the_normalized_region():
    img = Image.new("RGB", (1000, 1000), "blue")
    ImageDraw.Draw(img).rectangle((200, 100, 799, 899), fill="red")
    cropped = jev_score.crop(img, (800, 900, 200, 100), pad=0)
    assert cropped.size == (600, 800)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)
    assert cropped.getpixel((599, 799)) == (255, 0, 0)


def test_crop_degenerate_grows_around_center_not_full_page():
    img = Image.new("RGB", (100, 100), "white")
    img.putpixel((50, 40), (1, 2, 3))
    cropped = jev_score.crop(img, (500, 400, 500, 400), pad=0)
    assert cropped.size == (1, 1)
    assert cropped.getpixel((0, 0)) == (1, 2, 3)
    assert cropped.size != img.size


def test_crop_clamped_point_stays_inside_the_image():
    img = Image.new("RGB", (100, 80), "white")
    img.putpixel((99, 0), (7, 8, 9))
    assert jev_score.norm_bbox((5000, -20, 9000, -1)) == (1000, 0, 1000, 0)
    cropped = jev_score.crop(img, (5000, -20, 9000, -1), pad=0)
    assert cropped.size == (1, 1)
    assert cropped.getpixel((0, 0)) == (7, 8, 9)


def test_crop_thin_and_negative_boxes_are_not_the_full_page():
    img = Image.new("RGB", (1000, 1000), "white")
    assert jev_score.crop(img, (100, 0, 102, 1000), pad=0).size == (2, 1000)
    assert jev_score.crop(img, (-100, -100, 100, 200), pad=0).size == (100, 200)
    assert jev_score.crop(img, (0, 0, 0, 0)).size == (6, 6)
    assert jev_score.crop(img, (1000, 1000, 1000, 1000)).size == (6, 6)
    assert jev_score.crop(img, (100, 100, 200, 200), pad=6).size == (112, 112)


@pytest.mark.parametrize(("width", "expect"), [(1, 400), (400, 400), (1000, 1000), (1400, 1400), (2000, 1400)])
def test_render_width(width, expect):
    assert jev_score.render_width(Image.new("RGB", (width, 10))) == expect


def test_resume_jsonl_good_file():
    raw = b'{"key":"a/p0.json"}\n{"key":"b/p1.json"}\n'
    assert jev_score.resume_jsonl(raw) == ({"a/p0.json", "b/p1.json"}, len(raw))


def test_resume_jsonl_truncated_last_line(tmp_path):
    good = '{"key":"café"}\n'.encode()
    raw = good + b'{"key":"b"'
    done, end = jev_score.resume_jsonl(raw)
    assert done == {"café"} and end == len(good)
    assert jev_score.resume_jsonl(b'{"key":"only-partial"') == (set(), 0)
    path = tmp_path / "out.jsonl"
    path.write_bytes(raw)
    with path.open("rb+") as fh:
        fh.truncate(end)
    assert path.read_bytes() == good


def test_resume_jsonl_malformed_last_line_keeps_prior():
    good = b'{"key":"a"}\n'
    assert jev_score.resume_jsonl(good + b"{bad}\n") == ({"a"}, len(good))


def test_resume_jsonl_malformed_middle_line_raises():
    with pytest.raises(json.JSONDecodeError):
        jev_score.resume_jsonl(b'{bad\n{"key":"a"}\n{"key":"trail')
    with pytest.raises(KeyError):
        jev_score.resume_jsonl(b'{"key":"a"}\n{}\n{"key":"c"}\n')


def test_accepted_cer_uses_scored_pages_and_counts_unscored():
    cer = {"a": 0.0, "b": 0.5, "c": 1.0, "d": 0.25}
    prob = {"a": 1.0, "b": 0.0, "c": None}
    assert jev_a2.accepted_cer(cer, prob, 0.5) == {
        "cer_all": 0.25, "cer_accepted": 0.0, "accepted": 1, "of": 2, "unscored": 2,
    }
    tied = jev_a2.accepted_cer({"a": 0.5}, {"a": 0.5}, 0.5)
    assert tied["accepted"] == 1 and tied["of"] == 1 and tied["unscored"] == 0
    empty = jev_a2.accepted_cer({"a": 0.2}, {"a": None}, 0.5)
    assert empty["cer_all"] is None and empty["cer_accepted"] is None
    assert empty["accepted"] == 0 and empty["of"] == 0 and empty["unscored"] == 1


def test_missing_score_count():
    keys = ["a/p0.json", "a/p1.json", "b/p0.json"]
    assert jev_a2.missing_score_count(keys, {"a/p0.json": {0: 0.5}}) == 2
    assert jev_a2.missing_score_count(keys, set(keys)) == 0
    assert jev_a2.missing_score_count([], {"a": 1}) == 0


# ---- orchestration paths, with a fake model and tmp dirs ---------------------------------------------------


@pytest.fixture
def fake_render(monkeypatch):
    """render_text needs a macOS font; tests only need the requested width."""
    monkeypatch.setattr(jev_score, "render_text", lambda text, width=1000, size=28: Image.new("RGB", (width, 10)))


class FakeJev:
    load_s = 1.5

    def __init__(self, probs=None):
        self.calls = []
        self.probs = list(probs or [])

    def score(self, original, render, text):
        self.calls.append((original.size, render.size, text))
        return self.probs.pop(0) if self.probs else 0.25


def test_score_page_rows(fake_render, monkeypatch):
    j = FakeJev([0.9, 0.4])
    img = Image.new("RGB", (1000, 1000), "white")
    raw = "![text](0,0,500,500)\nhello\n![image](0,0,10,10)\n![title](500,500,1000,1000)\nT"
    rows = jev_score.score_page(j, img, raw)
    assert [(r["kind"], r.get("i"), r["prob"]) for r in rows] == [
        ("block", 0, 0.9), ("block", 1, None), ("block", 2, 0.4), ("page", None, 0.25)]
    assert j.calls[0][0] == (506, 506) and j.calls[0][1][0] == 506
    assert j.calls[-1][2] == "hello\n\n\nT"
    long_rows = jev_score.score_page(FakeJev(), img, "x" * (jev_score.SKIP_PAGE_SCORE + 1))
    assert long_rows == [{"kind": "page", "prob": None, "chars": jev_score.SKIP_PAGE_SCORE + 1, "skipped": "too_long"}]
    assert jev_score.score_page(FakeJev(), img, raw, page_level=False)[-1]["kind"] == "block"


def test_parity_uses_render_width(fake_render, monkeypatch, tmp_path, capsys):
    case = tmp_path / "c1"; case.mkdir()
    Image.new("RGB", (200, 50)).save(case / "blk.png"); Image.new("RGB", (30, 30)).save(case / "ref.png")
    (case / "case.json").write_text(json.dumps({
        "case_id": "c1", "block_image": "blk.png", "observed_prob_yes": {"good": 0.8},
        "reference_render_of": "good", "reference_render": "ref.png",
        "candidates": [{"role": "good", "text": "a"}, {"role": "bad", "text": "b"}]}))
    fake = FakeJev([0.5, 0.6, 0.7])
    monkeypatch.setattr(jev_score, "Jev", lambda: fake)
    jev_score.parity(str(tmp_path))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "load_s 1.5"
    assert json.loads(lines[1]) == {"case": "c1", "role": "good", "observed": 0.8, "pil_render": 0.5, "ref_render": 0.6}
    assert json.loads(lines[2]) == {"case": "c1", "role": "bad", "observed": None, "pil_render": 0.7}
    assert fake.calls[0][1][0] == 400 and fake.calls[1][1] == (30, 30)


def test_testset_writes_jev_json(fake_render, monkeypatch, tmp_path, capsys):
    root = tmp_path / "root"
    for d in ("packet", "synth", "public"): (root / "testset" / d).mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(root / "testset" / "synth" / "pg1.png")
    (root / "testset" / "synth" / "notes.txt").write_text("x")
    run = tmp_path / "run"; run.mkdir()
    (run / "pg1.txt").write_text("hello"); (run / "other.txt").write_text("skip")
    monkeypatch.setattr(jev_score, "ROOT", root); monkeypatch.setattr(jev_score, "Jev", lambda: FakeJev([0.3]))
    jev_score.testset([str(run)])
    doc = json.loads((run / "_jev.json").read_text())
    assert list(doc["pages"]) == ["pg1"] and doc["pages"]["pg1"][0]["prob"] == 0.3 and doc["load_s"] == 1.5
    assert json.loads(capsys.readouterr().out)["scored"] == 1


class FakeSrc:
    opened: ClassVar[list] = []

    def __init__(self, pdf):
        FakeSrc.opened.append(pdf); self.closed = False

    def render(self, idx, longest_px):
        assert longest_px == 1540
        return Image.new("RGB", (100 + idx, 100))

    def close(self):
        self.closed = True


def _fake_packet(monkeypatch, docs, hold):
    m = types.SimpleNamespace(docs=docs)
    pt = types.ModuleType("packet.eval.politax"); pt.load_manifest = lambda: m
    split = types.ModuleType("packet.eval.politax_split"); split.holdout_file_names = lambda _m: hold
    src = types.ModuleType("packet.source"); src.PdfPageSource = FakeSrc
    ev = types.ModuleType("packet.eval"); ev.politax = pt
    monkeypatch.setitem(sys.modules, "packet.eval", ev)
    monkeypatch.setitem(sys.modules, "packet.eval.politax", pt)
    monkeypatch.setitem(sys.modules, "packet.eval.politax_split", split)
    monkeypatch.setitem(sys.modules, "packet.source", src)


def _ocr(data, sha, n):
    (data / "ocr" / sha).mkdir(parents=True)
    for i in range(n): (data / "ocr" / sha / f"p{i:05d}.json").write_text(json.dumps({"raw": f"page {i}"}))


def test_politax_scores_resumes_and_strides(fake_render, monkeypatch, tmp_path, capsys):
    data = tmp_path / "data"; _ocr(data, "s1", 4); _ocr(data, "s2", 1)
    (data / "a.pdf").write_text("pdf")
    docs = [types.SimpleNamespace(file_name="a.pdf", sha256="s1"), types.SimpleNamespace(file_name="b.pdf", sha256="s2")]
    _fake_packet(monkeypatch, docs, {"a.pdf"})
    monkeypatch.setattr(jev_score, "Jev", lambda: FakeJev())
    out = tmp_path / "o.jsonl"
    out.write_bytes(b'{"key":"s1/p00000.json"}\n{"key":"s1/p0')
    FakeSrc.opened = []
    jev_score.politax(str(data), str(out), stride=2, page_level=True)
    lines = out.read_text().splitlines()
    assert json.loads(lines[0]) == {"key": "s1/p00000.json"}
    rows = [json.loads(x) for x in lines[1:]]
    assert [(r["key"], r["file"], r["page"]) for r in rows] == [("s1/p00002.json", "a.pdf", 2)]
    assert rows[0]["rows"][-1] == {"kind": "page", "prob": 0.25, "chars": 6, "s": rows[0]["rows"][-1]["s"]}
    assert FakeSrc.opened == [data / "a.pdf"]
    assert json.loads(capsys.readouterr().out)["pages"] == 1
    FakeSrc.opened = []
    jev_score.politax(str(data), str(out), stride=2)
    assert FakeSrc.opened == [] and len(out.read_text().splitlines()) == 2


def test_politax_missing_pdf_exits(monkeypatch, tmp_path):
    data = tmp_path / "data"; _ocr(data, "s1", 1)
    _fake_packet(monkeypatch, [types.SimpleNamespace(file_name="a.pdf", sha256="s1")], {"a.pdf"})
    monkeypatch.setattr(jev_score, "Jev", lambda: FakeJev())
    with pytest.raises(SystemExit, match="missing politax PDF: .*a.pdf"):
        jev_score.politax(str(data), str(tmp_path / "o.jsonl"))


def _a2_dirs(monkeypatch, tmp_path):
    runs, a2 = tmp_path / "runs", tmp_path / "a2"
    monkeypatch.setattr(jev_a2, "RUNS", runs); monkeypatch.setattr(jev_a2, "A2", a2); monkeypatch.setattr(jev_a2, "P1", tmp_path)
    return runs, a2


def test_dev_labels_and_accepted_cer(monkeypatch, tmp_path):
    runs, a2 = _a2_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(jev_a2, "ENGINES", ["e1"])
    gold = {"p1": {"full_text": "hello world"}, "p2": {"full_text": "foo"}}
    monkeypatch.setattr(jev_a2.sc, "gold_for", gold.get)
    e = runs / "e1"; e.mkdir(parents=True)
    (e / "p1.txt").write_text("![text](0,0,1,1)\nhello\n![text](0,0,1,1)\nnope\n![text](0,0,1,1)\n")
    (e / "p2.txt").write_text("![text](0,0,1,1)\nfoo"); (e / "p3.txt").write_text("![text](0,0,1,1)\nx")
    pages = {"p1": [{"kind": "block", "i": 0, "prob": 0.9}, {"kind": "block", "i": 1, "prob": 0.2},
                    {"kind": "block", "i": 2, "prob": 0.5}, {"kind": "page", "prob": 0.8}],
             "p2": [{"kind": "page", "prob": 0.1}, {"kind": "block", "i": 0, "prob": None}],
             "p3": [{"kind": "block", "i": 0, "prob": 0.4}, {"kind": "page", "prob": None}]}
    (e / "_jev.json").write_text(json.dumps({"wall_s": 30, "pages": pages}))
    (e / "_scores.json").write_text(json.dumps({"pages": [{"page": "p1", "cer": 0.0}, {"page": "p2", "cer": 0.5},
                                                          {"page": "p3", "cer": 0.1}, {"page": "p4"}]}))
    out = jev_a2.dev()
    eng = out["engines"]["e1"]
    assert (eng["pages"], eng["page_err"], eng["blocks"], eng["block_err"]) == (2, 1, 2, 1)
    assert eng["page_auroc"] == 1.0 and eng["block_auroc"] == 1.0 and eng["page_prob_median"] == 0.8
    assert eng["pages_detail"] == [("p2", 0.1, False), ("p1", 0.8, True)]
    assert out["pooled_page"]["n"] == 2 and out["pooled_page"]["err"] == 1 and out["pooled_block"]["err"] == 1
    t = out["pooled_page"]["tuned"]["t"]
    assert 0.1 < t <= 0.8
    assert {k: eng[k] for k in ("cer_all", "cer_accepted", "accepted", "of", "unscored")} == {
        "cer_all": 0.25, "cer_accepted": 0.0, "accepted": 1, "of": 2, "unscored": 1}
    assert eng["jev_pages_per_min"] == 6.0
    assert json.loads((a2 / "dev.json").read_text())["engines"]["e1"]["of"] == 2


def test_drop_testset_copies_and_drops(monkeypatch, tmp_path, capsys):
    runs, _ = _a2_dirs(monkeypatch, tmp_path)
    src = runs / "lighton-1b-grounding"; src.mkdir(parents=True)
    (src / "p1.txt").write_text("![text](0,0,1,1)\nlow\n![text](0,0,1,1)\nhigh")
    (src / "p1_labels.json").write_text("{}")
    (src / "_jev.json").write_text(json.dumps({"pages": {"p1": [{"kind": "block", "i": 0, "prob": 0.1},
                                                               {"kind": "block", "i": 1, "prob": 0.9},
                                                               {"kind": "page", "prob": 0.0}]}}))
    jev_a2.drop_testset(0.5)
    dst = runs / "lighton-1b-grounding-jevdrop"
    assert (dst / "p1.txt").read_text() == "![text](0,0,1,1)\nhigh" and (dst / "p1_labels.json").exists()
    assert json.loads(capsys.readouterr().out)["dropped_blocks"] == 1


def _fake_exp_a(monkeypatch, cache, made):
    ex = types.ModuleType("exp_a"); ex.LIGHTON_CACHE = cache
    ex.candidate_dir = lambda p: made.append(p) or p
    ex.politax_link = lambda s, d: made.append(("link", s, d))
    monkeypatch.setitem(sys.modules, "exp_a", ex)


def test_drop_politax_writes_dropped_records(monkeypatch, tmp_path, capsys):
    _, a2 = _a2_dirs(monkeypatch, tmp_path); a2.mkdir()
    cache = tmp_path / "cache"; (cache / "ocr" / "s1").mkdir(parents=True)
    (cache / "ocr" / "s1" / "p00000.json").write_text(json.dumps({"raw": "![text](0,0,1,1)\nbad\n![text](0,0,1,1)\nok", "n": 1}))
    (cache / "ocr" / "s9").mkdir(); (cache / "ocr" / "s9" / "p00000.json").write_text("{}")
    _fake_packet(monkeypatch, [types.SimpleNamespace(file_name="a.pdf", sha256="s1"),
                               types.SimpleNamespace(file_name="z.pdf", sha256="s9")], {"a.pdf"})
    made = []; _fake_exp_a(monkeypatch, cache, made)
    (a2 / "politax-lighton.jsonl").write_text(json.dumps({"key": "s1/p00000.json", "rows": [
        {"kind": "block", "i": 0, "prob": 0.1}, {"kind": "block", "i": 1, "prob": 0.8}, {"kind": "page", "prob": 0.0}]}) + "\n")
    jev_a2.drop_politax(0.5)
    dst = tmp_path / "politax-lighton-jevdrop"
    assert made == [dst, ("link", cache, dst)]
    assert json.loads((dst / "ocr" / "s1" / "p00000.json").read_text()) == {"raw": "![text](0,0,1,1)\nok", "n": 1}
    assert not (dst / "ocr" / "s9").exists()
    assert json.loads(capsys.readouterr().out) == {"dst": str(dst), "records": 1, "dropped_blocks": 1, "scored_pages": 1}


def test_drop_politax_refuses_unscored_records(monkeypatch, tmp_path):
    _, a2 = _a2_dirs(monkeypatch, tmp_path); a2.mkdir()
    cache = tmp_path / "cache"; (cache / "ocr" / "s1").mkdir(parents=True)
    for i in range(2): (cache / "ocr" / "s1" / f"p0000{i}.json").write_text(json.dumps({"raw": "x"}))
    _fake_packet(monkeypatch, [types.SimpleNamespace(file_name="a.pdf", sha256="s1")], {"a.pdf"})
    made = []; _fake_exp_a(monkeypatch, cache, made)
    (a2 / "politax-lighton.jsonl").write_text(json.dumps({"key": "s1/p00000.json", "rows": []}) + "\n")
    with pytest.raises(SystemExit, match="drop-politax: 1 holdout records have no scores"):
        jev_a2.drop_politax(0.5)
    assert made == []
