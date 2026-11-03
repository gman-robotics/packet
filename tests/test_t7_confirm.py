"""GMA-462: T7 confirm bench locks. No model load and no network."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "bench" / "t7_confirm" / "README.md"
BANDS = ("[0.35, 0.74]", "[0.45, 0.74]", "[0.50, 0.65]")


def _run_py() -> Path:
    # Mutator workers overlay the mutant at cwd. Path(__file__).resolve() follows
    # the tests/ symlink back to the original tree and would load the unmutated file.
    overlay = Path.cwd() / "bench" / "t7_confirm" / "run.py"
    if overlay.is_file():
        return overlay
    return ROOT / "bench" / "t7_confirm" / "run.py"


RUN = _run_py()


def _load():
    """Compile from source each time (no stale .pyc under a mutation run)."""
    run_path = _run_py()
    module = types.ModuleType("t7_confirm_run")
    module.__file__ = str(run_path)
    sys.modules["t7_confirm_run"] = module
    exec(compile(run_path.read_text(), str(run_path), "exec"), module.__dict__)  # noqa: S102 - trusted repo file
    return module


def _trump_names() -> list[str]:
    raw = json.loads((ROOT / "evals" / "politax.json").read_text())
    return [doc["file_name"] for doc in raw["docs"] if doc["subject"] == "trump"]


def test_readme_lists_bands_and_every_trump_train_file():
    text = README.read_text()
    for band in BANDS:
        assert band in text
    names = _trump_names()
    assert len(names) == 23
    assert len(set(names)) == 23
    for name in names:
        assert f"\n- {name}\n" in text


def test_importing_run_leaves_mlx_unloaded():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('t7_confirm_run', {str(RUN)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.BANDS == ((0.35, 0.74), (0.45, 0.74), (0.50, 0.65))\n"
        "bad = [name for name in ('mlx', 'mlx_vlm') if name in sys.modules]\n"
        "assert not bad, bad\n"
        "print('clean')\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_run_source_has_no_mlx_import_and_uses_the_ocr_cache_split():
    text = RUN.read_text()
    modules: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    for name in modules:
        root = name.split(".")[0]
        assert root != "mlx"
        assert not name.startswith("mlx_vlm")
    for banned in ("import mlx", "from mlx", "run_tf", "ocr_compare", "huggingface.co"):
        assert banned not in text
    assert "partition_docs" in text
    assert "ocr_cache_pipeline" in text
    assert "mlx_vlm" in text


def test_write_config_emits_one_band_without_loading(tmp_path, capsys):
    mod = _load()
    path = tmp_path / "nested" / "deep" / "confirm.json"
    assert mod.main(["--set", "politax", "--write-config", str(path)]) == 0
    body = json.loads(path.read_text())
    assert body == {
        "low": 0.35,
        "high": 0.74,
        "task": "confirm_boundary",
        "decision": {
            "backend": "mlx_vlm",
            "model": mod.MODEL,
            "revision": mod.REVISION,
            "send_images": True,
            "capabilities": {"images": True},
        },
    }
    other = tmp_path / "narrow.json"
    assert mod.main(["--write-config", str(other), "--low", "0.50", "--high", "0.65"]) == 0
    narrow = json.loads(other.read_text())
    assert narrow["low"] == 0.50
    assert narrow["high"] == 0.65
    again = tmp_path / "nested" / "deep" / "again.json"
    assert mod.main(["--write-config", str(again)]) == 0
    assert path.read_text().endswith("\n")
    assert again.read_text().endswith("\n")
    assert mod.main(["--low", "0.2"]) == 2
    assert "both" in capsys.readouterr().err
    rejected = tmp_path / "no.json"
    assert mod.main(["--low", "0.8", "--high", "0.1", "--write-config", str(rejected)]) == 2
    assert not rejected.exists()
    assert mod.main(["--set", "politax"]) == 2
    assert "subset" in capsys.readouterr().err
    assert mod.main(["--set", "builtin", "--subset", "train"]) == 2
    assert "only for" in capsys.readouterr().err


def test_percentile_is_nearest_rank_and_false_splits_sum():
    mod = _load()
    assert mod._percentile([], 50) is None
    values = [0.4, 0.1, 0.3, 0.2]
    assert mod._percentile(values, 50) == 0.2
    assert mod._percentile(values, 95) == 0.4
    rows = [
        SimpleNamespace(boundary=SimpleNamespace(predicted=[1, 2, 3], matched=1, f1=0.5)),
        SimpleNamespace(boundary=SimpleNamespace(predicted=[4], matched=1, f1=1.0)),
    ]
    assert mod._false_splits(rows) == 2
    assert mod._false_splits([]) == 0
    assert mod._macro_f1(rows) == pytest.approx(0.75)


def test_summary_marks_holdout_unrun_and_keeps_the_first_max(tmp_path):
    mod = _load()
    rows = [
        {"low": 0.35, "high": 0.74, "macro_f1": 0.5, "eligible": True,
         "false_splits": 1, "baseline_false_splits": 2},
        {"low": 0.45, "high": 0.74, "macro_f1": 0.5, "eligible": True,
         "false_splits": 1, "baseline_false_splits": 2},
        {"low": 0.50, "high": 0.65, "macro_f1": 0.9, "eligible": False,
         "false_splits": 4, "baseline_false_splits": 2},
    ]
    assert mod._best_eligible(rows) == {"low": 0.35, "high": 0.74, "macro_f1": 0.5}
    args = SimpleNamespace(out=tmp_path, subset="train")
    mod._emit_summary(args, "politax", rows)
    body = json.loads((tmp_path / "summary-politax-train.json").read_text())
    assert body["holdout"] == "not run by this script"
    assert body["best_eligible"]["low"] == 0.35
    assert [band["eligible"] for band in body["bands"]] == [True, True, False]


class _StubDecision:
    """No-model backend: every confirm call says split."""

    def __init__(self) -> None:
        from packet.decision import DecisionCapabilities

        self.capabilities = DecisionCapabilities()
        self.send_images = False
        self.degraded_pages: list[int] = []
        self.load_seconds = 0.0
        self.calls = 0

    def decide(self, request, *, render=None):
        from packet.decision import DecisionResult

        self.calls += 1
        return DecisionResult(probability=0.9)

    def reset_parse_budget(self) -> None:
        self.degraded_pages.clear()


def test_builtin_run_scores_with_a_stub_backend(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_new_timed", lambda: mod._TimedBackend(_StubDecision()))
    out = tmp_path / "res"
    code = mod.main(["--set", "builtin", "--dir", str(tmp_path / "fx"), "--out", str(out)])
    assert code == 0
    files = sorted(p.name for p in out.iterdir())
    assert files == [
        "builtin-na-0.35-0.74.json",
        "builtin-na-0.45-0.74.json",
        "builtin-na-0.50-0.65.json",
        "summary-builtin-na.json",
    ]
    body = json.loads((out / "builtin-na-0.35-0.74.json").read_text())
    assert body["fixtures"]
    assert body["baseline_false_splits"] >= 0
    assert body["pages_confirmed"] >= 0
    assert 0 <= body["wall_seconds"] < 120
    summary = json.loads((out / "summary-builtin-na.json").read_text())
    assert summary["holdout"] == "not run by this script"


def test_score_one_counts_confirmed_reasons(tmp_path):
    from packet.confirm import BackendConfirmer
    from packet.eval.fixtures import build_builtin
    from packet.pipeline import Pipeline, PipelineConfig

    mod = _load()
    fixtures = build_builtin(tmp_path)
    confirmer = BackendConfirmer(backend=_StubDecision(), low=0.0, high=1.0)
    pipe = Pipeline(config=PipelineConfig(confirmer=confirmer))
    rows, confirmed, degraded = mod._score_many(fixtures, pipe, confirmer)
    assert len(rows) == len(fixtures)
    assert confirmed >= 0
    assert len(degraded) == len(fixtures)
    base = mod._plain_rows(fixtures)
    assert mod._changes(base, base) == 0


def test_politax_bundle_picks_only_the_requested_subset(monkeypatch, tmp_path):
    import packet.eval.politax_split as split
    from packet.eval import run

    mod = _load()
    docs = [SimpleNamespace(file_name="a.pdf"), SimpleNamespace(file_name="b.pdf")]
    fixtures = [
        SimpleNamespace(notes="a.pdf", pdf=Path("x/a.pdf")),
        SimpleNamespace(notes="", pdf=Path("x/b.pdf")),
    ]
    manifest = SimpleNamespace(docs=docs)
    monkeypatch.setattr(run, "prepare_politax", lambda data: (manifest, fixtures))
    monkeypatch.setattr(split, "partition_docs", lambda m: ([docs[0]], [docs[1]]))
    args = SimpleNamespace(data_dir=tmp_path, subset="train")
    got_m, picked, data = mod._politax_bundle(args)
    assert got_m is manifest and data == tmp_path
    assert picked == [fixtures[0]]
    args.subset = "holdout"
    assert mod._politax_bundle(args)[1] == [fixtures[1]]


def test_politax_rows_feed_each_file_through_the_cache_pipeline(monkeypatch, tmp_path):
    import packet.eval.politax_ocr_eval as ocr_eval

    mod = _load()
    manifest = SimpleNamespace(docs=[SimpleNamespace(file_name="a.pdf", sha256="s1")])
    fixtures = [SimpleNamespace(notes="a.pdf", pdf=Path("a.pdf"))]
    seen = []

    def fake_pipe(data, sha, stats, config):
        seen.append((data, sha, config))
        return "pipe"

    monkeypatch.setattr(ocr_eval, "ocr_cache_pipeline", fake_pipe)

    def fake_score(fx, pipe):
        del pipe
        return SimpleNamespace(name=fx.notes or fx.pdf.name), 2

    monkeypatch.setattr(mod, "_score_one", fake_score)
    rows, confirmed, degraded, _stats = mod._politax_rows(manifest, fixtures, tmp_path, None)
    assert confirmed == 2 and degraded == [] and len(rows) == 1
    assert seen == [(tmp_path, "s1", None)]
    unnamed = [SimpleNamespace(notes="", pdf=Path("a.pdf"))]
    _rows, _c, _d, _s = mod._politax_rows(manifest, unnamed, tmp_path, None)
    assert seen[-1] == (tmp_path, "s1", None)
    config = SimpleNamespace(confirmer=SimpleNamespace(degraded_pages=[4]))
    _rows, _c, degraded, _s = mod._politax_rows(manifest, fixtures, tmp_path, config)
    assert degraded == [{"file": "a.pdf", "pages": [4]}]


def test_score_many_sums_confirmed_pages(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_score_one", lambda fx, pipe: (SimpleNamespace(name=fx.name), 2))
    fixtures = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    rows, confirmed, degraded = mod._score_many(fixtures, "pipe", None)
    assert [row.name for row in rows] == ["a", "b"]
    assert confirmed == 4 and degraded == []
    spy = SimpleNamespace(degraded_pages=[9])
    _rows, confirmed, degraded = mod._score_many(fixtures, "pipe", spy)
    assert confirmed == 4
    assert degraded == [{"file": "a", "pages": [9]}, {"file": "b", "pages": [9]}]


def test_evaluate_politax_runs_baseline_then_confirmer(monkeypatch):
    mod = _load()
    calls = []
    monkeypatch.setattr(mod, "_politax_bundle", lambda args: ("m", ["fx"], "d"))

    def fake_rows(m, fixtures, data, config):
        calls.append(config)
        return ["row"], 1, [], "stats"

    monkeypatch.setattr(mod, "_politax_rows", fake_rows)
    got, base, confirmed, _degraded, stats, wall = mod._evaluate(
        SimpleNamespace(), "politax", "conf"
    )
    assert calls[0] is None and calls[1].confirmer == "conf"
    assert got == base == ["row"] and confirmed == 1 and stats == "stats"
    assert 0 <= wall < 10


def _ns(**kw):
    base = {"low": None, "high": None, "set": "builtin", "subset": None, "out": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_band_and_set_errors_cover_each_branch():
    mod = _load()
    assert mod._band_error(_ns()) is None
    assert "both" in mod._band_error(_ns(low=0.3))
    assert "both" in mod._band_error(_ns(high=0.3))
    assert mod._band_error(_ns(low=0.5, high=0.5)) is None
    assert "low must" in mod._band_error(_ns(low=0.6, high=0.5))
    assert mod._set_error(_ns(set="politax", subset="train")) is None
    assert mod._set_error(_ns(set="politax", subset="holdout")) is None
    assert "needs" in mod._set_error(_ns(set="politax"))
    assert mod._set_error(_ns(set="builtin")) is None
    assert "only for" in mod._set_error(_ns(set="dlnet-whole", subset="train"))
    assert mod._selected_band(_ns()) == (0.35, 0.74)
    assert mod._selected_band(_ns(low=0.4, high=0.6)) == (0.4, 0.6)
    assert mod._run_bands(_ns()) == list(mod.BANDS)
    assert mod._run_bands(_ns(low=0.4, high=0.6)) == [(0.4, 0.6)]
    assert mod._sets(_ns(set="all")) == ["builtin", "dlnet-whole"]
    assert mod._sets(_ns(set="politax")) == ["politax"]


def test_confirm_document_turns_images_on():
    mod = _load()
    doc = mod._confirm_document(0.4, 0.6)
    assert doc["decision"]["send_images"] is True
    assert doc["decision"]["capabilities"] == {"images": True}


def _row(name, predicted, matched, f1, notes=""):
    return SimpleNamespace(
        name=name,
        notes=notes,
        false_ocr_pages=[],
        boundary=SimpleNamespace(predicted=list(predicted), matched=matched, f1=f1),
    )


def test_metrics_helpers_exact_values(monkeypatch):
    mod = _load()
    rows = [_row("a", [1, 2, 3], 1, 0.2), _row("b", [4], 0, 0.6)]
    assert mod._false_splits(rows) == 3
    assert mod._macro_f1([]) == 0.0
    assert mod._macro_f1(rows) == pytest.approx(0.4)
    values = [float(v) for v in range(1, 11)]
    assert mod._percentile(values, 50) == 5.0
    assert mod._percentile(values, 95) == 10.0
    assert mod._percentile(values, 0) == 1.0
    assert mod._percentile(values, 10) == 1.0
    assert mod._percentile(values, 11) == 2.0
    import resource

    monkeypatch.setattr(resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=3))
    monkeypatch.setattr(mod.sys, "platform", "linux")
    assert mod._peak_rss_bytes() == 3072
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    assert mod._peak_rss_bytes() == 3


def test_changes_fixture_rows_and_payload(monkeypatch):
    mod = _load()
    base = [_row("a", [1], 1, 1.0), _row("b", [2], 1, 1.0)]
    got = [_row("a", [1], 1, 1.0), _row("b", [2, 5, 6], 1, 0.5, notes="n")]
    assert mod._changes(base, got) == 1
    rows = mod._fixture_rows(base, got)
    assert rows[1]["false_splits"] == 2
    assert rows[1]["baseline_predicted"] == [2] and rows[1]["notes"] == "n"
    monkeypatch.setattr(mod, "_peak_rss_bytes", lambda: 7)
    timed = SimpleNamespace(backend=SimpleNamespace(load_seconds=1.5), latencies=[0.1, 0.3])
    stats = SimpleNamespace(as_dict=lambda: {"hits": 1})
    body = mod._payload(_ns(subset="train"), "politax", 0.4, 0.6, timed, got, base, 3, [], stats, 2.0)
    assert body["false_splits"] == 2 and body["baseline_false_splits"] == 0
    assert body["eligible"] is False
    assert body["decision_changes"] == 1 and body["pages_confirmed"] == 3
    assert body["macro_f1"] == pytest.approx(0.75)
    assert body["latency_p50"] == 0.1 and body["latency_p95"] == 0.3
    assert body["ocr_cache"] == {"hits": 1} and body["peak_rss_bytes"] == 7
    same = mod._payload(_ns(), "builtin", 0.4, 0.6, timed, base, base, 0, [], None, 1.0)
    assert same["eligible"] is True and "ocr_cache" not in same


def test_best_eligible_emit_and_paths(tmp_path, capsys):
    mod = _load()
    rows = [
        {"low": 0.1, "high": 0.2, "macro_f1": 0.9, "eligible": False},
        {"low": 0.3, "high": 0.4, "macro_f1": 0.5, "eligible": True},
        {"low": 0.5, "high": 0.6, "macro_f1": 0.7, "eligible": True},
    ]
    assert mod._best_eligible(rows) == {"low": 0.5, "high": 0.6, "macro_f1": 0.7}
    assert mod._best_eligible(rows[:1]) is None
    args = _ns(out=tmp_path, subset="train")
    assert mod._result_path(args, "politax", 0.35, 0.74).name == "politax-train-0.35-0.74.json"
    assert mod._result_path(_ns(out=tmp_path), "builtin", 0.5, 0.65).name == "builtin-na-0.50-0.65.json"
    payload = {"set": "s", "subset": "t", "low": 0.1, "high": 0.2, "macro_f1": 0.5,
               "false_splits": 1, "eligible": True}
    written = tmp_path / "x" / "y" / "p.json"
    mod._emit(written, payload)
    mod._emit(written, payload)
    assert written.read_text().endswith("\n")
    out = capsys.readouterr().out
    assert "macro 0.5000 false_splits 1 eligible True" in out
    mod._emit(tmp_path / "s.json", {"set": "s", "subset": "t", "best_eligible": None})
    assert "best none" in capsys.readouterr().out
    mod._emit(tmp_path / "s.json", {"set": "s", "subset": "t",
                                     "best_eligible": {"low": 0.3, "high": 0.4}})
    assert "best [0.3, 0.4]" in capsys.readouterr().out


def test_score_one_counts_only_confirmed_reasons(tmp_path, monkeypatch):
    from packet.eval import metrics, run
    from packet.events import EventKind

    mod = _load()

    class _Pred:
        def document_count(self):
            return 1

    class _Pipe:
        def parse(self, pdf, on_event):
            for reason in ("confirmed_split", "fallback_eager", "confirmed_continue"):
                on_event(SimpleNamespace(kind=EventKind.BOUNDARY_DECIDED,
                                         payload={"reason": reason}))
            on_event(SimpleNamespace(kind=EventKind.PAGE_CLASSIFIED,
                                     payload={"reason": "confirmed_split"}))
            return _Pred()

    monkeypatch.setattr(metrics, "boundary_score", lambda gold, pred, tolerance: "score")
    monkeypatch.setattr(metrics, "false_ocr_hard_fail", lambda pred, pages: [])
    monkeypatch.setattr(run, "text_layer_pages", lambda pdf: [])
    fx = SimpleNamespace(
        pdf=tmp_path / "a.pdf",
        gold=SimpleNamespace(origin=SimpleNamespace(page_count=2), document_count=lambda: 1),
        name="a", condition="c", notes="",
    )
    result, confirmed = mod._score_one(fx, _Pipe())
    assert confirmed == 2
    assert result.boundary == "score" and result.pages == 2


def test_score_one_does_not_credit_a_one_page_miss(tmp_path, monkeypatch):
    from packet.eval import run
    from packet.types import LogicalDocument, Packet, PacketOrigin

    gold = Packet(
        id="g",
        origin=PacketOrigin(path="a.pdf", page_count=4),
        documents=[
            LogicalDocument(id="0", page_start=0, page_end=1),
            LogicalDocument(id="1", page_start=2, page_end=3),
        ],
    )
    pred = Packet(
        id="p",
        origin=PacketOrigin(path="a.pdf", page_count=4),
        documents=[
            LogicalDocument(id="0", page_start=0, page_end=2),
            LogicalDocument(id="1", page_start=3, page_end=3),
        ],
    )

    class _Pipe:
        def parse(self, pdf, on_event):
            del pdf, on_event
            return pred

    monkeypatch.setattr(run, "text_layer_pages", lambda pdf: set())
    fx = SimpleNamespace(
        pdf=tmp_path / "a.pdf",
        gold=gold,
        name="a",
        condition="c",
        notes="",
    )
    result, confirmed = _load()._score_one(fx, _Pipe())
    assert confirmed == 0
    assert result.boundary.matched == 0
    assert result.boundary.f1 == 0.0


def test_run_writes_summary_only_for_several_bands(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_new_timed", lambda: mod._TimedBackend(_StubDecision()))
    out = tmp_path / "one"
    assert mod.main(["--set", "builtin", "--dir", str(tmp_path / "fx"), "--out", str(out),
                     "--low", "0.4", "--high", "0.6"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["builtin-na-0.40-0.60.json"]


def test_evaluate_picks_fixture_loader_by_set(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_builtin_fixtures", lambda args: ["builtin"])
    monkeypatch.setattr(mod, "_dlnet_fixtures", lambda args: ["dlnet"])
    monkeypatch.setattr(mod, "_plain_rows", lambda fixtures: list(fixtures))
    monkeypatch.setattr(mod, "_score_many", lambda fixtures, pipe, confirmer: (list(fixtures), 0, []))
    for set_name, want in (("builtin", ["builtin"]), ("dlnet-whole", ["dlnet"])):
        got, base, confirmed, degraded, stats, _wall = mod._evaluate(_ns(set=set_name), set_name, None)
        assert (got, base, confirmed, degraded, stats) == (want, want, 0, [], None)
