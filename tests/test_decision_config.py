"""DB-5 (GMA-264): --decision-config CLI, timings, and checklist guards."""

from __future__ import annotations

import argparse
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from packet import cli
from packet.boundary_model import load_logistic_scorer
from packet.decision import (
    DecisionBackendError,
    DecisionResult,
    LogisticDecisionBackend,
    load_decision_config,
)
from packet.decision_http import HttpDecisionBackend
from packet.decision_transformers import TransformersDecisionBackend
from packet.eval.pdfutil import write_text_pdf
from packet.eval.run import run_fixture
from packet.extract import NativeTextExtractor
from packet.pipeline import PipelineConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "decision"
SENTINEL_KEY = "packet-ci-sentinel-key-do-not-print"
HTTP_500_MSG = "decision http 500 page 1"

_DECISION_BACKEND_IMPLEMENTERS: tuple[tuple[str, type], ...] = (
    ("http", HttpDecisionBackend),
    ("logistic", LogisticDecisionBackend),
    ("transformers", TransformersDecisionBackend),
)


def _minimal_pdf(tmp_path: Path, pages: int = 2) -> Path:
    lines = [[f"Page {i} body text."] for i in range(pages)]
    return write_text_pdf(tmp_path / "doc.pdf", lines)


def _native_extractors_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_extractors_or_exit", lambda **_kw: [NativeTextExtractor()])


def _sentinel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACKET_DECISION_API_KEY", SENTINEL_KEY)


def _packet_stderr_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("packet:")]


def _minimal_backend_instance(kind: str) -> object:
    if kind == "http":
        return HttpDecisionBackend(
            base_url="http://127.0.0.1:1/v1",
            model="m",
            max_calls=1,
            auth="none",
        )
    if kind == "logistic":
        return LogisticDecisionBackend(load_logistic_scorer(FIXTURES / "logistic_weights.json"))
    if kind == "transformers":
        return TransformersDecisionBackend(model="m", revision="r")
    raise AssertionError(kind)


def test_every_decision_backend_implementer_has_protocol_members():
    import test_decision_boundary

    implementers = list(_DECISION_BACKEND_IMPLEMENTERS) + [
        ("fake", test_decision_boundary.FakeDecisionBackend),
    ]
    for name, cls in implementers:
        if name == "fake":
            backend = cls({1: 0.5})
        else:
            backend = _minimal_backend_instance(name)
        assert hasattr(backend, "backend_label"), name
        assert hasattr(backend, "reset_parse_budget"), name
        assert hasattr(backend, "decide"), name
        assert hasattr(backend, "capabilities"), name
        assert hasattr(backend, "degraded_pages"), name
        assert isinstance(backend.backend_label, str)
        backend.reset_parse_budget()


def _capture_pipeline_config(monkeypatch: pytest.MonkeyPatch) -> list[PipelineConfig]:
    seen: list[PipelineConfig] = []
    real = cli.Pipeline

    def _wrap(*args, **kwargs):  # type: ignore[no-untyped-def]
        cfg = kwargs.get("config")
        if cfg is not None:
            seen.append(cfg)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "Pipeline", _wrap)
    return seen


def test_config_http_builds_http_adapter():
    backend = load_decision_config(FIXTURES / "config_http.json")
    assert isinstance(backend, HttpDecisionBackend)
    assert backend.base_url == "http://127.0.0.1:8080/v1"
    assert backend.auth == "env"
    assert backend.backend_label == "http"


def test_config_logistic_json_loads():
    from packet.boundary_model import load_logistic_scorer

    cfg = json.loads((FIXTURES / "config_logistic.json").read_text())
    assert cfg["weights"] == "logistic_weights.json"
    weights_path = FIXTURES / cfg["weights"]
    backend = load_decision_config(FIXTURES / "config_logistic.json")
    from packet.decision import LogisticDecisionBackend

    assert isinstance(backend, LogisticDecisionBackend)
    reference = load_logistic_scorer(weights_path)
    assert backend.scorer.as_weights() == reference.as_weights()


def test_config_logistic_missing_exits_before_parse(tmp_path, capsys):
    pdf = _minimal_pdf(tmp_path)
    rc = cli.main(
        [
            "parse",
            str(pdf),
            "-o",
            str(tmp_path / "out.json"),
            "--decision-config",
            str(FIXTURES / "config_logistic_missing.json"),
        ]
    )
    err = capsys.readouterr().err.strip()
    assert rc == 2
    assert err.count("\n") == 0
    assert err.startswith("packet:")
    assert not (tmp_path / "out.json").exists()


def test_both_cli_flags_exit_2(tmp_path, capsys):
    pdf = _minimal_pdf(tmp_path)
    rc = cli.main(
        [
            "parse",
            str(pdf),
            "-o",
            str(tmp_path / "out.json"),
            "--decision-config",
            str(FIXTURES / "config_http.json"),
            "--boundary-weights",
            str(FIXTURES / "logistic_weights.json"),
        ]
    )
    err = capsys.readouterr().err.strip()
    assert rc == 2
    assert err.count("\n") == 0
    assert "boundary-weights" in err


class _FakeChatHandler(BaseHTTPRequestHandler):
    script: ClassVar[list[tuple[int, bytes]]] = []

    def log_message(self, format, *args):  # type: ignore[no-untyped-def]
        del format, args

    def do_POST(self) -> None:
        status, body = (500, b"err")
        if self.script:
            status, body = self.script.pop(0)
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _loopback_config(tmp_path: Path, port: int) -> Path:
    path = tmp_path / "http.json"
    path.write_text(
        json.dumps(
            {
                "backend": "http",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "model": "local-model",
                "auth": "env",
                "max_calls": 100,
            }
        )
    )
    return path


def test_single_file_decision_backend_error_one_packet_line(
    tmp_path, capsys, monkeypatch
):
    _native_extractors_only(monkeypatch)
    _sentinel_env(monkeypatch)
    pdf = _minimal_pdf(tmp_path)
    rc = cli.main(
        [
            "parse",
            str(pdf),
            "-o",
            str(tmp_path / "out.json"),
            "--decision-config",
            str(FIXTURES / "config_http_max_calls_0.json"),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    lines = _packet_stderr_lines(err)
    assert len(lines) == 1
    assert "max_calls" in lines[0]
    assert SENTINEL_KEY not in err
    assert not (tmp_path / "out.json").exists()


def test_batch_decision_backend_error_skips_second_file(tmp_path, capsys, monkeypatch):
    _native_extractors_only(monkeypatch)
    _sentinel_env(monkeypatch)
    pdf1 = write_text_pdf(tmp_path / "a.pdf", [["one"], ["two"]])
    pdf2 = write_text_pdf(tmp_path / "b.pdf", [["three"], ["four"]])
    out_dir = tmp_path / "out"
    rc = cli.main(
        [
            "parse",
            str(pdf1),
            str(pdf2),
            "--out-dir",
            str(out_dir),
            "--decision-config",
            str(FIXTURES / "config_http_max_calls_0.json"),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert len(_packet_stderr_lines(err)) == 1
    assert SENTINEL_KEY not in err
    assert list(out_dir.glob("*.json")) == []


def test_timings_error_safe_500_text(tmp_path, capsys, monkeypatch):
    _native_extractors_only(monkeypatch)
    _sentinel_env(monkeypatch)
    _FakeChatHandler.script = [(500, b"err")]
    httpd = HTTPServer(("127.0.0.1", 0), _FakeChatHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        pdf = _minimal_pdf(tmp_path)
        timings = tmp_path / "timings.jsonl"
        config = _loopback_config(tmp_path, port)
        rc = cli.main(
            [
                "parse",
                str(pdf),
                "-o",
                str(tmp_path / "out.json"),
                "--decision-config",
                str(config),
                "--timings",
                str(timings),
            ]
        )
        err = capsys.readouterr().err
        assert rc == 2
        lines = _packet_stderr_lines(err)
        assert len(lines) == 1
        assert lines[0] == f"packet: {HTTP_500_MSG}"
        assert SENTINEL_KEY not in err
        row = json.loads(timings.read_text().strip())
        assert row["ok"] is False
        assert row["error"] == HTTP_500_MSG
        assert SENTINEL_KEY not in json.dumps(row)
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_reset_strikes_and_calls_between_files(tmp_path, monkeypatch):
    class _TrackBackend(HttpDecisionBackend):
        def __init__(self) -> None:
            super().__init__(
                base_url="http://127.0.0.1:1/v1",
                model="m",
                max_calls=100,
                auth="none",
            )
            self.parse_starts: list[tuple[int, int]] = []

        def reset_parse_budget(self) -> None:
            self.parse_starts.append((len(self.degraded_pages), self._calls_made))
            super().reset_parse_budget()

        def decide(self, request, *, render=None):  # type: ignore[no-untyped-def]
            if self._calls_made >= self.max_calls:
                raise DecisionBackendError("max_calls exceeded")
            self._calls_made += 1
            if len(self.parse_starts) == 1 and self._calls_made == 1:
                self.degraded_pages.append(1)
            return DecisionResult(probability=0.5)

    backend = _TrackBackend()
    _native_extractors_only(monkeypatch)
    monkeypatch.setattr(
        "packet.decision.load_decision_config",
        lambda _p: backend,
    )
    pdf1 = write_text_pdf(tmp_path / "a.pdf", [["one"], ["two"]])
    pdf2 = write_text_pdf(tmp_path / "b.pdf", [["three"], ["four"]])
    out_dir = tmp_path / "out"
    rc = cli.main(
        [
            "parse",
            str(pdf1),
            str(pdf2),
            "--out-dir",
            str(out_dir),
            "--decision-config",
            str(FIXTURES / "config_http.json"),
        ]
    )
    assert rc == 0
    assert len(list(out_dir.glob("*.json"))) == 2
    assert backend.parse_starts[0] == (0, 0)
    assert backend.parse_starts[1] == (1, 1)
    assert backend.degraded_pages == []
    assert backend._calls_made == 1


def test_eval_run_fixture_source_uses_pipeline_or_pipeline():
    assert "pipeline or Pipeline()" in inspect.getsource(run_fixture)


def test_eval_does_not_read_packet_decision_api_key():
    text = Path("src/packet/eval/run.py").read_text()
    assert "PACKET_DECISION_API_KEY" not in text


def test_inspect_builds_pipeline_without_scorer(tmp_path, monkeypatch):
    pdf = _minimal_pdf(tmp_path)
    configs: list[PipelineConfig] = []

    class _SpyPipeline:
        def __init__(self, classifier=None, extractors=None, config=None):  # type: ignore[no-untyped-def]
            self.config = config or PipelineConfig()
            configs.append(self.config)

        def stream(self, path):  # type: ignore[no-untyped-def]
            return iter([])

    monkeypatch.setattr(cli, "Pipeline", _SpyPipeline)
    _native_extractors_only(monkeypatch)
    rc = cli.main(["inspect", str(pdf)])
    assert rc == 0
    assert len(configs) == 1
    assert configs[0].scorer is None


def test_parse_logistic_decision_config_leaves_scorer_none(tmp_path, monkeypatch):
    _native_extractors_only(monkeypatch)
    seen = _capture_pipeline_config(monkeypatch)
    pdf = _minimal_pdf(tmp_path)
    rc = cli.main(
        [
            "parse",
            str(pdf),
            "-o",
            str(tmp_path / "out.json"),
            "--decision-config",
            str(FIXTURES / "config_logistic.json"),
        ]
    )
    assert rc == 0
    assert len(seen) == 1
    assert seen[0].scorer is None
    assert seen[0].decision_backend is not None


def test_parse_http_decision_config_leaves_scorer_none(tmp_path, monkeypatch):
    _native_extractors_only(monkeypatch)
    _sentinel_env(monkeypatch)
    seen = _capture_pipeline_config(monkeypatch)
    pdf = _minimal_pdf(tmp_path)
    rc = cli.main(
        [
            "parse",
            str(pdf),
            "-o",
            str(tmp_path / "out.json"),
            "--decision-config",
            str(FIXTURES / "config_http_max_calls_0.json"),
        ]
    )
    assert rc == 2
    assert len(seen) == 1
    assert seen[0].scorer is None
    assert seen[0].decision_backend is not None


def test_transformers_config_does_not_set_scorer():
    args = argparse.Namespace(
        boundary_weights=None,
        decision_config=str(FIXTURES / "config_transformers.json"),
    )
    config = cli._parse_pipeline_config(args)
    assert config is not None
    assert config.scorer is None
    assert isinstance(config.decision_backend, TransformersDecisionBackend)


def test_parse_transformers_config_through_cli_loop(tmp_path, monkeypatch):
    _native_extractors_only(monkeypatch)

    class _StubRunner:
        def generate(self, prompt: str) -> str:
            return "0.62"

    def _stub_load_transformers_runner(
        model: str, revision: str, device: str, *, max_output: int = 16
    ) -> _StubRunner:
        return _StubRunner()

    monkeypatch.setattr(
        "packet.decision_transformers.load_transformers_runner",
        _stub_load_transformers_runner,
    )
    pdf = _minimal_pdf(tmp_path)
    out = tmp_path / "out.json"
    rc = cli.main(
        [
            "parse",
            str(pdf),
            "-o",
            str(out),
            "--decision-config",
            str(FIXTURES / "config_transformers.json"),
        ]
    )
    assert rc == 0
    assert out.is_file()
    backend = load_decision_config(FIXTURES / "config_transformers.json")
    assert backend.backend_label == "transformers"


def test_timings_success_includes_decision_fields(tmp_path, monkeypatch):
    class _DegradeOnDecide(HttpDecisionBackend):
        def decide(self, request, *, render=None):  # type: ignore[no-untyped-def]
            self.degraded_pages.append(request.page_index)
            return DecisionResult(probability=0.5)

    backend = _DegradeOnDecide(
        base_url="http://127.0.0.1:1/v1",
        model="m",
        max_calls=100,
        auth="none",
    )
    _native_extractors_only(monkeypatch)
    monkeypatch.setattr("packet.decision.load_decision_config", lambda _p: backend)
    pdf = _minimal_pdf(tmp_path)
    timings = tmp_path / "timings.jsonl"
    rc = cli.main(
        [
            "parse",
            str(pdf),
            "-o",
            str(tmp_path / "out.json"),
            "--decision-config",
            str(FIXTURES / "config_http.json"),
            "--timings",
            str(timings),
        ]
    )
    assert rc == 0
    row = json.loads(timings.read_text().strip())
    assert row["decision_backend"] == "http"
    assert row["decision_degraded_pages"] == [1]
