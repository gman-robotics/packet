"""DB-5 (GMA-264): --decision-config CLI, timings, and checklist guards."""

from __future__ import annotations

import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from packet import cli
from packet.decision import DecisionBackendError, DecisionResult, load_decision_config
from packet.decision_http import HttpDecisionBackend
from packet.eval.pdfutil import write_text_pdf
from packet.eval.run import run_fixture
from packet.extract import NativeTextExtractor
from packet.pipeline import PipelineConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "decision"
SENTINEL_KEY = "packet-ci-sentinel-key-do-not-print"


def _minimal_pdf(tmp_path: Path, pages: int = 2) -> Path:
    lines = [[f"Page {i} body text."] for i in range(pages)]
    return write_text_pdf(tmp_path / "doc.pdf", lines)


def _native_extractors_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_extractors_or_exit", lambda **_kw: [NativeTextExtractor()])


def _packet_stderr_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("packet:")]


def test_config_http_builds_http_adapter():
    backend = load_decision_config(FIXTURES / "config_http.json")
    assert isinstance(backend, HttpDecisionBackend)
    assert backend.base_url == "http://127.0.0.1:8080/v1"
    assert backend.auth == "env"


def test_config_logistic_json_loads():
    backend = load_decision_config(FIXTURES / "config_logistic.json")
    from packet.decision import LogisticDecisionBackend

    assert isinstance(backend, LogisticDecisionBackend)


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
                "auth": "none",
                "max_calls": 100,
            }
        )
    )
    return path


def test_single_file_decision_backend_error_one_packet_line(
    tmp_path, capsys, monkeypatch
):
    _native_extractors_only(monkeypatch)
    monkeypatch.delenv("PACKET_DECISION_API_KEY", raising=False)
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
    monkeypatch.delenv("PACKET_DECISION_API_KEY", raising=False)
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
    assert list(out_dir.glob("*.json")) == []


def test_timings_error_safe_500_text(tmp_path, capsys, monkeypatch):
    _native_extractors_only(monkeypatch)
    monkeypatch.delenv("PACKET_DECISION_API_KEY", raising=False)
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
        assert SENTINEL_KEY not in err
        row = json.loads(timings.read_text().strip())
        assert row["ok"] is False
        assert "500" in row["error"]
        assert "1" in row["error"]
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


def test_inspect_builds_pipeline_without_scorer():
    source = Path("src/packet/cli.py").read_text()
    start = source.index("path = Path(args.path)")
    end = source.index("def _parse_pipeline_config")
    inspect_block = source[start:end]
    assert "Pipeline(extractors=_extractors_or_exit" in inspect_block
    assert "decision_config" not in inspect_block
    assert "config.scorer" not in inspect_block


def test_http_config_does_not_set_default_scorer():
    backend = load_decision_config(FIXTURES / "config_http.json")
    config = PipelineConfig(decision_backend=backend)
    assert config.scorer is None


def test_transformers_config_does_not_set_scorer():
    with pytest.raises(DecisionBackendError, match="unknown decision backend"):
        load_decision_config(FIXTURES / "config_transformers.json")
    config = PipelineConfig()
    assert config.scorer is None


def test_timings_success_includes_decision_fields(tmp_path, monkeypatch):
    backend = HttpDecisionBackend(
        base_url="http://127.0.0.1:1/v1",
        model="m",
        max_calls=100,
        auth="none",
    )
    backend.degraded_pages.append(2)

    def _fake_decide(request, *, render=None):  # type: ignore[no-untyped-def]
        return DecisionResult(probability=0.5)

    backend.decide = _fake_decide  # type: ignore[method-assign]
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
    assert row["decision_degraded_pages"] == []
