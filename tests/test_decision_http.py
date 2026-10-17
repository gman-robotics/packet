"""DB-3 (GMA-262): HTTP decision adapter and fake server."""

from __future__ import annotations

import importlib
import json
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from packet.decision import (
    DecisionBackendError,
    DecisionCapabilities,
    DecisionRequest,
    load_decision_config,
)
from packet.decision_http import (
    _BOUNDARY_RESPONSE_FORMAT,
    HttpDecisionBackend,
    load_http_decision_backend,
)
from packet.types import Block, BlockKind, PageClass, PageRef, SourceKind

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "decision"
SENTINEL_KEY = "packet-ci-sentinel-key-do-not-print"


def _load_facts() -> list[dict]:
    return json.loads((FIXTURES / "logistic_facts.json").read_text())


def _facts_for_page(page_index: int) -> list[dict]:
    facts = _load_facts()
    template = facts[1]
    while len(facts) <= page_index:
        facts.append(dict(template))
    return facts


def _page1_request(**overrides: Any) -> DecisionRequest:
    facts = _load_facts()
    ref = PageRef(packet_id="p", index=1, page_class=PageClass.MIXED, class_confidence=0.45)
    block = Block(
        id="b0",
        page=1,
        kind=BlockKind.TITLE,
        text="alpha",
        source=SourceKind.NATIVE,
        confidence=0.8,
    )
    return DecisionRequest(
        task="boundary",
        page_index=1,
        facts=facts,
        blocks=[block],
        ref=ref,
        **overrides,
    )


def _envelope(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


class _FakeChatHandler(BaseHTTPRequestHandler):
    script: ClassVar[list[tuple[int, str | bytes, dict[str, str] | None]]] = []
    captured: ClassVar[list[dict[str, Any]]] = []
    captured_headers: ClassVar[list[dict[str, str]]] = []
    connect_count: ClassVar[int] = 0

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_POST(self) -> None:
        _FakeChatHandler.connect_count += 1
        host = self.headers.get("Host", "").split(":")[0]
        assert host == "127.0.0.1", f"unexpected host {host!r}"
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        _FakeChatHandler.captured_headers.append(
            {name: self.headers[name] for name in self.headers}
        )
        _FakeChatHandler.captured.append(json.loads(raw.decode("utf-8")))
        if not _FakeChatHandler.script:
            self.send_response(500)
            self.end_headers()
            return
        status, body, headers = _FakeChatHandler.script.pop(0)
        self.send_response(status)
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        if isinstance(body, str):
            body_bytes = body.encode("utf-8")
        else:
            body_bytes = body
        if status == 200 and body_bytes:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        if body_bytes:
            self.wfile.write(body_bytes)

    def do_GET(self) -> None:
        self.send_response(405)
        self.end_headers()


class _FakeServer:
    def __init__(self) -> None:
        _FakeChatHandler.script = []
        _FakeChatHandler.captured = []
        _FakeChatHandler.captured_headers = []
        _FakeChatHandler.connect_count = 0
        self._httpd = HTTPServer(("127.0.0.1", 0), _FakeChatHandler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=5)

    def queue(self, status: int, body: str | bytes, headers: dict[str, str] | None = None) -> None:
        _FakeChatHandler.script.append((status, body, headers))

    def queue_json(self, status: int, payload: dict[str, Any]) -> None:
        self.queue(status, json.dumps(payload))


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACKET_DECISION_API_KEY", SENTINEL_KEY)


def _backend(
    server: _FakeServer,
    *,
    max_calls: int = 10,
    input_mode: str = "default",
    send_images: bool = False,
    capabilities: dict[str, bool] | None = None,
    max_retries: int = 2,
    auth: str = "env",
) -> HttpDecisionBackend:
    caps = capabilities or {
        "structured_output": False,
        "logprobs": False,
        "images": False,
    }
    return HttpDecisionBackend(
        base_url=server.base_url,
        model="local-model",
        max_calls=max_calls,
        input_mode=input_mode,
        send_images=send_images,
        capabilities=DecisionCapabilities(
            structured_output=caps.get("structured_output", False),
            logprobs=caps.get("logprobs", False),
            images=caps.get("images", False),
        ),
        max_retries=max_retries,
        auth=auth,
    )


def _snapshot_backend(backend: HttpDecisionBackend, result: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "probability": result.probability,
        "scores": result.scores,
        "label": result.label,
        "degraded_pages": list(backend.degraded_pages),
        "no_structured_output": backend.no_structured_output,
        "no_logprobs": backend.no_logprobs,
    }
    return out


def test_import_packet_avoids_torch_transformers() -> None:
    for name in list(sys.modules):
        if name in ("torch", "transformers") or name.startswith(("torch.", "transformers.")):
            del sys.modules[name]
    importlib.import_module("packet")
    importlib.import_module("packet.decision")
    importlib.import_module("packet.decision_http")

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_captured_request_features_matches_fixture():
    expected = json.loads((FIXTURES / "http_request_features.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(server, input_mode="features")
        server.queue_json(200, _envelope("0.5"))
        backend.decide(_page1_request(input_mode="features"))
        assert _FakeChatHandler.captured[-1] == expected
    finally:
        server.close()


def test_captured_request_default_matches_fixture():
    expected = json.loads((FIXTURES / "http_request_default.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(server, input_mode="default")
        server.queue_json(200, _envelope("0.5"))
        backend.decide(_page1_request(input_mode="default"))
        assert _FakeChatHandler.captured[-1] == expected
    finally:
        server.close()


def test_structured_response_matches_expected():
    content = (FIXTURES / "http_response_structured.json").read_text().rstrip("\n")
    expected = json.loads((FIXTURES / "http_structured_expected.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(
            server,
            capabilities={"structured_output": True, "logprobs": True, "images": False},
        )
        server.queue_json(200, _envelope(content))
        result = backend.decide(_page1_request())
        body = _FakeChatHandler.captured[-1]
        assert body["response_format"] == _BOUNDARY_RESPONSE_FORMAT
        snap = _snapshot_backend(backend, result)
        assert snap["probability"] == expected["probability"]
        assert snap["label"] == expected["label"]
        assert snap["scores"] == expected["scores"]
        assert snap["degraded_pages"] == expected["degraded_pages"]
        assert snap["no_structured_output"] == expected["no_structured_output"]
        assert snap["no_logprobs"] == expected["no_logprobs"]
    finally:
        server.close()


def test_plain_response_matches_expected():
    content = (FIXTURES / "http_response_plain.txt").read_text()
    expected = json.loads((FIXTURES / "http_plain_expected.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(server)
        server.queue_json(200, _envelope(content))
        result = backend.decide(_page1_request())
        snap = _snapshot_backend(backend, result)
        assert snap["probability"] == expected["probability"]
        assert snap["scores"] == expected["scores"]
        assert snap["degraded_pages"] == expected["degraded_pages"]
        assert snap["no_structured_output"] == expected["no_structured_output"]
        assert snap["no_logprobs"] == expected["no_logprobs"]
    finally:
        server.close()


def test_three_plain_responses():
    expected = json.loads((FIXTURES / "three_plain_expected.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=10)
        for _ in expected["results"]:
            server.queue_json(200, _envelope("0.62"))
        probs = []
        for page in (1, 2, 3):
            req = _page1_request()
            result = backend.decide(
                DecisionRequest(
                    task=req.task,
                    page_index=page,
                    facts=_facts_for_page(page),
                    blocks=req.blocks,
                    ref=PageRef(
                        packet_id="p",
                        index=page,
                        page_class=PageClass.MIXED,
                        class_confidence=0.45,
                    ),
                )
            )
            probs.append(result.probability)
        assert probs == expected["results"]
        assert backend.degraded_pages == expected["degraded_pages"]
        assert backend.no_structured_output == expected["no_structured_output"]
        assert backend.no_logprobs == expected["no_logprobs"]
    finally:
        server.close()


def test_garbage_response_strike():
    content = (FIXTURES / "http_response_garbage.txt").read_text()
    expected = json.loads((FIXTURES / "http_garbage_expected.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(server)
        server.queue_json(200, _envelope(content))
        result = backend.decide(_page1_request())
        assert result.probability == expected["probability"]
        assert backend.degraded_pages == expected["degraded_pages"]
    finally:
        server.close()


def test_third_strike_raises():
    expected = json.loads((FIXTURES / "third_strike_expected.json").read_text())
    garbage = (FIXTURES / "http_response_garbage.txt").read_text()
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=10)
        for page in (1, 2, 3):
            server.queue_json(200, _envelope(garbage))
        for page in (1, 2):
            backend.decide(
                DecisionRequest(
                    task="boundary",
                    page_index=page,
                    facts=_facts_for_page(page),
                    blocks=_page1_request().blocks,
                    ref=PageRef(packet_id="p", index=page, page_class=PageClass.MIXED, class_confidence=0.45),
                )
            )
        assert backend.degraded_pages == expected["degraded_pages"]
        with pytest.raises(DecisionBackendError) as exc:
            backend.decide(
                DecisionRequest(
                    task="boundary",
                    page_index=3,
                    facts=_facts_for_page(3),
                    blocks=_page1_request().blocks,
                    ref=PageRef(packet_id="p", index=3, page_class=PageClass.MIXED, class_confidence=0.45),
                )
            )
        msg = str(exc.value)
        assert "unparseable" in msg
        assert "3" in msg
        assert garbage not in msg
        assert SENTINEL_KEY not in msg
    finally:
        server.close()


def _retry_strike_with_urlopen(exc: BaseException) -> None:
    sleeps: list[float] = []
    server = _FakeServer()
    try:
        backend = _backend(server, max_retries=2, max_calls=10)
        backend._sleep = sleeps.append  # type: ignore[method-assign]
        calls = 0

        def urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            raise exc

        backend._urlopen = urlopen  # type: ignore[method-assign]
        result = backend.decide(_page1_request())
        assert result.probability is None
        assert calls == 3
        assert sleeps == [0.5, 1.0]
        assert backend.degraded_pages == [1]
    finally:
        server.close()


def test_timeout_bare_retries_to_strike():
    _retry_strike_with_urlopen(TimeoutError("timed out"))


def test_timeout_urlerror_retries_to_strike():
    _retry_strike_with_urlopen(urllib.error.URLError(TimeoutError("timed out")))


def test_connection_refused_retries_to_strike():
    sleeps: list[float] = []
    server = _FakeServer()
    try:
        backend = _backend(server, max_retries=2, max_calls=10)
        backend._sleep = sleeps.append  # type: ignore[method-assign]
        calls = 0

        def urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

        backend._urlopen = urlopen  # type: ignore[method-assign]
        result = backend.decide(_page1_request())
        assert result.probability is None
        assert calls == 3
        assert sleeps == [0.5, 1.0]
        assert backend.degraded_pages == [1]
    finally:
        server.close()


def test_connection_reset_on_read_retries_to_strike():
    sleeps: list[float] = []
    server = _FakeServer()
    try:
        backend = _backend(server, max_retries=2, max_calls=10)
        backend._sleep = sleeps.append  # type: ignore[method-assign]
        calls = 0

        class _Resp:
            status = 200

            def read(self) -> bytes:
                raise ConnectionResetError(104, "Connection reset by peer")

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return _Resp()

        backend._urlopen = urlopen  # type: ignore[method-assign]
        result = backend.decide(_page1_request())
        assert result.probability is None
        assert calls == 3
        assert sleeps == [0.5, 1.0]
        assert backend.degraded_pages == [1]
    finally:
        server.close()


def test_200_invalid_json_records_strike():
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=10)
        server.queue(200, b"{not-json")
        result = backend.decide(_page1_request())
        assert result.probability is None
        assert backend.degraded_pages == [1]
    finally:
        server.close()


def test_200_empty_choices_records_strike():
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=10)
        server.queue_json(200, {"choices": []})
        result = backend.decide(_page1_request())
        assert result.probability is None
        assert backend.degraded_pages == [1]
    finally:
        server.close()


def test_200_non_string_content_records_strike():
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=10)
        server.queue_json(200, {"choices": [{"message": {"content": 0.62}}]})
        result = backend.decide(_page1_request())
        assert result.probability is None
        assert backend.degraded_pages == [1]
    finally:
        server.close()


def test_third_strike_on_invalid_json_envelope():
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=10)
        for _ in range(3):
            server.queue(200, b"not-json")
        for page in (1, 2):
            backend.decide(
                DecisionRequest(
                    task="boundary",
                    page_index=page,
                    facts=_facts_for_page(page),
                    blocks=_page1_request().blocks,
                    ref=PageRef(packet_id="p", index=page, page_class=PageClass.MIXED, class_confidence=0.45),
                )
            )
        with pytest.raises(DecisionBackendError, match="unparseable"):
            backend.decide(
                DecisionRequest(
                    task="boundary",
                    page_index=3,
                    facts=_facts_for_page(3),
                    blocks=_page1_request().blocks,
                    ref=PageRef(packet_id="p", index=3, page_class=PageClass.MIXED, class_confidence=0.45),
                )
            )
    finally:
        server.close()


def test_plain_nan_inf_not_probability():
    from packet.decision_http import HttpDecisionBackend as HB

    assert HB._try_plain_probability("nan") is None
    assert HB._try_plain_probability("inf") is None
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=10)
        server.queue_json(200, _envelope("nan"))
        result = backend.decide(_page1_request())
        assert result.probability is None
        assert backend.degraded_pages == [1]
    finally:
        server.close()


def test_auth_env_sends_bearer_not_in_body():
    server = _FakeServer()
    try:
        backend = _backend(server, auth="env", max_calls=10)
        server.queue_json(200, _envelope("0.5"))
        backend.decide(_page1_request())
        headers = _FakeChatHandler.captured_headers[-1]
        assert headers.get("Authorization") == f"Bearer {SENTINEL_KEY}"
        body_text = json.dumps(_FakeChatHandler.captured[-1])
        assert SENTINEL_KEY not in body_text
    finally:
        server.close()


def test_auth_none_omits_authorization():
    server = _FakeServer()
    try:
        backend = _backend(server, auth="none", max_calls=10)
        server.queue_json(200, _envelope("0.5"))
        backend.decide(_page1_request())
        headers = _FakeChatHandler.captured_headers[-1]
        assert "Authorization" not in headers
    finally:
        server.close()


def test_retry_exhaustion_one_strike_fake_clock():
    sleeps: list[float] = []
    server = _FakeServer()
    try:
        backend = _backend(server, max_retries=2)
        backend._sleep = sleeps.append  # type: ignore[method-assign]
        for _ in range(3):
            server.queue(503, b"")
        result = backend.decide(_page1_request())
        assert result.probability is None
        assert backend.degraded_pages == [1]
        assert sleeps == [0.5, 1.0]
        assert _FakeChatHandler.connect_count == 3
    finally:
        server.close()


def test_400_and_401_not_retried():
    for status in (400, 401):
        server = _FakeServer()
        try:
            backend = _backend(server, max_retries=2)
            sleeps: list[float] = []
            backend._sleep = sleeps.append  # type: ignore[method-assign]
            server.queue(status, b"bad")
            with pytest.raises(DecisionBackendError) as exc:
                backend.decide(_page1_request())
            assert str(status) in str(exc.value)
            assert "1" in str(exc.value)
            assert sleeps == []
            assert _FakeChatHandler.connect_count == 1
        finally:
            server.close()


def test_500_raises_without_retry():
    server = _FakeServer()
    try:
        backend = _backend(server, max_retries=2)
        sleeps: list[float] = []
        backend._sleep = sleeps.append  # type: ignore[method-assign]
        server.queue(500, b"err")
        with pytest.raises(DecisionBackendError) as exc:
            backend.decide(_page1_request())
        msg = str(exc.value)
        assert "500" in msg
        assert "1" in msg
        assert SENTINEL_KEY not in msg
        assert sleeps == []
        assert _FakeChatHandler.connect_count == 1
    finally:
        server.close()


def test_302_not_followed():
    server = _FakeServer()
    try:
        backend = _backend(server)
        server.queue(
            302,
            b"",
            {"Location": "http://example.invalid/v1/chat/completions"},
        )
        with pytest.raises(DecisionBackendError) as exc:
            backend.decide(_page1_request())
        assert "302" in str(exc.value)
        assert SENTINEL_KEY not in str(exc.value)
        assert _FakeChatHandler.connect_count == 1
    finally:
        server.close()


def test_missing_and_empty_api_key_before_socket(monkeypatch: pytest.MonkeyPatch):
    for value in (None, ""):
        server = _FakeServer()
        try:
            if value is None:
                monkeypatch.delenv("PACKET_DECISION_API_KEY", raising=False)
            else:
                monkeypatch.setenv("PACKET_DECISION_API_KEY", value)
            backend = _backend(server, max_calls=10)
            before = _FakeChatHandler.connect_count
            with pytest.raises(DecisionBackendError) as exc:
                backend.decide(_page1_request())
            assert "PACKET_DECISION_API_KEY" in str(exc.value)
            assert _FakeChatHandler.connect_count == before
        finally:
            server.close()


def test_max_calls_zero_and_one():
    expected0 = json.loads((FIXTURES / "max_calls_0_expected.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=0)
        if expected0["raises_on_first_decide"]:
            with pytest.raises(DecisionBackendError):
                backend.decide(_page1_request())
    finally:
        server.close()

    expected1 = json.loads((FIXTURES / "max_calls_1_expected.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(server, max_calls=1, max_retries=2)
        for _ in range(3):
            server.queue(503, b"")
        if expected1["first_decide_ok"]:
            result = backend.decide(_page1_request())
            assert result.probability is None
        if expected1["second_decide_raises"]:
            with pytest.raises(DecisionBackendError):
                backend.decide(_page1_request())
    finally:
        server.close()


def test_page_text_request_fixture():
    expected = json.loads((FIXTURES / "http_request_page_text.json").read_text())
    facts = _load_facts()
    facts[1]["text"] = "a" * 4001
    server = _FakeServer()
    try:
        backend = _backend(server, input_mode="page_text")
        server.queue_json(200, _envelope("0.5"))
        backend.decide(
            DecisionRequest(
                task="boundary",
                page_index=1,
                facts=facts,
                blocks=[],
                ref=PageRef(packet_id="p", index=1),
                input_mode="page_text",
            )
        )
        assert _FakeChatHandler.captured[-1] == expected
    finally:
        server.close()


def test_image_request_fixture():
    expected = json.loads((FIXTURES / "http_request_image.json").read_text())
    png = (FIXTURES / "one_pixel.png").read_bytes()
    server = _FakeServer()
    try:
        backend = _backend(
            server,
            input_mode="default",
            send_images=True,
            capabilities={"structured_output": False, "logprobs": False, "images": True},
        )
        server.queue_json(200, _envelope("0.5"))
        backend.decide(_page1_request(), render=lambda: png)
        assert _FakeChatHandler.captured[-1] == expected
    finally:
        server.close()


def test_images_off_skips_render_thunk():
    expected = json.loads((FIXTURES / "http_request_images_off.json").read_text())
    server = _FakeServer()
    try:
        backend = _backend(server, input_mode="default", send_images=True)
        server.queue_json(200, _envelope("0.5"))
        called = False

        def _render() -> bytes:
            nonlocal called
            called = True
            return b""

        backend.decide(_page1_request(), render=_render)
        assert not called
        assert _FakeChatHandler.captured[-1] == expected
    finally:
        server.close()


def test_config_secret_rejections(tmp_path: Path):
    with pytest.raises(DecisionBackendError):
        load_decision_config(FIXTURES / "config_with_key.json")

    nested = tmp_path / "nested_token.json"
    nested.write_text(
        json.dumps(
            {
                "backend": "http",
                "base_url": "http://127.0.0.1:9/v1",
                "model": "m",
                "opts": {"Token": "secret"},
            }
        )
    )
    with pytest.raises(DecisionBackendError):
        load_decision_config(nested)

    userinfo = tmp_path / "userinfo.json"
    userinfo.write_text(
        json.dumps(
            {
                "backend": "http",
                "base_url": "http://key:secret@127.0.0.1:9/v1",
                "model": "m",
            }
        )
    )
    with pytest.raises(DecisionBackendError):
        load_decision_config(userinfo)

    query = tmp_path / "query.json"
    query.write_text(
        json.dumps(
            {
                "backend": "http",
                "base_url": "http://127.0.0.1:9/v1?api_key=leak",
                "model": "m",
            }
        )
    )
    with pytest.raises(DecisionBackendError):
        load_decision_config(query)


def test_tokenizer_key_allowed(tmp_path: Path):
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "backend": "http",
                "base_url": "http://127.0.0.1:9/v1",
                "model": "m",
                "tokenizer": "ok",
            }
        )
    )
    backend = load_decision_config(path)
    assert isinstance(backend, HttpDecisionBackend)


def test_logistic_config_userinfo_rejected(tmp_path: Path):
    path = tmp_path / "logistic_userinfo.json"
    path.write_text(
        json.dumps(
            {
                "backend": "logistic",
                "weights": "logistic_weights.json",
                "base_url": "http://user:pass@127.0.0.1:9/v1",
            }
        )
    )
    with pytest.raises(DecisionBackendError, match="userinfo"):
        load_decision_config(path)


def test_load_http_config(tmp_path: Path):
    data = {
        "backend": "http",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "local-model",
        "auth": "env",
        "max_calls": 5,
    }
    path = tmp_path / "http_config.json"
    path.write_text(json.dumps(data))
    backend = load_decision_config(path)
    assert isinstance(backend, HttpDecisionBackend)
    assert backend.max_calls == 5


def test_http_backend_from_loader_factory():
    backend = load_http_decision_backend(
        {
            "base_url": "http://127.0.0.1:1/v1",
            "model": "m",
        }
    )
    assert backend.model == "m"
