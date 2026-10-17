"""control-packet cleanup: PACKET_VERIFY_DELETE_ROOT=1 must work for a marked run root
and must still refuse anything outside /tmp/packet-verify-<run-id>."""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".cursor/skills/verify-packet/control-packet"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PACKET_VERIFY_")}
    env.update(extra)
    return env


def _make_run(run_id: str) -> Path:
    root = Path("/tmp") / f"packet-verify-{run_id}"
    (root / "cache" / "builtin").mkdir(parents=True)
    (root / "cache" / "builtin" / "x.pdf").write_text("x")
    (root / "scratch").mkdir()
    (root / "scratch" / "s.txt").write_text("s")
    (root / "other").mkdir()
    (root / "artifacts").mkdir()
    (root / "artifacts" / "proof.txt").write_text("keep me")
    (root / ".packet-verify-owned").write_text(f"run_id={run_id}\ncreated_by=control-packet\n")
    return root


def _cleanup(run_id: str, **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "cleanup"],
        check=False,
        env=_env(PACKET_VERIFY_RUN_ID=run_id, **extra),
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def run_id():
    rid = f"t{uuid.uuid4().hex[:10]}"
    yield rid
    shutil.rmtree(Path("/tmp") / f"packet-verify-{rid}", ignore_errors=True)


def test_delete_root_removes_everything_but_artifacts(run_id):
    root = _make_run(run_id)
    r = _cleanup(run_id, PACKET_VERIFY_DELETE_ROOT="1")
    assert r.returncode == 0, r.stderr
    assert not (root / "cache").exists()
    assert not (root / "scratch").exists()
    assert not (root / "other").exists()
    assert (root / "artifacts" / "proof.txt").read_text() == "keep me"


def test_delete_root_when_only_cache_and_artifacts(run_id):
    root = _make_run(run_id)
    shutil.rmtree(root / "scratch")
    shutil.rmtree(root / "other")
    r = _cleanup(run_id, PACKET_VERIFY_DELETE_ROOT="1")
    assert r.returncode == 0, r.stderr
    assert not (root / "cache").exists()
    assert (root / "artifacts" / "proof.txt").exists()


def test_delete_root_refuses_unmarked_root(run_id):
    root = _make_run(run_id)
    (root / ".packet-verify-owned").unlink()
    r = _cleanup(run_id, PACKET_VERIFY_DELETE_ROOT="1")
    assert r.returncode != 0
    assert (root / "other").exists()


def test_delete_root_refuses_root_outside_prefix(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "sentinel").write_text("s")
    (victim / ".packet-verify-owned").write_text("run_id=x\n")
    r = subprocess.run(
        ["bash", str(SCRIPT), "cleanup"],
        check=False,
        env=_env(PACKET_VERIFY_ROOT=str(victim), PACKET_VERIFY_DELETE_ROOT="1"),
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode != 0
    assert (victim / "sentinel").exists()


def test_bare_prefix_is_never_a_root():
    r = subprocess.run(
        ["bash", str(SCRIPT), "cleanup"],
        check=False,
        env=_env(PACKET_VERIFY_ROOT="/tmp/packet-verify-", PACKET_VERIFY_DELETE_ROOT="1"),
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode != 0
