"""control-packet materialize-fixtures: an eval gate failure (exit 1) must not abort
materialize when the builtin PDFs were written; a missing PDF still fails."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".cursor/skills/verify-packet/control-packet"
NAMES = ("text-layer", "multi-doc", "shared-header", "scan")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _fake_packet(home: Path, names: tuple[str, ...], exit_code: int) -> None:
    """A stand-in `packet` that writes the given builtin PDFs, then exits like a gate."""
    bindir = home / ".local" / "bin"
    bindir.mkdir(parents=True)
    writes = "\n".join(f'touch "$d/builtin/{n}.pdf"' for n in names)
    script = bindir / "packet"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'd=""; while [[ $# -gt 0 ]]; do [[ $1 == --dir ]] && d=$2; shift; done\n'
        'mkdir -p "$d/builtin"\n'
        f"{writes}\n"
        'echo "GATE FAIL"\n'
        f"exit {exit_code}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def _materialize(run_id: str, home: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PACKET_VERIFY_")}
    env.update(PACKET_VERIFY_RUN_ID=run_id, HOME=str(home),
               PATH=f"{home}/.local/bin:/usr/bin:/bin")
    return subprocess.run(["bash", str(SCRIPT), "materialize-fixtures"], env=env,
                          capture_output=True, text=True, timeout=60)


@pytest.fixture
def run_id():
    rid = f"t{uuid.uuid4().hex[:10]}"
    yield rid
    shutil.rmtree(Path("/tmp") / f"packet-verify-{rid}", ignore_errors=True)


def test_gate_failure_still_materializes_fixtures(run_id, tmp_path):
    _fake_packet(tmp_path, NAMES, exit_code=1)
    r = _materialize(run_id, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "cache_dir=" in r.stdout
    for n in NAMES:
        assert f"{n}.pdf" in r.stdout


def test_missing_pdf_still_fails(run_id, tmp_path):
    _fake_packet(tmp_path, NAMES[:3], exit_code=1)
    r = _materialize(run_id, tmp_path)
    assert r.returncode != 0
    assert "did not write builtin PDFs (exit 1)" in r.stderr
