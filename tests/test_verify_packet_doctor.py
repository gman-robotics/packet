"""control-packet doctor: site-packages force-include copy must match src/packet entirely."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".cursor/skills/verify-packet/control-packet"
REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _imported_packet_dir() -> Path:
    """Where `import packet` resolves without pytest's src prepended."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import packet; from pathlib import Path; print(Path(packet.__file__).resolve().parent)",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip())


def test_doctor_fails_when_site_packages_copy_diverges_from_src():
    imported = _imported_packet_dir()
    if imported == (REPO / "src" / "packet").resolve():
        pytest.skip("no site-packages shadow copy in this environment")
    assert "site-packages" in str(imported)

    cli = imported / "cli.py"
    backup = cli.read_text(encoding="utf-8")
    cli.write_text(backup + "\n# stale probe\n", encoding="utf-8")
    run_id = f"doctor-stale-{uuid.uuid4().hex[:8]}"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PACKET_VERIFY_RUN_ID"] = run_id
    try:
        proc = subprocess.run(
            ["bash", str(SCRIPT), "doctor"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "FAIL" in combined
    finally:
        cli.write_text(backup, encoding="utf-8")
        shutil.rmtree(Path("/tmp") / f"packet-verify-{run_id}", ignore_errors=True)
