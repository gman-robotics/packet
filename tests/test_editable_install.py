"""GMA-219: ``pip install -e`` must not leave a copy of src/packet in site-packages.

A copy there shadows the editable finder, so ``import packet`` loads stale code instead
of the checkout. The static test runs in CI; the install tests build real venvs (they
fetch the build backend, so they are marked ``needs_network``).
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
import venv
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC_PKG = (REPO / "src" / "packet").resolve()


def test_pyproject_does_not_force_include_the_package():
    """``force-include`` is applied to editable builds too, copying src/packet next to the finder."""
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    wheel = cfg["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "src/packet" not in wheel.get("force-include", {}), "force-include copies src/packet into editable installs"
    assert wheel["packages"] == ["src/packet"]


def _venv(tmp: Path) -> Path:
    venv.EnvBuilder(with_pip=True, clear=True).create(tmp / "venv")
    return tmp / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"


@pytest.mark.needs_network
def test_editable_install_imports_the_checkout(tmp_path):
    py = _venv(tmp_path)
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO)], check=True)
    site = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                          check=True, capture_output=True, text=True).stdout.strip()
    assert not (Path(site) / "packet").exists(), f"stale copy of packet in {site}"
    out = subprocess.run([str(py), "-c", "import importlib.util as u; print(u.find_spec('packet').origin)"], cwd=tmp_path,
                         check=True, capture_output=True, text=True).stdout.strip()
    assert Path(out).resolve().is_relative_to(SRC_PKG), out


@pytest.mark.needs_network
def test_wheel_still_ships_the_package(tmp_path):
    py = _venv(tmp_path)
    subprocess.run([str(py), "-m", "pip", "wheel", "-q", "--no-deps", "-w", str(tmp_path / "dist"), str(REPO)],
                   check=True)
    (whl,) = (tmp_path / "dist").glob("packet_dl-*.whl")
    names = zipfile.ZipFile(whl).namelist()
    assert "packet/__init__.py" in names and "packet/cli.py" in names
    assert len(names) == len(set(names)), "duplicate entries in the wheel"
    assert not any(n.startswith("src/") for n in names)
