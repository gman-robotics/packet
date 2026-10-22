"""GMA-275 guards: CI and ``packet eval --set politax`` never read the OCR cache or
import OCR extras. These run in CI (no OCR extras installed)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from packet.eval.pdfutil import write_blank_scan_pdf, write_text_pdf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BLOCKED = ("mlx", "mlx_vlm", "huggingface_hub", "paddleocr", "paddle", "packet.eval.politax_ocr")

_CHILD = r"""
import json, sys
sys.path.insert(0, {src!r})
BLOCKED = {blocked!r}
attempts, opens = [], []
import os
OCR_ROOTS = (os.path.join({data!r}, "ocr"), os.path.join(os.path.realpath({data!r}), "ocr"))

def under_ocr(p):
    p = os.path.abspath(p)
    return any(p == r or p.startswith(r + os.sep) for r in OCR_ROOTS)

class Recorder:
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            attempts.append(name)
            raise ImportError("blocked by guard: " + name)
        return None

sys.meta_path.insert(0, Recorder())

def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if under_ocr(p):
            opens.append(p)
    if event in ("os.listdir", "os.scandir") and args and args[0]:
        p = os.fsdecode(args[0]) if not isinstance(args[0], int) else ""
        if p and under_ocr(p):
            opens.append(p)

sys.addaudithook(hook)
from pathlib import Path
from packet.eval import politax
politax.MANIFEST = Path({manifest!r})
from packet import cli
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = cli.main(["eval", "--set", "politax", "--data-dir", {data!r}, "--json"])
loaded = [m for m in sys.modules if any(m == b or m.startswith(b + ".") for b in BLOCKED)]
print(json.dumps({{"rc": rc, "attempts": attempts, "opens": opens, "loaded": loaded,
                  "out": buf.getvalue()}}))
"""


def _corpus(data: Path) -> Path:
    data.mkdir(parents=True)
    docs, ann = [], []
    specs = [("t1.pdf", "trump", 2, True), ("t2.pdf", "trump", 3, False), ("o1.pdf", "obama", 2, True)]
    for name, subject, pages, text in specs:
        p = data / name
        if text:
            write_text_pdf(p, [[f"{name} page {k} " + "word " * 12] for k in range(pages)])
        else:
            write_blank_scan_pdf(p, pages)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        docs.append({"subject": subject, "file_name": name, "file": name,
                     "url": f"https://example.invalid/{name}", "pages": pages, "sha256": sha,
                     "license": "test", "gold_cuts": 0})
        ann.append({"file_name": name, "page_count": pages,
                    "sub_documents": [{"type": "f", "start_page": 1, "end_page": pages}]})
    ap = data / "annotations.jsonl"
    ap.write_text("\n".join(json.dumps(a) for a in ann) + "\n")
    manifest = {"name": "t", "version": 1, "license_note": "CC0 test",
                "annotations": {"url": "https://example.invalid/a", "file": ap.name,
                                "sha256": hashlib.sha256(ap.read_bytes()).hexdigest(),
                                "license": "CC0-1.0"},
                "docs": docs}
    mp = data.parent / "manifest.json"
    mp.write_text(json.dumps(manifest))
    return mp


def _poison(data: Path) -> None:
    for d in json.loads((data.parent / "manifest.json").read_text())["docs"]:
        rec = data / "ocr" / d["sha256"]
        rec.mkdir(parents=True)
        (rec / "p00000.json").write_text("{garbage")
    locked = data / "ocr" / "locked"
    locked.mkdir()
    locked.chmod(0)


def _eval_child(manifest: Path, data: Path) -> dict:
    code = _CHILD.format(src=str(SRC), blocked=BLOCKED, manifest=str(manifest), data=str(data))
    proc = subprocess.run([sys.executable, "-I", "-c", code], check=False, capture_output=True, text=True,
                          timeout=300, cwd=str(data.parent))
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_eval_politax_never_reads_cache_or_tries_ocr_imports(tmp_path):
    data = tmp_path / "politax"
    manifest = _corpus(data)
    clean = _eval_child(manifest, data)
    _poison(data)
    try:
        poisoned = _eval_child(manifest, data)
    finally:
        (data / "ocr" / "locked").chmod(0o700)
    for res in (clean, poisoned):
        assert res["rc"] == 0
        assert res["attempts"] == [], res["attempts"]
        assert res["loaded"] == []
        assert res["opens"] == [], res["opens"]
    assert json.loads(poisoned["out"]) == json.loads(clean["out"])


def _politax_ocr_refs(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        if any(n.endswith("politax_ocr") for n in names):
            out.append((path.name, node.lineno))
    return out


def test_only_the_politax_ocr_cli_handler_imports_the_cache_module():
    cli = SRC / "packet" / "cli.py"
    allowed = set()
    for node in ast.walk(ast.parse(cli.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "_cmd_politax_ocr":
            allowed = set(range(node.lineno, node.end_lineno + 1))
    assert allowed, "_cmd_politax_ocr missing"
    for py in (SRC / "packet").rglob("*.py"):
        if py.name == "politax_ocr.py":
            continue
        for name, line in _politax_ocr_refs(py):
            assert py == cli and line in allowed, f"{name}:{line} imports politax_ocr"


def test_cache_module_has_no_top_level_engine_imports():
    tree = ast.parse((SRC / "packet" / "eval" / "politax_ocr.py").read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or node.names[0].name
            assert not mod.split(".")[0] in {"mlx", "mlx_vlm", "huggingface_hub", "paddleocr", "PIL"}


def test_ocr_extras_stay_optional_and_out_of_ci():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    base = " ".join(py["project"].get("dependencies", []))
    dev = " ".join(py["project"]["optional-dependencies"]["dev"])
    for dep in ("mlx", "mlx-vlm", "huggingface_hub", "paddleocr"):
        assert not re.search(rf"(^|[\s\"']){re.escape(dep)}\b", base + " " + dev)
    install = re.compile(r"pip install[^\n]*\[[^\]\n]*ocr-(mlx|paddle)")
    for wf in (ROOT / ".github" / "workflows").glob("*.y*ml"):
        assert not install.search(wf.read_text()), f"{wf.name} installs OCR extras"


def test_data_dir_inside_repo_is_refused(tmp_path):
    from packet import cli

    assert cli.main(["politax-ocr", "--data-dir", str(ROOT / "tmp-ocr"), "--dry-run"]) == 2
    link = tmp_path / "link"
    try:
        os.symlink(ROOT, link)
    except OSError:
        pytest.skip("no symlinks")
    assert cli.main(["politax-ocr", "--data-dir", str(link / "x"), "--dry-run"]) == 2
    assert not (ROOT / "tmp-ocr").exists()
