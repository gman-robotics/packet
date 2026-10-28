"""GMA-445 Exp A2: openbmb/MiniCPM-OCR-Jev @90bc4c47 region scorer on transformers + torch MPS (native qwen3_5, no remote code).

Prompt bytes and the 2-class value head follow upstream OpenBMB/MiniCPM-OCR@458718f8 scripts/minicpm-ocr-jev/infer_vllm.py
(md5 of build_prompt("{text}") asserted below). The value head is `score.weight` [2, hidden] applied to the final
normed hidden state of the last prompt token; prob_yes = softmax(logits)[1].
Rendering: upstream renders candidates with Chromium + KaTeX. This harness uses a plain PIL text renderer
(render_text) instead, so absolute scores can drift; parity is checked on the upstream examples (see `parity`).
Run only from the sandbox venv /Users/Shared/gman-sandbox/lfm3-mps/venv with HF_HUB_OFFLINE=1."""
import hashlib
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO, REV = "openbmb/MiniCPM-OCR-Jev", "90bc4c4768c082f769f5352479fa0df84da42c1a"
SANDBOX = "/Users/Shared/gman-sandbox/lfm3-mps/venv"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
INSTRUCTION = ("第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
               "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
               "这是一个二分类任务：一致标为 1，不一致标为 0。")
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT_TEMPLATE_MD5 = "9aab68e8ba8ab4da17ba9752bddb199c"
MAX_ASPECT_RATIO = 199.0


def build_prompt(text: str) -> str:
    return (f"<|im_start|>user\n{IMAGE_PLACEHOLDER}\nOCR文本：\n{text}\n{IMAGE_PLACEHOLDER}\n{INSTRUCTION}{ASSISTANT_SUFFIX}")


if hashlib.md5(build_prompt("{text}").encode("utf-8")).hexdigest() != PROMPT_TEMPLATE_MD5:
    raise RuntimeError("Jev prompt template drifted from upstream md5")


def pad_ratio(img: Image.Image) -> Image.Image:
    """Upstream load_image: pad (never resize) images past 199:1 with white."""
    img = img.convert("RGB"); w, h = img.size
    if max(w, h) / max(1, min(w, h)) <= MAX_ASPECT_RATIO: return img
    nw, nh = (w, max(h, int(w / MAX_ASPECT_RATIO + 0.999999))) if w >= h else (max(w, int(h / MAX_ASPECT_RATIO + 0.999999)), h)
    c = Image.new("RGB", (nw, nh), "white"); c.paste(img, ((nw - w) // 2, (nh - h) // 2)); return c


_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def render_markup(text: str) -> str:
    """Markdown/HTML -> visible text, as a browser would show it: drop heading/emphasis marks, tables -> tab-joined rows."""
    import html as _h
    t = re.sub(r"(?is)</t[dh]>\s*", "\t", text); t = re.sub(r"(?is)</tr>\s*", "\n", t); t = re.sub(r"(?s)<[^>]+>", "", t)
    t = re.sub(r"(?m)^\s*#{1,6}\s+", "", t); t = re.sub(r"\*\*|__", "", t); t = re.sub(r"(?m)^\s*[-*]\s+", "\u2022 ", t)
    return "\n".join(l.rstrip("\t ") for l in _h.unescape(t).split("\n") if l.strip() or not l)


def render_text(text: str, width: int = 1000, size: int = 28) -> Image.Image:
    """Plain black-on-white render of candidate text (approximation of the upstream Chromium/KaTeX renderer)."""
    font = ImageFont.truetype(_FONT, size); cols = max(20, int(width / (size * 0.5)))
    lines = []
    for para in (render_markup(text) or " ").replace("\t", "    ").split("\n"):
        lines += textwrap.wrap(para, cols) or [""]
    lh = int(size * 1.35); img = Image.new("RGB", (width, max(lh, lh * len(lines) + 2 * size)), "white"); d = ImageDraw.Draw(img)
    for i, line in enumerate(lines): d.text((size, size + i * lh), line, fill="black", font=font)
    return img


class Jev:
    def __init__(self):
        if os.path.realpath(sys.prefix) != os.path.realpath(SANDBOX) or os.environ.get("HF_HUB_OFFLINE") != "1":
            sys.exit(f"run from {SANDBOX} with HF_HUB_OFFLINE=1")
        import torch
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.torch = torch; t = time.time()
        snap = snapshot_download(REPO, revision=REV, local_files_only=True)
        m = Qwen3_5ForConditionalGeneration.from_pretrained(snap, local_files_only=True, dtype=torch.bfloat16)
        self.model = m.model.to("mps").eval()  # backbone only; lm_head is not in the checkpoint
        with safe_open(f"{snap}/model.safetensors", "pt") as f:
            self.head = f.get_tensor("score.weight").to("mps", torch.float32)  # [2, hidden], no bias
        self.proc = AutoProcessor.from_pretrained(snap, local_files_only=True)
        torch.mps.synchronize(); self.load_s = time.time() - t

    def score(self, original: Image.Image, render: Image.Image, text: str) -> float:
        torch = self.torch
        inp = self.proc(text=[build_prompt(text)], images=[pad_ratio(original), pad_ratio(render)], return_tensors="pt").to("mps")
        with torch.inference_mode():
            h = self.model(**inp).last_hidden_state[0, -1].float()
            logits = self.head @ h
        return torch.softmax(logits, -1)[1].item()


def parity(root: str) -> None:
    """Upstream examples: score with the upstream reference render and with render_text; print vs observed."""
    j = Jev(); print("load_s", round(j.load_s, 2))
    for case in sorted(Path(root).glob("*/case.json")):
        c = json.loads(case.read_text()); blk = Image.open(case.parent / c["block_image"])
        for cand in c["candidates"]:
            row = {"case": c["case_id"], "role": cand["role"], "observed": c["observed_prob_yes"].get(cand["role"]),
                   "pil_render": round(j.score(blk, render_text(cand["text"], blk.width), cand["text"]), 5)}
            if c.get("reference_render_of") == cand["role"]:
                row["ref_render"] = round(j.score(blk, Image.open(case.parent / c["reference_render"]), cand["text"]), 5)
            print(json.dumps(row, ensure_ascii=False), flush=True)


# ---- regions -----------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
TAG = re.compile(r"!\[([a-z_]+)\+?\]\((\d+),(\d+),(\d+),(\d+)\)")
NO_TEXT = {"image", "header_image", "footer_image"}  # picture regions: nothing to verify, never scored or dropped
SKIP_PAGE_SCORE = 12000  # chars; page-level candidates longer than this are recorded unscored (latency guard)


def regions(raw: str):
    """Split LightOn grounding output into (label, bbox_0_1000, text) blocks; [] when untagged."""
    ms = list(TAG.finditer(raw)); out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(raw)
        out.append((m.group(1), tuple(int(m.group(k)) for k in range(2, 6)), raw[m.end():end].strip()))
    return out


def crop(img: Image.Image, bb, pad: int = 6) -> Image.Image:
    w, h = img.size; x0, y0, x1, y1 = bb
    box = (max(0, int(x0 * w / 1000) - pad), max(0, int(y0 * h / 1000) - pad), min(w, int(x1 * w / 1000) + pad), min(h, int(y1 * h / 1000) + pad))
    return img.crop(box) if box[2] > box[0] + 2 and box[3] > box[1] + 2 else img


def page_text(raw: str) -> str:
    return TAG.sub("", raw).strip()


def score_page(j: Jev, img: Image.Image, raw: str, page_level: bool = True):
    """Rows for one page: every grounding block (region-level) plus the whole page (page-level)."""
    rows = []
    for k, (lab, bb, txt) in enumerate(regions(raw)):
        if lab in NO_TEXT or not txt:
            rows.append({"kind": "block", "i": k, "label": lab, "bbox": bb, "prob": None, "chars": len(txt)}); continue
        c = crop(img, bb); t = time.perf_counter()
        p = j.score(c, render_text(txt, max(400, min(1400, c.width))), txt)
        rows.append({"kind": "block", "i": k, "label": lab, "bbox": bb, "prob": p, "chars": len(txt), "s": time.perf_counter() - t})
    if page_level:
        txt = page_text(raw); t = time.perf_counter()
        if len(txt) > SKIP_PAGE_SCORE:
            rows.append({"kind": "page", "prob": None, "chars": len(txt), "skipped": "too_long"})
        else:
            p = j.score(img, render_text(txt or " ", 1000), txt)
            rows.append({"kind": "page", "prob": p, "chars": len(txt), "s": time.perf_counter() - t})
    return rows


def testset(runs):
    """Dev split: score every page of each Exp A testset run; writes RUN/_jev.json."""
    j = Jev(); img_of = {}
    for d in ("packet", "synth", "public"):
        for f in (ROOT / "testset" / d).iterdir():
            if f.suffix in (".png", ".jpeg", ".jpg"): img_of[f.stem] = f
    for run in map(Path, runs):
        out, t0 = {}, time.perf_counter()
        for f in sorted(run.glob("*.txt")):
            if f.stem not in img_of: continue
            out[f.stem] = score_page(j, Image.open(img_of[f.stem]).convert("RGB"), f.read_text())
        wall = time.perf_counter() - t0
        (run / "_jev.json").write_text(json.dumps({"repo": REPO, "rev": REV, "renderer": "pil", "wall_s": wall, "load_s": j.load_s, "pages": out}, indent=1))
        n = sum(1 for v in out.values() for r in v if r["prob"] is not None)
        print(json.dumps({"run": str(run), "pages": len(out), "scored": n, "wall_s": round(wall, 1), "pages_per_min": round(60 * len(out) / wall, 2)}), flush=True)


def politax(data: str, out: str, stride: int = 1, page_level: bool = True):
    """Holdout (report only): score OCR records in DATA/ocr; resumable jsonl at OUT. Never trains on these rows."""
    sys.path.insert(0, str(ROOT / "src"))
    from packet.eval import politax as pt
    from packet.eval.politax_split import holdout_file_names
    from packet.source import PdfPageSource
    data, outp = Path(data).expanduser(), Path(out).expanduser()
    m = pt.load_manifest(); hold = holdout_file_names(m)
    done = {json.loads(l)["key"] for l in outp.read_text().splitlines()} if outp.exists() else set()
    j = Jev(); n = 0; t0 = time.perf_counter()
    with outp.open("a") as fo:
        for d in m.docs:
            if d.file_name not in hold: continue
            recs = sorted((data / "ocr" / d.sha256).glob("p*.json"))
            src = None
            for k, f in enumerate(recs):
                key = f"{d.sha256}/{f.name}"
                if k % stride or key in done: continue
                if src is None: src = PdfPageSource(Path.home() / ".cache/packet/politax" / d.file_name)
                rec = json.loads(f.read_text()); idx = int(f.stem[1:])
                img = src.render(idx, longest_px=1540).convert("RGB"); t = time.perf_counter()
                rows = score_page(j, img, rec["raw"], page_level)
                fo.write(json.dumps({"key": key, "file": d.file_name, "page": idx, "s": time.perf_counter() - t, "rows": rows}) + "\n"); fo.flush(); n += 1
            if src is not None: src.close()
    print(json.dumps({"pages": n, "wall_s": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "parity": parity(a[0])
    elif cmd == "testset": testset(a)
    elif cmd == "politax": politax(a[0], a[1], int(os.environ.get("STRIDE", "1")), os.environ.get("PAGE_LEVEL", "1") == "1")
    else: sys.exit(__doc__)
