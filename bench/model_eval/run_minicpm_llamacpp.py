"""GMA-441 Exp A: openbmb/MiniCPM-OCR-1.0 GGUF (F16 LLM + f16 4x mmproj) on Homebrew llama.cpp (llama-server, Metal).

Page-level use of the card's page prompt "Text Recognition:" (no PP-DocLayoutV3 layout stage), temperature 0.
The server is started by this script on 127.0.0.1 and stopped at the end; cold start = spawn -> /health ok.
usage: run_minicpm_llamacpp.py OUTDIR IMG...   (MAXTOK env, default 2048; GGUF_DIR env = snapshot dir)"""
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO, REV = "openbmb/MiniCPM-OCR-1.0-gguf", "674e62bf4ef7560ba502fe9b366a2780028aef20"
LLM, MMPROJ, PORT = "MiniCPM-OCR-1_0-F16.gguf", "mmproj-model-f16-4x.gguf", 18499
PROMPT = "Text Recognition:"


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start_server(gguf: Path, log: Path):
    t0 = time.time()
    cmd = ["/usr/bin/time", "-l", "llama-server", "--model", str(gguf / LLM), "--mmproj", str(gguf / MMPROJ),
           "--host", "127.0.0.1", "--port", str(PORT), "--ctx-size", "40960", "--n-gpu-layers", "999", "--jinja", "--temp", "0"]
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    while time.time() - t0 < 300:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200: return proc, time.time() - t0
        except OSError:
            if proc.poll() is not None: sys.exit(f"llama-server exited; see {log}")
            time.sleep(0.2)
    proc.kill(); sys.exit(f"llama-server not healthy after 300 s; see {log}")


def main():
    out, imgs = Path(sys.argv[1]), sys.argv[2:]; out.mkdir(parents=True, exist_ok=True)
    maxtok = int(os.environ.get("MAXTOK", "2048"))
    gguf = Path(os.environ.get("GGUF_DIR") or Path.home() / f".cache/huggingface/hub/models--{REPO.replace('/', '--')}/snapshots/{REV}")
    proc, cold = start_server(gguf, out / "_server.log")
    ver = subprocess.run(["llama-server", "--version"], check=False, capture_output=True, text=True).stderr.strip().splitlines()[-1]
    recs = []
    try:
        for p in imgs:
            name = Path(p).stem; mime = "image/jpeg" if p.endswith((".jpg", ".jpeg")) else "image/png"
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                                                               {"type": "text", "text": PROMPT}]}],
                    "temperature": 0, "max_tokens": maxtok}
            t = time.time(); r = post("/v1/chat/completions", body); dt = time.time() - t
            txt = r["choices"][0]["message"]["content"] or ""; fin = r["choices"][0].get("finish_reason")
            (out / f"{name}.txt").write_text(txt)
            u = r.get("usage", {})
            rec = {"page": name, "seconds": round(dt, 2), "gen_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"),
                   "truncated": fin == "length"}
            recs.append(rec); print(json.dumps(rec), flush=True)
    finally:
        proc.send_signal(signal.SIGINT); proc.wait(timeout=60)
    srv = (out / "_server.log").read_text()
    foot = [int(line.split()[0]) for line in srv.splitlines() if "peak memory footprint" in line]
    g = sum(x["seconds"] for x in recs)
    s = {"model": REPO, "revision": REV, "files": [LLM, MMPROJ], "prompt": PROMPT, "backend": f"llama.cpp (Homebrew) {ver}, Metal, -ngl 999",
         "dtype": "F16 GGUF + f16 mmproj", "server_spawn_to_health_s": round(cold, 2), "pages": len(recs), "total_gen_s": round(g, 1),
         "pages_per_min": round(60 * len(recs) / max(g, 1e-9), 2), "truncated": sum(x["truncated"] for x in recs),
         "server_peak_footprint_gb": round(foot[0] / 1e9, 2) if foot else None, "per_page": recs}
    (out / "_summary.json").write_text(json.dumps(s, indent=1))
    print("SUMMARY", json.dumps({k: v for k, v in s.items() if k != "per_page"}))


if __name__ == "__main__":
    main()
