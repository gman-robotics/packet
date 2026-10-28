"""GMA-441 Exp A: per-slice table from score.py/_scores.json, public_rubric, md-f1, builtin eval and _summary.json.
usage: summarize.py OUT_JSON RUN_DIR...   (slices by page condition; prints markdown)"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYN = {p["name"]: p for p in json.loads((ROOT / "testset/synth/manifest.json").read_text())["pages"]}


def slice_of(name):
    if name.startswith("hs-"): return "hard-scan"
    if name.startswith("md-"): return "synth-degraded" if SYN[name]["scan"] else "synth-clean"
    if name.startswith("pk-scan"): return "blank"
    if name.startswith("pk-"): return "born-digital"
    return "public"


def agg(rows):
    def m(k):
        v = [r[k] for r in rows if r.get(k) is not None]; return round(sum(v) / len(v), 4) if v else None
    cers = [r["cer"] for r in rows if r.get("cer") is not None]
    return {"n": len(rows), "cer_mean": m("cer"), "cer_median": round(statistics.median(cers), 4) if cers else None,
            "wer_mean": m("wer"), "line_recall": m("line_recall"), "table_cell_recall": m("table_cell_recall"),
            "furniture_recall": m("furniture_recall"), "furniture_labeled": m("furniture_labeled"),
            "blank_chars": sum(r["blank_page_output_chars"] for r in rows if "blank_page_output_chars" in r) if any("blank_page_output_chars" in r for r in rows) else None}


def main():
    out, runs = Path(sys.argv[1]), [Path(x) for x in sys.argv[2:]]; res = {}
    for run in runs:
        sc = json.loads((run / "_scores.json").read_text())["pages"]
        sl = {}
        for r in sc: sl.setdefault(slice_of(r["page"]), []).append(r)
        summ = json.loads((run / "_summary.json").read_text())
        bi = json.loads((run / "_packet_eval_builtin.json").read_text())
        res[run.name] = {"overall": agg(sc), "slices": {k: agg(v) for k, v in sorted(sl.items())},
                         "public": json.loads((run / "_public_rubric.json").read_text()),
                         "md_boundary": json.loads((run / "_md_f1.json").read_text()),
                         "builtin_gate": bi["gate"], "builtin_f1": {f["name"]: f.get("boundary_f1", f.get("f1")) for f in bi["fixtures"]},
                         "pages_per_min": summ["pages_per_min"], "mlx_peak_gb": summ.get("mlx_peak_gb"),
                         "mps_driver_peak_gb": summ.get("mps_driver_peak_gb"), "truncated": summ.get("truncated")}
    out.write_text(json.dumps(res, indent=1))
    keys = ["cer_mean", "cer_median", "wer_mean", "line_recall", "table_cell_recall", "furniture_recall", "furniture_labeled", "blank_chars"]
    for scope in ["overall"] + sorted({s for r in res.values() for s in r["slices"]}):
        print(f"\n### {scope}\n| run | n | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 2))
        for name, r in res.items():
            a = r["overall"] if scope == "overall" else r["slices"].get(scope)
            if a: print(f"| {name} | {a['n']} | " + " | ".join(str(a[k]) for k in keys) + " |")


if __name__ == "__main__":
    main()
