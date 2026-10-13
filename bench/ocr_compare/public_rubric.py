"""GMA-207 documented manual rubric for public samples without shipped ground truth.
Receipt (SROIE fixture): 9 key fields transcribed by hand from the image -> field hit rate.
Newspaper (PaddleOCR demo, 4-col Chinese): 7 anchors in true reading order -> hit rate + order correctness (Kendall-style: fraction of
anchor pairs found in correct relative order)."""
import json, re, sys, itertools
from pathlib import Path
RECEIPT = ["TD01167104", "25/12/2018", "8:13:39 PM", "MANIS", "CASH BILL", "9556939040118", "KF MODELLING CLAY KIDDY FISH", "9.000", "9.00"]
NEWS = ["助力双方交往", "身着中国传统民族服装", "学好中文", "在中国学习的经历", "中国驻厄立特里亚大使馆", "不远万里都是客人", "共同向世界展示非"]
def n(t): return re.sub(r"\s+", " ", re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)", " ", t))
def nz(t): return re.sub(r"\s+", "", n(t))
for d in sys.argv[1:]:
    d = Path(d); res = {}
    if (d / "sroie-receipt.txt").exists():
        t = n((d / "sroie-receipt.txt").read_text()); res["receipt_fields"] = round(sum(1 for f in RECEIPT if f in t) / len(RECEIPT), 3)
        res["receipt_missed"] = [f for f in RECEIPT if f not in t]
    if (d / "paddle-demo.txt").exists():
        t = nz((d / "paddle-demo.txt").read_text()); pos = [t.find(a) for a in NEWS]; found = [p for p in pos if p >= 0]
        pairs = list(itertools.combinations([p for p in pos if p >= 0], 2))
        res["news_anchor_hits"] = round(len(found) / len(NEWS), 3)
        res["news_order_pairs_correct"] = round(sum(1 for a, b in pairs if a < b) / len(pairs), 3) if pairs else None
        res["news_chars_out"] = len(t)
    (d / "_public_rubric.json").write_text(json.dumps(res, ensure_ascii=False)); print(d, json.dumps(res, ensure_ascii=False))
