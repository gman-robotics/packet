"""Time native extract vs boundary observe vs lookahead on dlnet-whole.

Uses ``Pipeline()``'s default ``NativeTextExtractor`` only (no OCR). Prints
three wall-clock seconds for one full dlnet-whole pass.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from packet.boundary import BoundaryDetector
from packet.classify import HeuristicClassifier
from packet.eval.run import DEFAULT_WHOLEDOC_CACHE
from packet.eval.wholedoc import build_fixtures, fetch, load_manifest
from packet.extract import NativeTextExtractor, Router
from packet.furniture import FurnitureTagger
from packet.lookahead import LookaheadSplitter, PageObs
from packet.pipeline import PipelineConfig, _lazy_lines, _lazy_render
from packet.source import PdfPageSource


def profile_dlnet_whole(cache_dir: Path, build_dir: Path) -> dict[str, float]:
    manifest = load_manifest()
    fetch(manifest, cache=cache_dir)
    fixtures = list(build_fixtures(manifest, cache_dir, build_dir))

    classifier = HeuristicClassifier()
    router = Router([NativeTextExtractor()])
    tagger = FurnitureTagger()
    config = PipelineConfig()

    extract_s = 0.0
    observe_s = 0.0
    lookahead_s = 0.0
    pages = 0

    for fixture in fixtures:
        detector = BoundaryDetector()
        splitter = LookaheadSplitter(
            window=config.lookahead_window,
            commit=config.boundary_commit_confidence,
            fallback=config.boundary_min_confidence,
        )
        with PdfPageSource(fixture.pdf) as source:
            for ref in source.pages():
                pages += 1
                text = source.text_for(ref.index)

                t0 = time.perf_counter()
                classified = classifier.classify(ref, text)
                blocks = tagger.tag_page(
                    classified,
                    router.extract(
                        classified,
                        text,
                        image=_lazy_render(source, classified.index),
                        layout=_lazy_lines(source, classified.index),
                    ),
                )
                extract_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                signal = detector.observe(classified, blocks)
                observe_s += time.perf_counter() - t0

                obs = PageObs(
                    classified.index,
                    signal,
                    detector.last_page_number,
                    detector.last_page_total,
                )
                t0 = time.perf_counter()
                for _ in splitter.push(obs):
                    pass
                lookahead_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in splitter.flush():
                pass
            lookahead_s += time.perf_counter() - t0

    return {
        "pages": float(pages),
        "extract_seconds": extract_s,
        "observe_seconds": observe_s,
        "lookahead_seconds": lookahead_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_WHOLEDOC_CACHE,
        help="dlnet-whole PDF cache (default: ~/.cache/packet/dlnet-whole)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(".packet-eval/dlnet-whole"),
        help="joined packet PDF output directory",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    stats = profile_dlnet_whole(args.cache, args.build_dir)
    stats["wall_seconds"] = time.perf_counter() - started
    stats["machine"] = platform.node()
    stats["platform"] = platform.platform()
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
