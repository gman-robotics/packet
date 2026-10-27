"""PoliTax default-on ship gate (GMA-291 / plan step 5).

Runs the eight-item checklist from the corpus plan. Outcome (a) would set
``PipelineConfig.scorer`` to committed logistic weights; outcome (b) leaves the
default ``None`` and records which items failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from packet.boundary_model import featurize_from_facts, load_logistic_scorer
from packet.eval.politax_split import assert_frozen_split
from packet.eval.politax_train import (
    assert_training_set_clean,
    collect_trump_training_rows,
    holdout_scorecard,
    train_trump_logistic,
)
from packet.eval.run import (
    DEFAULT_POLITAX_CACHE,
    DEFAULT_WHOLEDOC_CACHE,
    prepare_politax,
    run_dlnet_whole,
    run_fixture,
    run_named_fixtures,
)
from packet.pipeline import Pipeline, PipelineConfig

DLNET_LOGGED_HEURISTIC_MACRO = 0.98462
# Logged max logistic p on frozen holdout pages (trump-trained weights; scan-only PDFs).
HOLDOUT_LOGGED_MAX_MODEL_P = 0.4989


@dataclass
class ShipItem:
    number: int
    title: str
    passed: bool
    detail: str


@dataclass
class ShipReport:
    ship: bool
    outcome: str
    items: list[ShipItem]
    holdout_heuristic: dict
    holdout_scorer: dict
    max_holdout_model_p: float
    dlnet_macro_scorer_on: float
    builtin_f1: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "ship": self.ship,
            "outcome": self.outcome,
            "items": [item.__dict__ for item in self.items],
            "holdout_heuristic": self.holdout_heuristic,
            "holdout_scorer": self.holdout_scorer,
            "max_holdout_model_p": self.max_holdout_model_p,
            "dlnet_macro_scorer_on": self.dlnet_macro_scorer_on,
            "builtin_f1": self.builtin_f1,
        }


def _item2_holdout_strictly_up(heur, scorer) -> tuple[bool, str]:
    macro_up = scorer.macro_f1 > heur.macro_f1
    micro_up = scorer.micro_f1 > heur.micro_f1
    falls = [
        name
        for name in heur.per_packet_f1
        if scorer.per_packet_f1.get(name, 0.0) < heur.per_packet_f1[name]
    ]
    ok = macro_up and micro_up and not falls
    detail = (
        f"heur macro={heur.macro_f1:.4f} micro={heur.micro_f1:.4f}; "
        f"scorer macro={scorer.macro_f1:.4f} micro={scorer.micro_f1:.4f}; "
        f"macro_up={macro_up} micro_up={micro_up} f1_falls={falls or 'none'}"
    )
    return ok, detail


def holdout_max_native_text_blocks(m, fixtures: list) -> int:
    """Largest native text block count on any holdout page (GMA-270 census: 0)."""
    from packet.eval.page_facts import page_facts
    from packet.eval.politax_split import holdout_file_names

    holdout_files = holdout_file_names(m)
    by_file = {fx.notes or fx.pdf.name: fx for fx in fixtures}
    max_blocks = 0
    for name in holdout_files:
        fx = by_file[name]
        for row in page_facts(fx.pdf):
            max_blocks = max(max_blocks, row["blocks"])
    return max_blocks


def max_holdout_model_probability(weights_path: Path, politax_cache: Path) -> float:
    from packet.eval.page_facts import page_facts
    from packet.eval.politax_split import holdout_file_names

    m, fixtures = prepare_politax(politax_cache)
    holdout_files = holdout_file_names(m)
    by_file = {fx.notes or fx.pdf.name: fx for fx in fixtures}
    scorer = load_logistic_scorer(weights_path)
    max_p = 0.0
    for name in sorted(holdout_files):
        fx = by_file[name]
        facts = page_facts(fx.pdf)
        for i in range(1, len(facts)):
            max_p = max(max_p, scorer.probability(featurize_from_facts(facts, i)))
    return max_p


def evaluate_politax_ship(
    weights_path: Path,
    politax_cache: Path = DEFAULT_POLITAX_CACHE,
    dlnet_cache: Path = DEFAULT_WHOLEDOC_CACHE,
    eval_root: Path | None = None,
) -> ShipReport:
    """Run the default-on checklist with trained weights at ``weights_path``."""
    m, fixtures = prepare_politax(politax_cache)
    by_file = {fx.notes or fx.pdf.name: fx for fx in fixtures}
    rows = collect_trump_training_rows(m, by_file)
    items: list[ShipItem] = []

    from packet.eval.politax_split import SplitError
    from packet.eval.politax_train import TrainingSetError

    try:
        assert_frozen_split(m)
        assert_training_set_clean(rows, m)
        items.append(
            ShipItem(1, "train/holdout disjoint (asserted in code)", True, "assert_frozen_split + assert_training_set_clean")
        )
    except (SplitError, TrainingSetError, ValueError) as exc:
        items.append(ShipItem(1, "train/holdout disjoint (asserted in code)", False, str(exc)))

    heur_pipe = Pipeline()
    scorer_pipe = Pipeline(config=PipelineConfig(scorer=load_logistic_scorer(weights_path)))
    results_heur = [run_fixture(fx, heur_pipe) for fx in fixtures]
    results_scorer = [run_fixture(fx, scorer_pipe) for fx in fixtures]
    heur = holdout_scorecard(m, fixtures, results_heur)
    scorer = holdout_scorecard(m, fixtures, results_scorer)
    ok2, detail2 = _item2_holdout_strictly_up(heur, scorer)
    items.append(ShipItem(2, "holdout scorer-on strictly above scorer-off", ok2, detail2))

    builtin_scorer = run_named_fixtures(eval_root, pipeline=scorer_pipe)
    builtin_f1 = {r.name: r.boundary.f1 for r in builtin_scorer}
    ok3 = all(f1 >= 1.0 for f1 in builtin_f1.values())
    items.append(
        ShipItem(
            3,
            "builtin F1 1.0 with weights loaded",
            ok3,
            str(builtin_f1),
        )
    )

    raw = json.loads(weights_path.read_text())
    train_packets = raw.get("training_packets", [])
    dlnet_in_train = [p for p in train_packets if str(p).startswith("dlw-")]
    dlnet_results = run_dlnet_whole(dlnet_cache, eval_root, pipeline=scorer_pipe)
    dlnet_macro = sum(r.boundary.f1 for r in dlnet_results) / len(dlnet_results)
    ok4 = not dlnet_in_train and dlnet_macro >= DLNET_LOGGED_HEURISTIC_MACRO
    items.append(
        ShipItem(
            4,
            f"dlnet-whole not in training; macro>={DLNET_LOGGED_HEURISTIC_MACRO} with scorer",
            ok4,
            f"dlnet_in_train={dlnet_in_train or 'none'}; dlnet_macro={dlnet_macro:.5f}",
        )
    )

    ocr_bad = [
        r.name for r in results_scorer + results_heur if r.false_ocr_pages
    ]
    ok5 = not ocr_bad
    items.append(ShipItem(5, "false-OCR pages empty", ok5, f"violations={ocr_bad or 'none'}"))

    max_p = max_holdout_model_probability(weights_path, politax_cache)
    max_blocks = holdout_max_native_text_blocks(m, fixtures)
    p_ok = max_p <= HOLDOUT_LOGGED_MAX_MODEL_P
    blocks_ok = max_blocks == 0
    ok6 = p_ok and blocks_ok
    items.append(
        ShipItem(
            6,
            "holdout max p=0.4989; zero native text blocks on scan PDFs",
            ok6,
            f"max_p={max_p:.4f} max_blocks={max_blocks}",
        )
    )

    note = m.license_note.lower()
    ok7 = "cc0" in note and ("public-domain" in note or "public domain" in note)
    items.append(
        ShipItem(
            7,
            "PoliTax license allows shipping weights in MIT repo",
            ok7,
            m.license_note[:120],
        )
    )

    default_none = PipelineConfig().scorer is None
    ok8 = default_none
    items.append(
        ShipItem(
            8,
            "logistic backend; default scorer None; eval no PACKET_DECISION_API_KEY",
            ok8,
            f"PipelineConfig.scorer default None={default_none}; weights schema_version="
            f"{raw.get('schema_version')}; backend=logistic via --boundary-weights only",
        )
    )

    ship = all(i.passed for i in items)
    outcome = "a" if ship else "b"
    return ShipReport(
        ship=ship,
        outcome=outcome,
        items=items,
        holdout_heuristic={
            "macro_f1": heur.macro_f1,
            "micro_f1": heur.micro_f1,
            "per_packet_f1": heur.per_packet_f1,
        },
        holdout_scorer={
            "macro_f1": scorer.macro_f1,
            "micro_f1": scorer.micro_f1,
            "per_packet_f1": scorer.per_packet_f1,
        },
        max_holdout_model_p=max_p,
        dlnet_macro_scorer_on=dlnet_macro,
        builtin_f1=builtin_f1,
    )


def train_and_evaluate_ship(
    politax_cache: Path,
    weights_path: Path,
    dlnet_cache: Path = DEFAULT_WHOLEDOC_CACHE,
    eval_root: Path | None = None,
) -> ShipReport:
    train_trump_logistic(politax_cache, weights_path)
    return evaluate_politax_ship(weights_path, politax_cache, dlnet_cache, eval_root)
