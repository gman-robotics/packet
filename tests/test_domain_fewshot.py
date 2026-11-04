"""GMA-463: opt-in few-shot schema-constrained domain adapter."""

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from packet.citations import block_ids, citation_id_counts
from packet.eval.metrics import citation_integrity
from packet.eval.pdfutil import write_text_pdf
from packet.types import PACKET_FLAGS_KEY, LogicalDocument, Packet, PacketOrigin

_DIR = Path(__file__).resolve().parent / "fixtures" / "domain_fewshot"
_FIXTURE_NAMES = ("lfb_2015.json", "lfb_2017.json", "djt_mmm_2016.json")
_PRIVACY = re.compile(
    r"\d{3}-\d{2}-\d{4}"
    r"|\d{2}-\d{7}"
    r"|\$"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|(?<!\d)\d{9}(?!\d)"
)


class StubCompletion:
    """Records ``(schema, prompt)`` and returns one fixed string. Not a decision backend."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[dict, str]] = []

    def complete_json(self, *, schema: dict, prompt: str) -> str:
        self.calls.append((schema, prompt))
        return self.reply


def _schema():
    cached = getattr(_schema, "model", None)
    if cached is not None:
        return cached
    from packet.domain_fewshot import SourcedStr

    class FormFace(BaseModel):
        model_config = ConfigDict(extra="forbid")
        form_name: SourcedStr
        tax_year: SourcedStr | None = None

    _schema.model = FormFace
    return FormFace


def _load(name: str) -> dict:
    return json.loads((_DIR / name).read_text())


def _examples():
    from packet.domain_fewshot import Example

    schema = _schema()
    built = []
    for name in _FIXTURE_NAMES:
        raw = _load(name)
        built.append(
            Example(
                doc=LogicalDocument.model_validate(raw["document"]),
                output=schema.model_validate(raw["output"]),
                source_file=raw["source_file"],
            )
        )
    return built


def _target() -> LogicalDocument:
    return _examples()[0].doc.model_copy(deep=True)


def _body_ids(doc: LogicalDocument) -> list[str]:
    return [block.id for block in doc.sections[0].blocks]


def _reply(form_ids: list[str], year_ids: list[str] | None) -> str:
    payload: dict = {
        "form_name": {"value": "Form 1065", "source_block_ids": form_ids},
    }
    if year_ids is None:
        payload["tax_year"] = None
    else:
        payload["tax_year"] = {"value": "2015", "source_block_ids": year_ids}
    return json.dumps(payload)


def _adapter(reply: str, **kwargs):
    from packet.domain_fewshot import FewShotAdapter

    stub = StubCompletion(reply)
    adapter = FewShotAdapter(
        name="form-face",
        schema=_schema(),
        examples=_examples(),
        backend=stub,
        **kwargs,
    )
    return adapter, stub


def _tree_dump(doc: LogicalDocument) -> dict:
    payload = doc.model_dump(mode="json")
    return {
        "id": payload["id"],
        "title": payload["title"],
        "page_start": payload["page_start"],
        "page_end": payload["page_end"],
        "sections": payload["sections"],
        "furniture": payload["furniture"],
    }


def _tree_hash(doc: LogicalDocument) -> str:
    raw = json.dumps(_tree_dump(doc), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _record(doc: LogicalDocument) -> dict:
    return doc.extras["domain.fewshot"]


def _grounding_rate(record: dict) -> float:
    kept = sum(len(field["source_block_ids"]) for field in record["fields"].values())
    ungrounded_drops = sum(1 for item in record["dropped"] if item["reason"] == "ungrounded")
    return kept / (kept + ungrounded_drops)


def test_default_import_does_not_load_fewshot():
    src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src) if not prior else str(src) + os.pathsep + prior
    script = (
        "import sys\n"
        "sys.modules.pop('packet.domain_fewshot', None)\n"
        "import packet\n"
        "import packet.pipeline\n"
        "import packet.domain\n"
        "raise SystemExit('packet.domain_fewshot' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=src.parent,
    )
    assert completed.returncode == 0, completed.stderr


def test_default_pipeline_has_no_adapter(tmp_path):
    from packet.pipeline import Pipeline, PipelineConfig

    assert PipelineConfig().domain_adapters == []
    pdf = write_text_pdf(tmp_path / "one.pdf", [["Form 1065 cover sheet"]])
    packet = Pipeline().parse(pdf)
    assert packet.documents
    for doc in packet.documents:
        assert "domain.fewshot" not in doc.extras


def test_applies_defaults_false():
    from packet.domain import apply_domains

    doc = _target()
    adapter, stub = _adapter(_reply(["lfb2015-b1"], ["lfb2015-b2"]))
    assert adapter.applies(doc) is False
    out = apply_domains(doc, [adapter])
    assert stub.calls == []
    assert "domain.fewshot" not in out.extras


def test_tree_hash_stable():
    from packet.domain import apply_domains

    doc = _target()
    real = _body_ids(doc)[0]
    adapter, _stub = _adapter(
        _reply([real], ["not-in-doc"]),
        applies_to=lambda _doc: True,
    )
    before = _tree_dump(doc)
    before_hash = _tree_hash(doc)
    out = adapter.enrich(doc)
    assert _tree_hash(out) == before_hash
    assert _tree_dump(out) == before
    assert _tree_hash(doc) == before_hash
    applied = apply_domains(doc.model_copy(deep=True), [adapter])
    assert _tree_hash(applied) == before_hash
    assert _tree_dump(applied) == before


def test_tree_hash_detects_reorder():
    doc = _target()
    original = _tree_hash(doc)
    reordered = doc.model_copy(deep=True)
    blocks = reordered.sections[0].blocks
    blocks[0], blocks[1] = blocks[1], blocks[0]
    assert _tree_hash(reordered) != original
    swapped = doc.model_copy(deep=True)
    left, right = swapped.sections[0].blocks[0], swapped.sections[0].blocks[1]
    left.id, right.id = right.id, left.id
    assert _tree_hash(swapped) != original


def test_grounding_rate_half_on_stub():
    """kept / (kept + ungrounded) = 1 / (1 + 1) = 0.5; every kept field is 100% grounded."""

    doc = _target()
    real = _body_ids(doc)[0]
    adapter, stub = _adapter(
        _reply([real], ["not-in-doc"]),
        applies_to=lambda _doc: True,
    )
    out = adapter.enrich(doc)
    record = _record(out)
    assert set(record["fields"]) == {"form_name"}
    assert record["fields"]["form_name"]["source_block_ids"] == [real]
    known = block_ids(out)
    for field in record["fields"].values():
        assert field["source_block_ids"]
        assert set(field["source_block_ids"]) <= known
    assert record["dropped"] == [
        {"field": "tax_year", "reason": "ungrounded", "source_block_ids": ["not-in-doc"]}
    ]
    assert _grounding_rate(record) == 0.5
    assert len(stub.calls) == 1


def test_mixed_ids_dropped():
    doc = _target()
    real, other = _body_ids(doc)
    adapter, _stub = _adapter(
        _reply([real, "not-in-doc"], [other]),
        applies_to=lambda _doc: True,
    )
    record = _record(adapter.enrich(doc))
    assert "form_name" not in record["fields"]
    assert set(record["fields"]) == {"tax_year"}
    assert record["dropped"] == [
        {
            "field": "form_name",
            "reason": "ungrounded",
            "source_block_ids": [real, "not-in-doc"],
        }
    ]


def test_free_text_not_stored():
    doc = _target()
    replies = ("the form is 1040", "[1, 2]", "12", '{"form_name": "1040"}')
    for reply in replies:
        adapter, stub = _adapter(reply, applies_to=lambda _doc: True)
        out = adapter.enrich(doc)
        record = _record(out)
        assert record["fields"] == {}
        assert record["dropped"] == [
            {"field": None, "reason": "invalid_schema", "source_block_ids": []}
        ]
        assert len(stub.calls) == 1
        if reply == "the form is 1040":
            assert reply not in out.model_dump_json()


def test_schema_must_be_sourced():
    from packet.domain_fewshot import Example, FewShotAdapter, SourcedStr

    schema = _schema()
    examples = _examples()

    class Plain(BaseModel):
        model_config = ConfigDict(extra="forbid")
        form_name: str

    class Open(BaseModel):
        form_name: SourcedStr
        tax_year: SourcedStr | None = None

    with pytest.raises(ValueError):
        FewShotAdapter("bad", Plain, [], StubCompletion("{}"))
    with pytest.raises(ValueError):
        FewShotAdapter("bad", Open, [], StubCompletion("{}"))
    with pytest.raises(TypeError):
        FewShotAdapter("bad", dict, [], StubCompletion("{}"))

    for reserved in ("sections", "blocks", "furniture"):
        model = type(
            "Reserved",
            (BaseModel,),
            {
                "__annotations__": {reserved: SourcedStr},
                "model_config": ConfigDict(extra="forbid"),
            },
        )
        with pytest.raises(ValueError):
            FewShotAdapter("bad", model, [], StubCompletion("{}"))

    class Aliased(BaseModel):
        model_config = ConfigDict(extra="forbid")
        flagged: SourcedStr = Field(alias="packet.flags")

    with pytest.raises(ValueError):
        FewShotAdapter("bad", Aliased, [], StubCompletion("{}"))

    doc = examples[0].doc
    bad_output = schema.model_validate(
        {
            "form_name": {"value": "Form 1065", "source_block_ids": ["missing-block"]},
            "tax_year": None,
        }
    )
    with pytest.raises(ValueError):
        FewShotAdapter(
            "bad",
            schema,
            [Example(doc=doc, output=bad_output, source_file=examples[0].source_file)],
            StubCompletion("{}"),
        )
    empty = schema.model_construct(
        form_name=SourcedStr.model_construct(value="Form 1065", source_block_ids=[]),
        tax_year=None,
    )
    with pytest.raises(ValueError):
        FewShotAdapter(
            "bad",
            schema,
            [Example(doc=doc, output=empty, source_file=examples[0].source_file)],
            StubCompletion("{}"),
        )

    class Twin(BaseModel):
        model_config = ConfigDict(extra="forbid")
        form_name: SourcedStr
        tax_year: SourcedStr | None = None

    twin = Twin.model_validate(examples[0].output.model_dump())
    with pytest.raises(TypeError):
        FewShotAdapter(
            "bad",
            schema,
            [Example(doc=doc, output=twin, source_file=examples[0].source_file)],
            StubCompletion("{}"),
        )


def test_examples_train_not_holdout():
    from packet.eval.politax import load_manifest
    from packet.eval.politax_split import holdout_file_names, partition_docs

    manifest = load_manifest()
    train, _holdout = partition_docs(manifest)
    train_files = {doc.file_name for doc in train}
    holdout = holdout_file_names(manifest)
    texts = []
    for name in _FIXTURE_NAMES:
        raw_text = (_DIR / name).read_text()
        texts.append(raw_text)
        raw = json.loads(raw_text)
        assert raw["subject"] == "trump"
        assert raw["source_file"] in train_files
        assert raw["source_file"] not in holdout
        match = _PRIVACY.search(raw_text)
        assert match is None, match.group(0)
    combined = "\n".join(texts)
    for file_name in holdout:
        assert file_name not in combined
    for banned in ("obama", "harris", "biden"):
        assert banned not in combined.lower()


def test_prompt_carries_three_examples():
    doc = _target()
    examples = _examples()
    real = _body_ids(doc)[0]
    adapter, stub = _adapter(_reply([real], ["not-in-doc"]), applies_to=lambda _doc: True)
    adapter.enrich(doc)
    assert len(stub.calls) == 1
    schema, prompt = stub.calls[0]
    assert schema["title"] == _schema().model_json_schema()["title"]
    assert schema["title"] in prompt
    assert len(examples) == 3
    for example in examples:
        assert example.source_file in prompt
    first = examples[0].doc
    b1 = prompt.index("lfb2015-b1\t")
    b2 = prompt.index("lfb2015-b2\t")
    header = prompt.index("lfb2015-h1\t")
    assert b1 < b2 < header
    assert first.furniture[0].id == "lfb2015-h1"


def test_packet_flags_untouched():
    from packet.domain import DomainAdapter, apply_domains

    flags = [{"kind": "weak_boundary", "page": 0, "ref": None, "detail": {"note": "seed"}}]
    snapshot = copy.deepcopy(flags)
    doc = _target()
    doc.extras[PACKET_FLAGS_KEY] = flags
    real = _body_ids(doc)[0]
    adapter, _stub = _adapter(_reply([real], ["not-in-doc"]), applies_to=lambda _doc: True)
    out = apply_domains(doc, [adapter])
    assert out.extras[PACKET_FLAGS_KEY] == snapshot
    assert flags == snapshot
    assert out.extras[PACKET_FLAGS_KEY] is not flags
    assert "domain.fewshot" in out.extras

    class HostileFlags:
        name = "hostile-flags"

        def applies(self, doc: LogicalDocument) -> bool:
            return True

        def enrich(self, doc: LogicalDocument) -> LogicalDocument:
            current = doc.extras.get(PACKET_FLAGS_KEY)
            if isinstance(current, list):
                current.append({"kind": "tampered", "page": 1, "ref": None, "detail": {}})
            doc.extras[PACKET_FLAGS_KEY] = [
                {"kind": "replaced", "page": 9, "ref": None, "detail": {"note": "hostile"}}
            ]
            doc.extras["hostile.ran"] = True
            return doc

    hostile: DomainAdapter = HostileFlags()
    held = _target()
    held.extras[PACKET_FLAGS_KEY] = flags
    restored = apply_domains(held, [hostile])
    assert restored.extras[PACKET_FLAGS_KEY] == snapshot
    assert flags == snapshot
    assert restored.extras[PACKET_FLAGS_KEY] is not flags
    assert "hostile.ran" in restored.extras

    bare = _target()
    assert PACKET_FLAGS_KEY not in bare.extras
    cleaned = apply_domains(bare, [hostile])
    assert PACKET_FLAGS_KEY not in cleaned.extras
    assert "hostile.ran" in cleaned.extras


def test_citation_integrity_ignores_nested_ids():
    doc = _target()
    real = _body_ids(doc)[0]
    adapter, _stub = _adapter(
        _reply([real], ["not-in-doc"]),
        applies_to=lambda _doc: True,
    )
    out = adapter.enrich(doc)
    packet = Packet(
        id="p",
        origin=PacketOrigin(path="lfb2015.pdf", page_count=1),
        documents=[out],
    )
    total, unresolved = citation_id_counts(out, block_ids(out))
    assert total == 0
    assert unresolved == []
    assert citation_integrity(packet) == 1.0
    assert "not-in-doc" in json.dumps(_record(out))


def test_prompt_too_long_skips_backend():
    import inspect

    from packet.domain_fewshot import FewShotAdapter

    default = inspect.signature(FewShotAdapter.__init__).parameters["max_prompt_chars"].default
    assert default == 8000
    doc = _target()
    adapter, stub = _adapter(
        _reply(["lfb2015-b1"], ["lfb2015-b2"]),
        applies_to=lambda _doc: True,
        max_prompt_chars=8,
    )
    assert adapter.max_prompt_chars == 8
    out = adapter.enrich(doc)
    assert stub.calls == []
    record = _record(out)
    assert record["fields"] == {}
    assert record["dropped"] == [
        {"field": None, "reason": "prompt_too_long", "source_block_ids": []}
    ]


def test_type_hint_absent_already_set_and_filled():
    doc = _target()
    real = _body_ids(doc)[0]
    reply = _reply([real], None)
    absent, _stub = _adapter(reply, applies_to=lambda _doc: True)
    omitted = absent.enrich(doc)
    assert omitted.type_hint is None
    assert _tree_hash(omitted) == _tree_hash(doc)

    labeled, _stub = _adapter(reply, applies_to=lambda _doc: True, type_hint="tax_return")
    preset = doc.model_copy(deep=True)
    preset.type_hint = "already"
    kept = labeled.enrich(preset)
    assert kept.type_hint == "already"

    filled = labeled.enrich(doc.model_copy(deep=True))
    assert filled.type_hint == "tax_return"
    assert _tree_hash(filled) == _tree_hash(doc)


def test_none_sourced_omitted_without_drop():
    doc = _target()
    real = _body_ids(doc)[0]
    adapter, _stub = _adapter(_reply([real], None), applies_to=lambda _doc: True)
    record = _record(adapter.enrich(doc))
    assert set(record["fields"]) == {"form_name"}
    assert "tax_year" not in record["fields"]
    assert record["dropped"] == []


def test_union_annotation_accepts_only_sourced_str_or_none():
    from packet.domain_fewshot import FewShotAdapter, SourcedStr

    class OptionalSourced(BaseModel):
        model_config = ConfigDict(extra="forbid")
        form_name: SourcedStr | None = None

    accepted = FewShotAdapter("ok", OptionalSourced, [], StubCompletion("{}"))
    assert accepted.name == "ok"

    class SourcedOrInt(BaseModel):
        model_config = ConfigDict(extra="forbid")
        form_name: SourcedStr | int

    with pytest.raises(ValueError):
        FewShotAdapter("bad", SourcedOrInt, [], StubCompletion("{}"))

    class OptionalStr(BaseModel):
        model_config = ConfigDict(extra="forbid")
        form_name: str | None = None

    with pytest.raises(ValueError):
        FewShotAdapter("bad", OptionalStr, [], StubCompletion("{}"))


def test_enrich_does_not_mutate_caller_document():
    doc = _target()
    real = _body_ids(doc)[0]
    nested = doc.sections[0].blocks[0].model_dump(mode="json")
    extras = copy.deepcopy(doc.extras)
    adapter, _stub = _adapter(_reply([real], None), applies_to=lambda _doc: True)
    out = adapter.enrich(doc)
    assert doc.sections[0].blocks[0].model_dump(mode="json") == nested
    assert doc.extras == extras
    assert "domain.fewshot" not in doc.extras
    out.sections[0].blocks[0].text = "changed on the result only"
    out.extras["domain.fewshot"]["fields"]["form_name"]["value"] = "changed"
    assert doc.sections[0].blocks[0].text == nested["text"]
    assert doc.extras == extras


def test_prompt_equal_to_cap_is_sent():
    doc = _target()
    real = _body_ids(doc)[0]
    reply = _reply([real], None)
    probe, probe_stub = _adapter(
        reply,
        applies_to=lambda _doc: True,
        max_prompt_chars=100_000,
    )
    probe.enrich(doc)
    exact = len(probe_stub.calls[0][1])

    at_limit, at_stub = _adapter(
        reply,
        applies_to=lambda _doc: True,
        max_prompt_chars=exact,
    )
    at_out = at_limit.enrich(doc.model_copy(deep=True))
    assert len(at_stub.calls) == 1
    assert len(at_stub.calls[0][1]) == exact
    assert at_out.extras["domain.fewshot"]["dropped"] == []

    over, over_stub = _adapter(
        reply,
        applies_to=lambda _doc: True,
        max_prompt_chars=exact - 1,
    )
    over_out = over.enrich(doc.model_copy(deep=True))
    assert over_stub.calls == []
    assert over_out.extras["domain.fewshot"]["dropped"] == [
        {"field": None, "reason": "prompt_too_long", "source_block_ids": []}
    ]


def test_prompt_json_uses_sorted_keys():
    from packet.domain_fewshot import Example, FewShotAdapter, SourcedStr

    class Zebra(BaseModel):
        model_config = ConfigDict(extra="forbid")
        zebra: SourcedStr
        apple: SourcedStr | None = None

    doc = _target()
    block_id = doc.sections[0].blocks[0].id
    output = Zebra(
        zebra=SourcedStr(value="Form 1065", source_block_ids=[block_id]),
        apple=None,
    )
    stub = StubCompletion("{}")
    adapter = FewShotAdapter(
        name="zebra",
        schema=Zebra,
        examples=[Example(doc=doc, output=output, source_file="train.pdf")],
        backend=stub,
        applies_to=lambda _doc: True,
    )
    adapter.enrich(doc.model_copy(deep=True))
    assert len(stub.calls) == 1
    prompt = stub.calls[0][1]
    schema_json = json.dumps(
        Zebra.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )
    plain_schema = json.dumps(
        Zebra.model_json_schema(),
        sort_keys=False,
        separators=(",", ":"),
    )
    assert schema_json != plain_schema
    assert schema_json in prompt
    assert plain_schema not in prompt
    sorted_output = json.dumps(output.model_dump(), sort_keys=True, separators=(",", ":"))
    plain_output = json.dumps(output.model_dump(), sort_keys=False, separators=(",", ":"))
    assert sorted_output != plain_output
    assert sorted_output in prompt
    assert plain_output not in prompt
