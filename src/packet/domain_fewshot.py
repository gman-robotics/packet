"""Opt-in few-shot schema-constrained domain adapter.

Not imported by ``packet``, ``pipeline``, or ``domain``. A caller has to pass
the adapter. Completion goes through ``StructuredCompletion``, not
``DecisionBackend.decide``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from packet.citations import block_ids
from packet.types import LogicalDocument, Section

DOMAIN_FEWSHOT_KEY = "domain.fewshot"

_FORBIDDEN_FIELDS = frozenset({"sections", "blocks", "furniture", "packet.flags"})


class SourcedStr(BaseModel):
    """One extracted string and the block ids it was read from."""

    model_config = ConfigDict(extra="forbid")
    value: str
    source_block_ids: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class Example:
    doc: LogicalDocument
    output: BaseModel
    source_file: str


class StructuredCompletion(Protocol):
    def complete_json(self, *, schema: dict[str, Any], prompt: str) -> str: ...


def _applies_to_none(_doc: LogicalDocument) -> bool:
    return False


def _is_sourced(annotation: object) -> bool:
    if annotation is SourcedStr:
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is type(SourcedStr | None):
        args = get_args(annotation)
        others = [arg for arg in args if arg is not type(None)]
        return len(others) == 1 and others[0] is SourcedStr and type(None) in args
    return False


def _fold(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")


def _iter_blocks(doc: LogicalDocument) -> list:
    ordered: list = []

    def walk(section: Section) -> None:
        ordered.extend(section.blocks)
        for child in section.children:
            walk(child)

    for section in doc.sections:
        walk(section)
    ordered.extend(doc.furniture)
    return ordered


def _block_lines(doc: LogicalDocument) -> str:
    return "\n".join(f"{block.id}\t{_fold(block.text)}" for block in _iter_blocks(doc))


def _document_drop(reason: str) -> dict[str, Any]:
    return {"field": None, "reason": reason, "source_block_ids": []}


class FewShotAdapter:
    """Read a closed document, validate a stub JSON reply, store grounded fields."""

    def __init__(
        self,
        name: str,
        schema: type[BaseModel],
        examples: list[Example],
        backend: StructuredCompletion,
        applies_to: Callable[[LogicalDocument], bool] = _applies_to_none,
        type_hint: str | None = None,
        max_prompt_chars: int = 8000,
    ) -> None:
        self._require_schema(schema)
        self.name = name
        self.schema = schema
        self.examples = list(examples)
        self.backend = backend
        self.applies_to = applies_to
        self.type_hint = type_hint
        self.max_prompt_chars = max_prompt_chars
        for example in self.examples:
            self._require_example(example)

    def applies(self, doc: LogicalDocument) -> bool:
        return self.applies_to(doc)

    def enrich(self, doc: LogicalDocument) -> LogicalDocument:
        updated = doc.model_copy(deep=True)
        prompt = self._prompt(updated)
        if len(prompt) > self.max_prompt_chars:
            return self._finish(updated, {}, [_document_drop("prompt_too_long")])
        raw = self.backend.complete_json(
            schema=self.schema.model_json_schema(),
            prompt=prompt,
        )
        parsed = self._parse(raw)
        if parsed is None:
            return self._finish(updated, {}, [_document_drop("invalid_schema")])
        fields, dropped = self._ground(updated, parsed)
        return self._finish(updated, fields, dropped)

    def _require_schema(self, schema: object) -> None:
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("schema must be a pydantic BaseModel subclass")
        if schema.model_config.get("extra") != "forbid":
            raise ValueError("schema must set extra='forbid'")
        for name, info in schema.model_fields.items():
            aliases = {
                item
                for item in (info.alias, getattr(info, "validation_alias", None))
                if isinstance(item, str)
            }
            if name in _FORBIDDEN_FIELDS or aliases & _FORBIDDEN_FIELDS:
                raise ValueError(f"schema field {name!r} is reserved")
            if not _is_sourced(info.annotation):
                raise ValueError(f"schema field {name!r} must be SourcedStr or SourcedStr | None")

    def _require_example(self, example: Example) -> None:
        if not isinstance(example.output, self.schema):
            raise TypeError("example output must be an instance of schema")
        known = block_ids(example.doc)
        for name in self.schema.model_fields:
            value = getattr(example.output, name)
            if value is None:
                continue
            ids = list(value.source_block_ids)
            if not ids or not set(ids).issubset(known):
                raise ValueError(
                    f"example field {name!r} source_block_ids are not a non-empty "
                    "subset of the document"
                )

    def _prompt(self, doc: LogicalDocument) -> str:
        schema_json = json.dumps(
            self.schema.model_json_schema(),
            sort_keys=True,
            separators=(",", ":"),
        )
        parts = [f"schema:\n{schema_json}"]
        for example in self.examples:
            output_json = json.dumps(
                example.output.model_dump(),
                sort_keys=True,
                separators=(",", ":"),
            )
            parts.append(
                "\n".join(
                    [
                        "example:",
                        f"source_file:\t{example.source_file}",
                        "blocks:",
                        _block_lines(example.doc),
                        "output:",
                        output_json,
                    ]
                )
            )
        parts.append("\n".join(["target:", "blocks:", _block_lines(doc)]))
        return "\n".join(parts)

    def _parse(self, raw: str) -> BaseModel | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        try:
            return self.schema.model_validate_json(raw)
        except ValidationError:
            return None

    def _ground(self, doc: LogicalDocument, parsed: BaseModel) -> tuple[dict[str, Any], list]:
        known = block_ids(doc)
        fields: dict[str, Any] = {}
        dropped: list[dict[str, Any]] = []
        for name in self.schema.model_fields:
            value = getattr(parsed, name)
            if value is None:
                continue
            dumped = value.model_dump()
            ids = list(dumped["source_block_ids"])
            if ids and set(ids).issubset(known):
                fields[name] = dumped
            else:
                dropped.append({"field": name, "reason": "ungrounded", "source_block_ids": ids})
        return fields, dropped

    def _finish(
        self,
        doc: LogicalDocument,
        fields: dict[str, Any],
        dropped: list[dict[str, Any]],
    ) -> LogicalDocument:
        doc.extras[DOMAIN_FEWSHOT_KEY] = {
            "name": self.name,
            "fields": fields,
            "dropped": dropped,
        }
        if self.type_hint is not None and doc.type_hint is None:
            doc.type_hint = self.type_hint
        return doc
