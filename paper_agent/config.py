"""Strict YAML configuration loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: str | Path, schema: type[T]) -> T:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")
    return schema.model_validate(raw)


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
