from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return payload


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
        handle.write("\n")


def with_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Return a copy of cfg with dotted keys (e.g. 'optimizer.head_lr') set."""
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        node = cfg
        *parents, leaf = dotted_key.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = copy.deepcopy(value)
    return cfg
