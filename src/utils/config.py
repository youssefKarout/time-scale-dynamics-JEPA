"""Minimal YAML config loader with attribute access and CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


class DotDict(dict):
    """A dict whose keys are also accessible as attributes (recursively)."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_dotdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    return obj


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> DotDict:
    """Load a YAML config file into a DotDict, applying dotted overrides.

    overrides maps dotted keys to values, e.g. {"run.variant": "shared"}.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = _to_dotdict(copy.deepcopy(raw))

    if overrides:
        for dotted_key, value in overrides.items():
            _set_dotted(cfg, dotted_key, value)

    return cfg


def _set_dotted(cfg: DotDict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], (dict, DotDict)):
            node[part] = DotDict()
        node = node[part]
    node[parts[-1]] = _to_dotdict(value)


def save_config(cfg: DotDict, path: str | Path) -> None:
    """Persist a config back to YAML (used for run provenance)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_plain(cfg), fh, sort_keys=False)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def to_dotdict(obj: Any) -> Any:
    """Public wrapper: convert a plain nested dict into a DotDict tree."""
    return _to_dotdict(obj)
