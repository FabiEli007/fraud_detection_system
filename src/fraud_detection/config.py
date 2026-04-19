from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_local_path(project_root: str | Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (Path(project_root) / candidate).resolve()


def resolve_tracking_uri(project_root: str | Path, tracking_uri: str) -> str:
    if "://" in tracking_uri:
        return tracking_uri
    return resolve_local_path(project_root, tracking_uri).as_uri()

