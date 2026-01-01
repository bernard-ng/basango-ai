from __future__ import annotations

import os
from pathlib import Path


def get_root_path() -> Path:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    return Path(root)


def get_data_path(layer: str) -> Path:
    path = get_root_path() / "data" / layer
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["get_root_path", "get_data_path"]
