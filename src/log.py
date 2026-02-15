from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Any, Iterable


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    ensure_parent(path)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
