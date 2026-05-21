"""Import / export atoms as JSON or YAML.

Users grow their brain by:
1. Curating a JSON/YAML file with atom records.
2. Running `lattice brain import --db DB FILE`.

Sharing a brain pack is sharing a JSON file. Brain packs compose with
the built-in seed pack and accumulated experience atoms.

JSON shape (one atom per element):

    [
      {
        "content": "Use snake_case for Python functions.",
        "type": "preference",
        "region": "python:style",
        "tags": ["naming", "pep8"],
        "importance": 0.7,
        "confidence": 1.0
      },
      ...
    ]

YAML is the same shape, parsed via pyyaml (already a transitive dep
of libcst / sentence-transformers).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lattice.atoms.atom import AtomType
from lattice.atoms.store import AtomStore


def _load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"expected a list of atom records, got {type(data).__name__}")
    return data


def import_atoms(store: AtomStore, path: str | Path) -> int:
    """Load atoms from a JSON or YAML file into *store*.

    Each record must have at least `content`. Missing fields take
    library defaults (type='fact', region='', tags=(), importance=0.5).
    """
    records = _load_records(Path(path))
    added = 0
    for rec in records:
        if not isinstance(rec, dict) or "content" not in rec:
            raise ValueError(f"atom record missing 'content': {rec!r}")
        atype_raw = rec.get("type", "fact")
        try:
            atype = AtomType(atype_raw)
        except ValueError as exc:
            raise ValueError(f"unknown atom type {atype_raw!r}") from exc
        tags = tuple(rec.get("tags", ()))
        store.add(
            rec["content"],
            type=atype,
            region=rec.get("region", ""),
            tags=tags,
            importance=float(rec.get("importance", 0.5)),
            confidence=float(rec.get("confidence", 1.0)),
        )
        added += 1
    return added


def export_atoms(store: AtomStore, path: str | Path) -> int:
    """Dump every atom in *store* to a JSON file at *path*. Returns count."""
    import sqlite3

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cur = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT content, type, region, tags, importance, confidence FROM atoms"
    )
    rows: list[sqlite3.Row] = list(cur.fetchall())
    records = []
    for r in rows:
        records.append(
            {
                "content": r[0],
                "type": r[1],
                "region": r[2],
                "tags": json.loads(r[3]),
                "importance": r[4],
                "confidence": r[5],
            }
        )
    target.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(records)


def stats(store: AtomStore) -> dict[str, dict[str, int]]:
    """Return per-region per-type atom counts."""
    cur = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT region, type, COUNT(*) FROM atoms GROUP BY region, type ORDER BY region, type"
    )
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for region, atype, count in cur.fetchall():
        out[region][atype] = count
    return dict(out)
