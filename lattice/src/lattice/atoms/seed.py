"""Seed atoms — the brain's starter knowledge.

A curated set of Python programming atoms loaded on demand so a fresh
lattice install isn't a blank slate. Categories:

- antipattern: well-known traps the agent should be warned about
- preference: Python/project-style conventions
- skill: reusable techniques (test layout, error handling, etc.)
- fact: stable knowledge about common libraries

These are intentionally short and concrete — atoms surface as hints
in the LLM's Observation, and small models follow concrete advice
better than abstract principles.

Future iterations let the user override / extend / disable categories
via a YAML config; v0 is the hard-coded list below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from lattice.atoms.atom import AtomType
from lattice.atoms.store import AtomStore


@dataclass(frozen=True)
class SeedAtom:
    content: str
    type: AtomType
    region: str
    tags: tuple[str, ...]
    importance: float = 0.6


SEED_ATOMS: tuple[SeedAtom, ...] = (
    # ---- Antipatterns (concrete, single-rule things to avoid) ----
    SeedAtom(
        content="Do not use `except Exception:` or bare `except:` — catch the specific exception you can actually handle.",
        type=AtomType.ANTIPATTERN,
        region="python:errors",
        tags=("exceptions", "antipattern"),
        importance=0.8,
    ),
    SeedAtom(
        content="Mutable default arguments (def f(x=[])) leak state between calls. Use None and create the list inside.",
        type=AtomType.ANTIPATTERN,
        region="python:functions",
        tags=("functions", "mutable-default"),
        importance=0.85,
    ),
    SeedAtom(
        content="Never use eval() or exec() on input that could be untrusted; both arbitrary-code-execute.",
        type=AtomType.ANTIPATTERN,
        region="python:security",
        tags=("security", "eval"),
        importance=0.95,
    ),
    SeedAtom(
        content="Do not import network libraries at module top-level in performance-sensitive or import-cycle-prone files; use a lazy import inside the function that needs it.",
        type=AtomType.ANTIPATTERN,
        region="python:imports",
        tags=("imports", "performance"),
        importance=0.6,
    ),
    SeedAtom(
        content="Don't catch and silently ignore exceptions (`except: pass`) — at minimum log them.",
        type=AtomType.ANTIPATTERN,
        region="python:errors",
        tags=("exceptions", "logging"),
        importance=0.75,
    ),
    SeedAtom(
        content="Avoid string-concatenation SQL building; use parameterized queries to prevent SQL injection.",
        type=AtomType.ANTIPATTERN,
        region="python:db",
        tags=("sql", "security"),
        importance=0.9,
    ),
    SeedAtom(
        content="Don't use `os.system` for shell commands; use `subprocess.run` with a list of args.",
        type=AtomType.ANTIPATTERN,
        region="python:shell",
        tags=("shell", "subprocess"),
        importance=0.7,
    ),
    SeedAtom(
        content="Comparing to None with `==` works but is non-idiomatic; use `is None` / `is not None`.",
        type=AtomType.ANTIPATTERN,
        region="python:style",
        tags=("style", "none"),
        importance=0.5,
    ),
    SeedAtom(
        content="Don't reuse loop variables outside the loop's body — name leakage is a footgun.",
        type=AtomType.ANTIPATTERN,
        region="python:style",
        tags=("style", "scoping"),
        importance=0.5,
    ),
    SeedAtom(
        content="Avoid `time.sleep` in production polling loops; use proper async/await or event-driven primitives.",
        type=AtomType.ANTIPATTERN,
        region="python:async",
        tags=("async", "polling"),
        importance=0.65,
    ),
    # ---- Preferences (style conventions) ----
    SeedAtom(
        content="Python functions and variables use snake_case; classes use PascalCase; constants use UPPER_SNAKE_CASE.",
        type=AtomType.PREFERENCE,
        region="python:style",
        tags=("naming", "pep8"),
        importance=0.7,
    ),
    SeedAtom(
        content="Type-annotate public function signatures (parameters and return). Use `| None` instead of `Optional[X]` on 3.10+.",
        type=AtomType.PREFERENCE,
        region="python:typing",
        tags=("types", "annotations"),
        importance=0.7,
    ),
    SeedAtom(
        content="Tests live in a top-level `tests/` directory mirroring the package layout; test files are named `test_<module>.py`.",
        type=AtomType.PREFERENCE,
        region="python:testing",
        tags=("tests", "layout"),
        importance=0.7,
    ),
    SeedAtom(
        content="Source code lives under `src/<package>/` so the package is only importable when installed; avoids accidental local-shadow imports.",
        type=AtomType.PREFERENCE,
        region="python:layout",
        tags=("layout", "src"),
        importance=0.6,
    ),
    SeedAtom(
        content="Prefer dataclasses or Pydantic models over loose dicts when a value has a fixed shape.",
        type=AtomType.PREFERENCE,
        region="python:design",
        tags=("dataclasses", "pydantic"),
        importance=0.6,
    ),
    # ---- Skills (concrete reusable techniques) ----
    SeedAtom(
        content="To take a credit card payment in Python, the standard library is `stripe` (PyPI). Never call card endpoints directly.",
        type=AtomType.SKILL,
        region="python:payments",
        tags=("payments", "stripe"),
        importance=0.8,
    ),
    SeedAtom(
        content="For HTTP requests, prefer `httpx` (sync+async, modern) or `requests` (sync only). Stdlib `urllib.request` is fine for trivial cases.",
        type=AtomType.SKILL,
        region="python:http",
        tags=("http", "httpx", "requests"),
        importance=0.7,
    ),
    SeedAtom(
        content="For SQL, SQLAlchemy 2.0 ORM is the dominant choice. Avoid building raw SQL strings; use the Core or ORM expression language.",
        type=AtomType.SKILL,
        region="python:db",
        tags=("sql", "sqlalchemy"),
        importance=0.7,
    ),
    SeedAtom(
        content="For pytest fixtures, prefer narrow `tmp_path` and `monkeypatch` over module-level state. Use `pytest.fixture(scope='session')` only for genuinely shared expensive setup.",
        type=AtomType.SKILL,
        region="python:testing",
        tags=("pytest", "fixtures"),
        importance=0.6,
    ),
    SeedAtom(
        content="To add CLI commands, use `argparse` (stdlib) for small tools and `click` or `typer` for richer interfaces.",
        type=AtomType.SKILL,
        region="python:cli",
        tags=("cli", "argparse"),
        importance=0.55,
    ),
    SeedAtom(
        content="Use context managers (`with` blocks) for any resource needing cleanup: files, locks, DB sessions, network connections.",
        type=AtomType.SKILL,
        region="python:resources",
        tags=("with", "context-manager"),
        importance=0.7,
    ),
    SeedAtom(
        content="Use `pathlib.Path` instead of `os.path` for new code. Path arithmetic with `/` is more readable than os.path.join.",
        type=AtomType.SKILL,
        region="python:io",
        tags=("pathlib", "io"),
        importance=0.55,
    ),
    SeedAtom(
        content="For JSON, use `json.loads` / `json.dumps` from stdlib. For schemas/validation, use Pydantic.",
        type=AtomType.SKILL,
        region="python:serialization",
        tags=("json", "pydantic"),
        importance=0.55,
    ),
    SeedAtom(
        content="For async code, `asyncio` with `async`/`await` is the default. Don't mix threading and asyncio carelessly.",
        type=AtomType.SKILL,
        region="python:async",
        tags=("async", "asyncio"),
        importance=0.65,
    ),
    SeedAtom(
        content="For dates and times, prefer `datetime.now(timezone.utc)` over `datetime.utcnow()` (deprecated semantics). Always be explicit about tz.",
        type=AtomType.SKILL,
        region="python:datetime",
        tags=("datetime", "timezone"),
        importance=0.65,
    ),
    # ---- Facts (stable knowledge about libraries) ----
    SeedAtom(
        content="FastAPI uses Pydantic models for request/response validation and OpenAPI auto-generation; routes are async functions.",
        type=AtomType.FACT,
        region="python:web",
        tags=("fastapi", "web"),
        importance=0.6,
    ),
    SeedAtom(
        content="Pydantic v2 uses model_dump() (not dict()), model_validate() (not parse_obj_as), and ConfigDict (not Config class).",
        type=AtomType.FACT,
        region="python:pydantic",
        tags=("pydantic", "v2"),
        importance=0.7,
    ),
    SeedAtom(
        content="numpy arrays are zero-copy views by default; use .copy() when you need a new buffer.",
        type=AtomType.FACT,
        region="python:numpy",
        tags=("numpy", "memory"),
        importance=0.55,
    ),
    SeedAtom(
        content="SQLite is single-writer; concurrent writes require WAL journal mode (`PRAGMA journal_mode=WAL`).",
        type=AtomType.FACT,
        region="python:sqlite",
        tags=("sqlite", "wal"),
        importance=0.65,
    ),
    SeedAtom(
        content="Python's GIL means CPU-bound work in threads does not parallelize. Use multiprocessing for CPU-bound, threads/asyncio for IO-bound.",
        type=AtomType.FACT,
        region="python:concurrency",
        tags=("gil", "concurrency"),
        importance=0.6,
    ),
)


def seed_store(store: AtomStore, *, atoms: Iterable[SeedAtom] | None = None) -> int:
    """Add the seed atoms to *store*. Returns the count added.

    Skips atoms whose content already exists in the store (cheap dedup
    via embedding similarity is handled by the store itself once we
    plumb that through; for v0 we trust the seed list is run once per
    store).
    """
    atoms = atoms if atoms is not None else SEED_ATOMS
    added = 0
    for a in atoms:
        store.add(
            a.content,
            type=a.type,
            region=a.region,
            tags=a.tags,
            importance=a.importance,
        )
        added += 1
    return added
