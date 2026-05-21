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


def _antipattern(content: str, region: str, tags: tuple[str, ...], importance: float = 0.7) -> SeedAtom:
    return SeedAtom(content=content, type=AtomType.ANTIPATTERN, region=region, tags=tags, importance=importance)


def _preference(content: str, region: str, tags: tuple[str, ...], importance: float = 0.6) -> SeedAtom:
    return SeedAtom(content=content, type=AtomType.PREFERENCE, region=region, tags=tags, importance=importance)


def _skill(content: str, region: str, tags: tuple[str, ...], importance: float = 0.6) -> SeedAtom:
    return SeedAtom(content=content, type=AtomType.SKILL, region=region, tags=tags, importance=importance)


def _fact(content: str, region: str, tags: tuple[str, ...], importance: float = 0.6) -> SeedAtom:
    return SeedAtom(content=content, type=AtomType.FACT, region=region, tags=tags, importance=importance)


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
    # ---- Expanded antipatterns ----
    _antipattern(
        "Don't use global state to share data between functions; pass arguments or use a class.",
        "python:design", ("globals", "state"), 0.75,
    ),
    _antipattern(
        "Avoid magic numbers in code; define a named constant or enum.",
        "python:style", ("magic-numbers", "constants"), 0.6,
    ),
    _antipattern(
        "Don't catch an exception just to re-raise it unchanged. Either handle it or let it propagate.",
        "python:errors", ("exceptions", "re-raise"), 0.7,
    ),
    _antipattern(
        "Avoid using `*args, **kwargs` in public APIs unless you're explicitly building a wrapper; it hides the contract.",
        "python:design", ("api", "signature"), 0.55,
    ),
    _antipattern(
        "Don't shadow built-in names like `list`, `dict`, `id`, `type` with local variables.",
        "python:style", ("builtins", "shadowing"), 0.7,
    ),
    _antipattern(
        "Don't store secrets in code or in the repo. Use environment variables or a secret manager.",
        "security", ("secrets", "config"), 0.95,
    ),
    _antipattern(
        "Don't run untrusted code, even via pickle.loads — pickle can execute arbitrary code on load.",
        "python:security", ("pickle", "deserialization"), 0.9,
    ),
    _antipattern(
        "Don't compare floats with `==`; use math.isclose or a small absolute tolerance.",
        "python:style", ("floats", "comparison"), 0.65,
    ),
    _antipattern(
        "Avoid bare `assert` for input validation in production — Python with `-O` strips assertions.",
        "python:validation", ("assert", "validation"), 0.7,
    ),
    _antipattern(
        "Don't use `from module import *`; it pollutes the namespace and breaks static analysis.",
        "python:imports", ("imports", "wildcards"), 0.6,
    ),
    _antipattern(
        "Don't mutate a list while iterating over it; copy first or build a new list.",
        "python:iteration", ("iteration", "mutation"), 0.7,
    ),
    _antipattern(
        "Don't hardcode file paths; use pathlib relative to a known root or load from config.",
        "python:io", ("paths", "config"), 0.55,
    ),
    _antipattern(
        "Avoid using mutable class attributes shared across instances unless you mean to share.",
        "python:classes", ("classes", "mutable"), 0.6,
    ),
    _antipattern(
        "Don't write tests that depend on each other's ordering or shared state.",
        "python:testing", ("tests", "isolation"), 0.7,
    ),
    _antipattern(
        "Don't catch `KeyboardInterrupt` or `SystemExit` in generic exception handlers.",
        "python:errors", ("exceptions", "control-flow"), 0.75,
    ),
    _antipattern(
        "Don't use deprecated APIs (e.g. datetime.utcnow, asyncio.coroutine, imp module). Check the warning.",
        "python:deprecation", ("deprecation",), 0.65,
    ),
    _antipattern(
        "Avoid deep nesting (>3 levels). Extract functions or use early returns.",
        "python:style", ("nesting", "readability"), 0.55,
    ),
    _antipattern(
        "Don't use `print` for application logging; use the `logging` module or a structured logger.",
        "python:logging", ("logging", "print"), 0.65,
    ),
    _antipattern(
        "Don't catch and log an exception and continue silently — the upstream code thinks it succeeded.",
        "python:errors", ("exceptions", "logging"), 0.75,
    ),
    _antipattern(
        "Don't run shell commands with user-supplied strings; use subprocess.run with a list and never shell=True on untrusted input.",
        "python:security", ("shell", "injection"), 0.9,
    ),

    # ---- Expanded preferences ----
    _preference(
        "Use `with open(...)` instead of bare open() so files get closed on exception.",
        "python:io", ("files", "context-manager"), 0.7,
    ),
    _preference(
        "Use f-strings for formatting, not %-formatting or .format() in new code.",
        "python:style", ("strings", "fstrings"), 0.65,
    ),
    _preference(
        "Module docstrings go at the top of the file. Function docstrings use triple double-quotes.",
        "python:style", ("docstrings", "pep257"), 0.55,
    ),
    _preference(
        "Public APIs export via `__all__` in __init__.py for clarity.",
        "python:packaging", ("packaging", "all"), 0.5,
    ),
    _preference(
        "Use `dataclasses.field(default_factory=list)` for mutable default factories — never `field(default=[])`.",
        "python:dataclasses", ("dataclasses", "defaults"), 0.7,
    ),
    _preference(
        "Prefer Pydantic v2 models for I/O-shaped data (API payloads, config). Use dataclasses for internal value types.",
        "python:design", ("pydantic", "dataclasses"), 0.6,
    ),
    _preference(
        "Type Optional values as `X | None` (PEP 604) on Python 3.10+.",
        "python:typing", ("typing", "optional"), 0.6,
    ),
    _preference(
        "Prefer `enum.StrEnum` (3.11+) over plain string constants for string-typed enums.",
        "python:typing", ("enum", "strenum"), 0.55,
    ),
    _preference(
        "Use ruff for linting + import sorting, mypy or pyright for type checking.",
        "python:tooling", ("ruff", "mypy"), 0.6,
    ),
    _preference(
        "Use uv (or pip-tools) for reproducible installs; pin transitive dependencies via a lockfile.",
        "python:tooling", ("uv", "packaging"), 0.55,
    ),
    _preference(
        "Format with ruff format (or black). Set line length to 88 (black default) or 100; keep it consistent.",
        "python:style", ("format", "ruff"), 0.55,
    ),
    _preference(
        "In tests, parametrize over inputs instead of writing N similar test functions.",
        "python:testing", ("pytest", "parametrize"), 0.6,
    ),
    _preference(
        "Keep function bodies short (~30 lines). When a function grows, extract helpers.",
        "python:style", ("readability",), 0.5,
    ),
    _preference(
        "Use `assert` only for invariants the author believes always hold — never for input validation.",
        "python:style", ("assert",), 0.55,
    ),

    # ---- Expanded skills ----
    _skill(
        "For environment-driven config, use pydantic-settings (or os.environ + a small parser). Pin types in the schema.",
        "python:config", ("config", "pydantic-settings"), 0.65,
    ),
    _skill(
        "For background tasks, asyncio.TaskGroup (3.11+) cancels siblings on failure. Cleaner than gather(*, return_exceptions=False).",
        "python:async", ("asyncio", "taskgroup"), 0.65,
    ),
    _skill(
        "For long-running data pipelines, use generators (yield) so memory doesn't grow with input size.",
        "python:performance", ("generators", "memory"), 0.6,
    ),
    _skill(
        "For caching, functools.lru_cache works for pure functions; cachetools.TTLCache for time-based eviction.",
        "python:caching", ("cache", "lru"), 0.6,
    ),
    _skill(
        "Use struct.pack / struct.unpack for binary protocol work; bytes.hex() / .fromhex() for hex.",
        "python:binary", ("struct", "binary"), 0.5,
    ),
    _skill(
        "For concurrent network I/O, prefer asyncio + httpx. For CPU-bound, concurrent.futures.ProcessPoolExecutor.",
        "python:concurrency", ("asyncio", "process-pool"), 0.65,
    ),
    _skill(
        "For schema migrations, use Alembic with SQLAlchemy. Never edit the DB directly in production.",
        "python:db", ("migrations", "alembic"), 0.7,
    ),
    _skill(
        "For HTTP servers, FastAPI (async-first, pydantic-native) for APIs; Flask if you need sync simplicity.",
        "python:web", ("fastapi", "flask"), 0.7,
    ),
    _skill(
        "For ML: PyTorch for research and most production; JAX for high-end numerics; transformers for HF models.",
        "python:ml", ("ml", "pytorch"), 0.6,
    ),
    _skill(
        "For data wrangling, polars (faster) is supplanting pandas; both are fine on small data.",
        "python:data", ("polars", "pandas"), 0.6,
    ),
    _skill(
        "Use `subprocess.run([...], check=True, capture_output=True, text=True)` to run a command and get its output.",
        "python:shell", ("subprocess",), 0.7,
    ),
    _skill(
        "For random sampling, use `random.SystemRandom` for security-sensitive work, plain `random` for reproducible simulation (with seed).",
        "python:random", ("random", "security"), 0.6,
    ),
    _skill(
        "For zipping, gzipping, tarring: stdlib zipfile / gzip / tarfile. Don't shell out unless you have to.",
        "python:io", ("archives", "stdlib"), 0.55,
    ),
    _skill(
        "Write tests in pytest. Use `@pytest.fixture` for setup. Use `tmp_path` instead of /tmp; use `monkeypatch` to scope changes.",
        "python:testing", ("pytest", "fixtures"), 0.7,
    ),
    _skill(
        "For mocking external services in tests, use unittest.mock.patch as a decorator or context manager. Prefer dependency injection where possible.",
        "python:testing", ("mock", "testing"), 0.65,
    ),
    _skill(
        "For property-based testing, use hypothesis. Especially good for parsers, normalizers, and data transformations.",
        "python:testing", ("hypothesis", "property"), 0.6,
    ),
    _skill(
        "For logging, configure once in main() via logging.basicConfig(); use logging.getLogger(__name__) per-module.",
        "python:logging", ("logging",), 0.65,
    ),
    _skill(
        "For structured logging, use a JSON formatter (e.g. python-json-logger) so logs are queryable.",
        "python:logging", ("logging", "json"), 0.6,
    ),
    _skill(
        "For retries with backoff, use the `tenacity` library — declarative @retry decorators, exponential backoff included.",
        "python:reliability", ("retry", "tenacity"), 0.6,
    ),
    _skill(
        "For task queues, use celery (heavy, mature) or arq (asyncio-native, lighter).",
        "python:queues", ("celery", "arq"), 0.55,
    ),
    _skill(
        "For locking across processes, use file locks (filelock library) or a Redis-based lock; not threading.Lock.",
        "python:concurrency", ("locks", "filelock"), 0.6,
    ),
    _skill(
        "Profile before optimizing: `cProfile`, `py-spy`, or pytest-benchmark for micro-benchmarks.",
        "python:performance", ("profiling",), 0.65,
    ),

    # ---- Expanded facts ----
    _fact(
        "Python 3.13 ships with the experimental free-threaded build (no GIL); production code shouldn't depend on it yet.",
        "python:concurrency", ("gil", "3.13"), 0.5,
    ),
    _fact(
        "asyncio coroutines must be awaited; calling one without await returns the coroutine object without running it.",
        "python:async", ("asyncio", "coroutines"), 0.7,
    ),
    _fact(
        "Pydantic v2 is a complete rewrite — model_config replaces Config class, model_validator replaces root_validator.",
        "python:pydantic", ("pydantic", "v2"), 0.65,
    ),
    _fact(
        "SQLAlchemy 2.0 uses `select(Model).where(...)`, not the legacy `Model.query`. Sessions are explicit; commit/rollback yourself.",
        "python:db", ("sqlalchemy", "2.0"), 0.65,
    ),
    _fact(
        "FastAPI dependency injection is via `Depends(...)`; dependencies can return generators for setup/teardown.",
        "python:web", ("fastapi", "depends"), 0.6,
    ),
    _fact(
        "JSON has no native datetime; use isoformat() strings and parse back with datetime.fromisoformat().",
        "python:serialization", ("json", "datetime"), 0.6,
    ),
    _fact(
        "Python lists are O(1) append, O(n) prepend. For frequent prepends, collections.deque is O(1) on both ends.",
        "python:data-structures", ("list", "deque"), 0.6,
    ),
    _fact(
        "dict and set lookups are O(1) average; sorted iteration of dict preserves insertion order (3.7+).",
        "python:data-structures", ("dict", "set"), 0.55,
    ),
    _fact(
        "Python integers have arbitrary precision; floats are IEEE 754 doubles (53-bit mantissa).",
        "python:numerics", ("int", "float"), 0.5,
    ),
    _fact(
        "Type hints are not enforced at runtime by Python itself; use Pydantic, dataclasses, or runtime checkers like beartype if you need enforcement.",
        "python:typing", ("typing", "runtime"), 0.55,
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
