"""The benchmark task corpus.

Each task is a small, concrete coding job with a deterministic
expected outcome. Tasks fall into three difficulty tiers:

- 'pattern': PatternProposer handles it deterministically. Fast,
  expected to pass 100% of the time.
- 'llm': needs the local LLM (slot-filling, multi-step, or
  novel-task interpretation). Expected to pass most of the time;
  measured.
- 'research': needs the LLM to emit a Research action and use the
  fetched docs. Hardest tier; expected to pass at least sometimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchTask:
    name: str
    tier: str  # 'pattern' | 'llm' | 'research'
    task: str
    files: dict[str, str]
    expects: dict[str, tuple[str, ...]] = field(default_factory=dict)
    not_expects: dict[str, tuple[str, ...]] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    timeout_s: float = 60.0  # bumped per-task for LLM/research tiers


def _flask_app() -> str:
    return '''\
"""Tiny Flask app."""

from flask import Flask


app = Flask(__name__)


@app.route("/")
def index() -> str:
    return "hello"
'''


def _client_module() -> str:
    return '''\
"""HTTP client."""


class Client:
    base_url: str = ""

    def get(self, path: str) -> str:
        return ""
'''


TASKS: tuple[BenchTask, ...] = (
    # ---- Pattern tier: structural mappings, deterministic ----
    BenchTask(
        name="import-plain",
        tier="pattern",
        task="Add an import of json to src/main.py",
        files={"src/main.py": "def main() -> None:\n    pass\n"},
        expects={"src/main.py": ("import json",)},
    ),
    BenchTask(
        name="import-from",
        tier="pattern",
        task="Add an import of Optional from typing to src/main.py",
        files={"src/main.py": "def main() -> None:\n    pass\n"},
        expects={"src/main.py": ("from typing import Optional",)},
    ),
    BenchTask(
        name="import-alias",
        tier="pattern",
        task="Add an import of numpy as np to src/calc.py",
        files={"src/calc.py": "def add(a, b): return a + b\n"},
        expects={"src/calc.py": ("import numpy as np",)},
    ),
    BenchTask(
        name="rename-cross-file",
        tier="pattern",
        task="Rename charge to take_payment in src/billing.py",
        files={
            "src/billing.py": "def charge(amount: int) -> None:\n    pass\n",
            "src/api.py": (
                "from src.billing import charge\n\n\n"
                "def handler() -> None:\n"
                "    charge(100)\n"
            ),
        },
        expects={
            "src/billing.py": ("def take_payment(",),
            "src/api.py": ("import take_payment", "take_payment(100)"),
        },
        not_expects={
            "src/billing.py": ("def charge(",),
            "src/api.py": ("charge(",),
        },
    ),
    BenchTask(
        name="add-parameter-method",
        tier="pattern",
        task=(
            "Add a keyword-only parameter named timeout of type float with "
            "default 5.0 to function get of class Client in src/api.py"
        ),
        files={"src/api.py": _client_module()},
        expects={"src/api.py": ("*, timeout: float = 5.0",)},
    ),
    BenchTask(
        name="add-field",
        tier="pattern",
        task=(
            "Add a field version of type str with default \"1.0\" to "
            "class Client in src/api.py"
        ),
        files={"src/api.py": _client_module()},
        expects={"src/api.py": ('version: str = "1.0"',)},
    ),

    # ---- LLM tier: needs the model for slot-filling / interpretation ----
    BenchTask(
        name="add-function-route",
        tier="llm",
        task=(
            "Use AddFunction to add a /health route to src/app.py. The "
            "source should be exactly:\n"
            "@app.route(\"/health\")\n"
            "def health() -> dict:\n"
            "    return {\"ok\": True}\n"
        ),
        files={"src/app.py": _flask_app()},
        expects={
            "src/app.py": (
                '@app.route("/health")',
                "def health() -> dict:",
                'return {"ok": True}',
            ),
        },
        flags=("--two-stage", "--model", "Qwen/Qwen2.5-1.5B-Instruct"),
        timeout_s=900.0,
    ),
    BenchTask(
        name="add-statement-cors-wire",
        tier="llm",
        task=(
            "Use AddStatement to insert a single statement cors = CORS(app) "
            "at the end of src/app.py"
        ),
        files={
            "src/app.py": _flask_app() + "\n\nfrom flask_cors import CORS\n",
        },
        expects={"src/app.py": ("cors = CORS(app)",)},
        flags=("--two-stage", "--model", "Qwen/Qwen2.5-0.5B-Instruct"),
        timeout_s=600.0,
    ),
    BenchTask(
        name="multi-step-pattern",
        tier="pattern",
        task=(
            "Add an import of json to src/api.py; add a keyword-only "
            "parameter timeout of type float with default 5.0 to function "
            "get of class Client in src/api.py"
        ),
        files={"src/api.py": _client_module()},
        expects={
            "src/api.py": (
                "import json",
                "*, timeout: float = 5.0",
            ),
        },
        flags=("--decompose",),
    ),
    BenchTask(
        name="wrap-in-try",
        tier="pattern",
        task="Wrap lines 2-3 of src/io.py in a try/except for IOError",
        files={
            "src/io.py": (
                "def read_one(path):\n"
                "    fh = open(path)\n"
                "    return fh.read()\n"
            ),
        },
        expects={"src/io.py": ("try:", "except IOError:")},
    ),
    BenchTask(
        name="add-decorator-route",
        tier="pattern",
        task='Add @app.route("/health") decorator to function health in src/app.py',
        files={
            "src/app.py": (
                "from flask import Flask\n\n\n"
                "app = Flask(__name__)\n\n\n"
                "def health() -> dict:\n"
                "    return {\"ok\": True}\n"
            ),
        },
        expects={"src/app.py": ('@app.route("/health")', "def health()")},
    ),
    BenchTask(
        name="add-decorator-cached",
        tier="pattern",
        task="Add @cached decorator to function compute in src/util.py",
        files={
            "src/util.py": "def compute(n: int) -> int:\n    return n * n\n",
        },
        expects={"src/util.py": ("@cached", "def compute")},
    ),
    BenchTask(
        name="decorate-with-staticmethod",
        tier="pattern",
        task="Decorate function from_dict of class Config in src/cfg.py with @classmethod",
        files={
            "src/cfg.py": (
                "class Config:\n"
                "    def from_dict(self, d: dict) -> 'Config':\n"
                "        return self\n"
            ),
        },
        expects={"src/cfg.py": ("@classmethod", "def from_dict")},
    ),
    BenchTask(
        name="insert-at-start-of-function",
        tier="pattern",
        task=(
            'Insert `logger.info("start")` at the start of function '
            "charge in src/billing.py"
        ),
        files={
            "src/billing.py": (
                "def charge(amount: int) -> None:\n"
                "    process(amount)\n"
            ),
        },
        expects={
            "src/billing.py": ('logger.info("start")', "def charge", "process(amount)"),
        },
    ),
    BenchTask(
        name="insert-at-end-of-method",
        tier="pattern",
        task=(
            "Insert `self._cleanup()` at the end of function close "
            "of class Conn in src/db.py"
        ),
        files={
            "src/db.py": (
                "class Conn:\n"
                "    def close(self) -> None:\n"
                "        self._socket.close()\n"
            ),
        },
        expects={"src/db.py": ("self._cleanup()", "def close")},
    ),

    # ---- Research tier: hardest; LLM must fetch web docs and apply them ----
    BenchTask(
        name="research-flask-cors-import",
        tier="research",
        task=(
            "Add Flask-CORS support to src/app.py. If you don't know the "
            "Flask-CORS setup pattern, first emit Research with "
            "url=https://flask-cors.readthedocs.io/en/latest/. After "
            "researching, emit AddImport for the flask_cors module into "
            "src/app.py."
        ),
        files={"src/app.py": _flask_app()},
        expects={"src/app.py": ("flask_cors", "CORS")},
        flags=("--two-stage", "--model", "Qwen/Qwen2.5-1.5B-Instruct"),
        timeout_s=900.0,
    ),
)
