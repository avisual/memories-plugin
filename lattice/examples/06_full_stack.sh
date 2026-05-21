#!/usr/bin/env bash
# Full stack: research + two-stage LLM + atom feedback + write.
#
# Demonstrates the composition that v0.1.1 made possible — small
# parts (PatternProposer fast-path, web fetcher, atom store, planner
# + executor LLM split, ast-verifier) becoming more than themselves.
#
# Requires: source the lattice venv first.
set -euo pipefail

WS=$(mktemp -d)
mkdir -p "$WS/src"
cat > "$WS/src/app.py" <<'PY'
"""Tiny Flask app."""

from flask import Flask


app = Flask(__name__)


@app.route("/")
def index() -> str:
    return "hello"
PY

echo "=== before ==="
cat "$WS/src/app.py"
echo

lattice init "$WS"
echo

echo "=== lattice do (research → atom → recall → action) ==="
lattice do "Add Flask-CORS support to src/app.py. If you don't know the Flask-CORS setup pattern, first emit Research with url=https://flask-cors.readthedocs.io/en/latest/. After researching, emit AddImport for the flask_cors module into src/app.py." \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --two-stage \
  --max-steps 5 \
  --write
echo

echo "=== after ==="
cat "$WS/src/app.py"
echo

echo "=== brain learned an experience atom from this run ==="
lattice atom recall --db "$WS/.lattice/brain.db" --query "flask cors import" --k 2
