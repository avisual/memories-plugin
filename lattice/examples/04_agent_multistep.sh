#!/usr/bin/env bash
# The stack-machine model: the agent loop runs until the next
# instruction would be a no-op (the goal is observably met), then
# exits with "stuck" — which the harness treats as success because
# the file IS in the desired state.
#
# No MarkDone required from the LLM. Observable state controls the
# program counter, not the LLM's self-report.
set -euo pipefail

WS=$(mktemp -d)
mkdir -p "$WS/src"
cat > "$WS/src/main.py" <<'PY'
"""Project entrypoint."""


def main() -> None:
    pass
PY

echo "=== before ==="
cat "$WS/src/main.py"
echo

lattice init "$WS"
echo

echo "=== lattice agent: one task, max 4 cycles, harness decides when done ==="
lattice agent "$WS" \
  --atom-db "$WS/.lattice/brain.db" \
  --task "Add an import of the json module to the file at src/main.py" \
  --max-steps 4 \
  --write
echo

echo "=== after ==="
cat "$WS/src/main.py"
echo
echo "Cycle 1: model emits AddImport(json) -> applied"
echo "Cycle 2: model emits AddImport(json) again -> no-op (file already has it)"
echo "Cycle 3: same -> harness's stuck detector fires -> exit"
echo "Result: file in target state, written to disk."
