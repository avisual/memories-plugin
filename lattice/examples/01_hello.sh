#!/usr/bin/env bash
# Minimum end-to-end: init a brain, run an agent, watch one file change.
set -euo pipefail

WS=$(mktemp -d)
mkdir -p "$WS/src"
cat > "$WS/src/foo.py" <<'PY'
def hello() -> None:
    print("hi")
PY

echo "=== before ==="
cat "$WS/src/foo.py"
echo

echo "=== lattice init ==="
lattice init "$WS"
echo

echo "=== lattice agent ==="
lattice agent "$WS" \
  --task "Add an import of json to src/foo.py" \
  --atom-db "$WS/.lattice/brain.db" \
  --max-steps 3 \
  --write
echo

echo "=== after ==="
cat "$WS/src/foo.py"

echo
echo "(workspace at $WS)"
