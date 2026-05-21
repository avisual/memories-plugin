#!/usr/bin/env bash
# The brain ships smart: a vague task resolves to the right specific
# library via recall. Same agent run, same model, different brain
# states show the recall lift.
set -euo pipefail

WS=$(mktemp -d)
mkdir -p "$WS/src/payments"
cat > "$WS/src/payments/charge.py" <<'PY'
"""Charge processing."""

def charge(amount: int) -> None:
    pass
PY

echo "=== seed the brain ==="
lattice init "$WS"
echo

echo "=== ask the brain: what would surface for a vague payment task? ==="
lattice brain inspect --db "$WS/.lattice/brain.db" \
  --task "I need to take a credit card payment" --k 3
echo

echo "=== run the agent with the seeded brain ==="
lattice agent "$WS" \
  --task "Add an import for the standard Python payment library to src/payments/charge.py" \
  --atom-db "$WS/.lattice/brain.db" \
  --max-steps 3 \
  --write
echo

echo "=== after ==="
cat "$WS/src/payments/charge.py"

echo
echo "(the brain steered the LLM to 'stripe' from the vague phrase 'payment library')"
