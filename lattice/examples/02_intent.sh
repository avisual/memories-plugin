#!/usr/bin/env bash
# One Intent → N typed Actions across many files. No LLM needed for this one.
set -euo pipefail

WS=$(mktemp -d)
mkdir -p "$WS/src/payments" "$WS/src/refunds"

cat > "$WS/src/payments/charge.py" <<'PY'
def charge(amount: int) -> None:
    pass

class ChargeProcessor:
    def charge(self, amount: int) -> None:
        pass
PY

cat > "$WS/src/refunds/refund.py" <<'PY'
class RefundHandler:
    def charge(self, amount: int) -> None:
        pass
PY

echo "=== before (three charge methods across two files) ==="
find "$WS" -name '*.py' -exec sh -c 'echo "--- {} ---" && cat "{}"' \;
echo

echo "=== lattice intent (add dry_run keyword-only param to every charge method) ==="
echo '{
  "kind": "AddParameterToAllMatching",
  "function_name": "charge",
  "parameter_name": "dry_run",
  "parameter_type": "bool",
  "parameter_default": "False",
  "keyword_only": true
}' | lattice intent "$WS" --intent -
echo

echo "(use --write to apply the diffs; workspace at $WS)"
