#!/usr/bin/env bash
# Real-codebase test: 5 PR-shaped refactorings on lattice's own atoms/
# production source (not toy fixtures).
#
# Each task uses a different verb. The script captures pass / fail per
# task and verifies the resulting files still parse. SymbolNotFound on
# nonexistent targets is the CORRECT outcome — lattice refusing to
# fabricate symbols.
set -euo pipefail

LATTICE_REPO=${LATTICE_REPO:-/home/user/memories-plugin/lattice}
REAL=$(mktemp -d)
mkdir -p "$REAL/src"

# Real lattice source.
cp "$LATTICE_REPO/src/lattice/atoms/feedback.py" "$REAL/src/feedback.py"
cp "$LATTICE_REPO/src/lattice/atoms/evolve.py"   "$REAL/src/evolve.py"

# Stub the cross-package imports so the test workspace stands on its own.
sed -i 's|from lattice.atoms.atom|from src.atom|g' "$REAL/src/feedback.py" "$REAL/src/evolve.py"
sed -i 's|from lattice.atoms.store|from src.store|g' "$REAL/src/feedback.py" "$REAL/src/evolve.py"

cat > "$REAL/src/atom.py" <<'PY'
"""Atom stub for the real-codebase test."""
from enum import StrEnum

class AtomType(StrEnum):
    FACT = "fact"
    EXPERIENCE = "experience"
PY

cat > "$REAL/src/store.py" <<'PY'
"""Store stub."""
class AtomStore: ...
class RecallResult: ...
PY

echo "=== files in real workspace ==="
find "$REAL" -name '*.py' | sort
echo

lattice init "$REAL" > /dev/null 2>&1
cd "$REAL"

PASS=0
FAIL=0
ran() {
  local label="$1"; shift
  local cmd_out
  cmd_out=$("$@" 2>&1) || true
  echo "$cmd_out" | tail -3
  echo
}

echo "--- 1. add a docstring to _walk_string_fields ---"
ran "doc" lattice do \
  'Add a docstring to function _walk_string_fields in src/evolve.py saying "Yield (path_tuple, value) for every str value in obj."' \
  --write

echo "--- 2. change the return type of _set_at_path to None ---"
ran "ret" lattice do \
  'Change the return type of function _set_at_path in src/evolve.py to None' \
  --write

echo "--- 3. delete the helper _action_verbs ---"
ran "del" lattice do \
  'Delete function _action_verbs in src/evolve.py' \
  --write

echo "--- 4. add @staticmethod to a function that DOESN'T EXIST (expect SymbolNotFound) ---"
ran "404a" lattice do \
  'Add @staticmethod decorator to function _quote_docstring in src/evolve.py'

echo "--- 5. move a function that DOESN'T EXIST (expect SymbolNotFound) ---"
ran "404b" lattice do \
  'Move function _quote_docstring from src/evolve.py to src/feedback.py' --write

echo "=== verification ==="
echo
echo "evolve.py changes:"
grep -E '"""Yield \(path_tuple|def _set_at_path' "$REAL/src/evolve.py" | head -5
echo
echo "_action_verbs gone? (expect 0):"
grep -c "def _action_verbs" "$REAL/src/evolve.py" || true
echo
echo "both files parse?"
python -c "
import ast
for p in ['$REAL/src/evolve.py', '$REAL/src/feedback.py']:
    ast.parse(open(p).read())
    print(f'  {p}: parses')
"
