#!/usr/bin/env bash
# Grow the brain with a custom atom pack — project-specific knowledge
# becomes a hint to the LLM on every run.
set -euo pipefail

WS=$(mktemp -d)
mkdir -p "$WS/src/billing"
cat > "$WS/src/billing/customer.py" <<'PY'
class Customer:
    def email(self) -> str:
        return ""
PY

lattice init "$WS"

echo "=== custom project atoms ==="
cat > "$WS/.lattice/project_atoms.json" <<'JSON'
[
  {
    "content": "Our project uses 'orjson' instead of stdlib json for performance — never import json directly in src/billing/.",
    "type": "preference",
    "region": "project:billing",
    "importance": 0.9
  },
  {
    "content": "Every Customer instance must have a non-empty email; raise InvalidCustomer otherwise.",
    "type": "skill",
    "region": "project:billing",
    "importance": 0.7
  }
]
JSON
lattice brain import --db "$WS/.lattice/brain.db" "$WS/.lattice/project_atoms.json"
echo

echo "=== brain inspect with project knowledge ==="
lattice brain inspect --db "$WS/.lattice/brain.db" \
  --task "I need to serialize a Customer to JSON in src/billing/customer.py" --k 3
echo

echo "(the project antipattern about orjson surfaces above the generic JSON skill)"
