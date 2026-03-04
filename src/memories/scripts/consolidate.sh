#!/bin/bash
# Daily memory consolidation runner.
# Invoked by the com.avisual.memories.consolidate LaunchAgent.
# Runs the full consolidation cycle: decay, prune, merge, promote.
set -euo pipefail

# Adjust this path to match your installation.
exec python -m memories reflect
