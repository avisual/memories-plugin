#!/bin/bash
# Wrapper script to run memories with correct PYTHONPATH
export PYTHONPATH="/Users/claudia/git/memories/src:$PYTHONPATH"
exec /Users/claudia/git/memories/.venv/bin/python -m memories "$@"
