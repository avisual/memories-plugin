"""Lattice task benchmark.

A small corpus of concrete coding tasks plus a runner that:
1. Sets up a fresh tempdir workspace for each task.
2. Writes the task's fixture files.
3. Runs `lattice do <task>` with the task's CLI flags.
4. Checks that the resulting files contain (or don't contain) the
   declared substrings.

The goal is to answer 'does lattice solve N/M real coding tasks?'
with a number, not a cherry-picked demo. Run with:

    python -m lattice.benchmarks.run

Output is a pass/fail table to stdout. Exit code = number of
failures (so CI can fail the build if regressions land).
"""
