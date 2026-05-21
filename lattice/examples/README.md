# Examples

Runnable shell scripts demonstrating the LATTICE harness end to end.
Each one is self-contained (uses `/tmp/` for workspaces) and prints
what's happening so you can see the loop.

## Prerequisites

From the repo root (`memories-plugin/`):

```bash
cd lattice
uv venv --python 3.13
uv pip install -e '.[llm]'        # ~1GB: torch + transformers + MiniLM
source .venv/bin/activate           # so `lattice` is on PATH
```

Then run any example:

```bash
bash examples/01_hello.sh
```

First-time runs download model weights to `~/.cache/huggingface/` —
about 500MB (Qwen2.5-0.5B-Instruct) + 90MB (MiniLM). Subsequent
runs use the cache and complete in seconds.

| Script | What it shows |
|---|---|
| `01_hello.sh` | The minimum: init a brain, run an agent, watch one file change. |
| `02_intent.sh` | A single high-level Intent expands into N typed Actions across many files. |
| `03_brain_recall.sh` | The brain ships smart: a vague task ("payment library") resolves to `stripe` via recall. |
| `04_agent_multistep.sh` | The stack-machine model: a 2-edit task done in two cycles by a 0.5B model. |
| `05_brain_grow.sh` | Import a custom atom pack and watch the brain absorb new project knowledge. |

All examples run on a CPU-only machine in under a minute each (after
the model is cached locally).
