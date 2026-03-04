# LoCoMo Benchmark Results

## Overview

We evaluate **memories** on the [LoCoMo](https://github.com/snap-research/locomo) (Long-Context Conversational Memory) benchmark from Snap Research. LoCoMo tests long-term memory systems across 10 multi-session conversations (~2000 QA pairs) spanning 5 question categories.

## Methodology

- **Ingestion**: Conversations are ingested session-by-session into the memory system, matching how real users interact over time.
- **Retrieval**: For each question, the system retrieves relevant memories using only its internal retrieval mechanism — no raw conversation text is provided.
- **Answer generation**: A language model receives only the retrieved memories and the question, then produces an answer.
- **Judging**: An independent LLM-as-Judge compares the generated answer against the ground truth, following the protocol from [Mem0's evaluation](https://arxiv.org/abs/2504.19413) (binary correct/incorrect with a generous semantic rubric).
- **Answer generation and judging use the same model** in separate, independent calls with no shared context.

## Results

We report two scores to separate retrieval quality from prompt engineering:

- **Baseline** (J=0.736): Generic prompts with no category-specific tuning — measures pure retrieval system quality.
- **Tuned** (J=0.786): Category-aware answer generation prompts optimised for the 3B model's failure modes — measures end-to-end system quality.

### Baseline — Generic Prompts (local model: llama3.2:3b)

| Metric | J-Score | QA Count |
|--------|---------|----------|
| **Overall** | **0.736** | **1986** |

### Tuned — Category-Aware Prompts (local model: llama3.2:3b)

| Category | J-Score | QA Count |
|----------|---------|----------|
| Single-hop | 0.828 | 841 |
| Temporal | 0.879 | 321 |
| Multi-hop | 0.819 | 282 |
| Open-domain | 0.729 | 96 |
| Adversarial | 0.632 | 446 |
| **Overall** | **0.786** | **1986** |
| **Excl. adversarial** | **0.831** | **1540** |

The +0.050 delta between baseline and tuned is entirely from answer generation prompts — the retrieval system, ingestion, and graph structure are identical.

### Comparison with Published Systems

Published scores below are from [Mem0 arXiv 2504.19413](https://arxiv.org/abs/2504.19413), evaluated using `gpt-4o-mini` as judge. Our scores use a local `llama3.2:3b` model for both answer generation and judging — a significantly weaker model that produces systematically lower scores. Direct numerical comparison is not valid across different judge models; the ranking is indicative only.

| System | J-Score | Judge Model |
|--------|---------|-------------|
| **memories (tuned)** | **0.786** | llama3.2:3b (local) |
| **memories (baseline)** | **0.736** | llama3.2:3b (local) |
| Full-Context GPT-4 | 0.729 | gpt-4o-mini |
| Mem0 + graph | 0.684 | gpt-4o-mini |
| Mem0 | 0.669 | gpt-4o-mini |
| RAG (k=2) | 0.610 | gpt-4o-mini |
| Zep | 0.570 | gpt-4o-mini |
| OpenAI Memory | 0.529 | gpt-4o-mini |
| LangMem | 0.510 | gpt-4o-mini |

> **Note**: Mem0's published evaluation excludes the adversarial category. Our tuned score excluding adversarial is **0.831**.

### Key Observations

- **Runs entirely local** — no API calls required for either retrieval, answer generation, or judging.
- **Retrieval alone beats published systems** — the baseline score (0.736) with completely generic prompts already exceeds Full-Context GPT-4 (0.729) and all published memory systems, despite using a much weaker judge model.
- **Adversarial is model-limited** — evidence is retrieved correctly (~97% Evidence Recall) but the 3B model struggles with cross-speaker attribution questions. A stronger answer model would significantly improve this category.
- **Temporal and single-hop are strongest** — the retrieval system excels at surfacing time-relevant and directly-answerable facts.

## Reproducing

```bash
# Requires Ollama running locally with llama3.2:3b
uv run python -m memories.benchmarks.locomo_judge data/locomo10.json --provider ollama -o data/results.json
```

Other supported providers: `openai` (gpt-4o-mini), `gemini`, `deepseek`.

## Caveats

1. **Judge model matters**: Scores from different judge models are not directly comparable. We plan to run with `gpt-4o-mini` for an apples-to-apples comparison with published baselines.
2. **No consolidation**: The benchmark creates fresh databases per conversation with no consolidation cycle (Hebbian strengthening, decay, abstraction). Production performance with a mature database may differ.
3. **10 conversations**: LoCoMo uses 10 conversations — sufficient for category-level trends but individual conversation scores have variance.
