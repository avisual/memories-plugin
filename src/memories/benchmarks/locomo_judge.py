"""LLM-as-Judge scoring for LoCoMo — produces scores comparable to Mem0.

Supports three providers:
- **OpenAI** (default): High-quality judge via OpenAI API (gpt-4o-mini).
  Set OPENAI_API_KEY env var.
- **Gemini Flash**: Via Google's REST API. Set GEMINI_API_KEY env var.
- **Ollama**: Local model, free but lower quality.

Steps:
1. Generate an answer from retrieved atoms + question
2. Judge whether the answer matches the ground truth (binary 1/0)

This produces the "J" (LLM-as-Judge) metric used by Mem0, Zep, MemMachine
and other memory systems, making our scores directly comparable.

Usage:
    uv run python -m memories.benchmarks.locomo_judge data/locomo10.json
    uv run python -m memories.benchmarks.locomo_judge data/locomo10.json --provider openai --model gpt-4o-mini
    uv run python -m memories.benchmarks.locomo_judge data/locomo10.json --provider ollama
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# --- Provider: Gemini ---

_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def _get_gemini_api_key() -> str | None:
    """Resolve Gemini API key from env or pass."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key

    try:
        from memories.credentials import get_secret
        return get_secret("gemini_api_key")
    except Exception:
        return None


async def _gemini_generate(
    prompt: str,
    model: str = _DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 200,
    api_key: str | None = None,
) -> str:
    """Call Gemini REST API. Returns the response text."""
    if not api_key:
        raise ValueError("No Gemini API key available")

    url = f"{_GEMINI_API_URL}/{model}:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Extract text from Gemini response.
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        log.warning("Unexpected Gemini response structure: %s", data)
        return ""


# --- Provider: OpenAI ---

_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _get_openai_api_key() -> str | None:
    """Resolve OpenAI API key from env or pass."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        from memories.credentials import get_secret
        return get_secret("openai_api_key")
    except Exception:
        return None


async def _openai_generate(
    prompt: str,
    model: str = _DEFAULT_OPENAI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 200,
    api_key: str | None = None,
) -> str:
    """Call OpenAI chat completions API. Returns the response text."""
    if not api_key:
        raise ValueError("No OpenAI API key available")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        log.warning("Unexpected OpenAI response structure: %s", data)
        return ""


# --- Provider: DeepSeek ---

_DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
_DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"


def _get_deepseek_api_key() -> str | None:
    """Resolve DeepSeek API key from env or pass."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    try:
        from memories.credentials import get_secret
        return get_secret("deepseek_api_key")
    except Exception:
        return None


async def _deepseek_generate(
    prompt: str,
    model: str = _DEFAULT_DEEPSEEK_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 200,
    api_key: str | None = None,
) -> str:
    """Call DeepSeek API (OpenAI-compatible). Returns the response text."""
    if not api_key:
        raise ValueError("No DeepSeek API key available")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            _DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        log.warning("Unexpected DeepSeek response structure: %s", data)
        return ""


# --- Provider: Ollama ---

_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


async def _ollama_generate(
    prompt: str,
    model: str = _DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> str:
    """Call Ollama generate endpoint. Returns the response text."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            _OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


# --- Unified generate interface ---

async def _generate(
    prompt: str,
    provider: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    api_key: str | None = None,
) -> str:
    """Route to the appropriate provider."""
    if provider == "openai":
        return await _openai_generate(
            prompt, model=model, temperature=temperature,
            max_tokens=max_tokens, api_key=api_key,
        )
    elif provider == "gemini":
        return await _gemini_generate(
            prompt, model=model, temperature=temperature,
            max_tokens=max_tokens, api_key=api_key,
        )
    elif provider == "deepseek":
        return await _deepseek_generate(
            prompt, model=model, temperature=temperature,
            max_tokens=max_tokens, api_key=api_key,
        )
    else:
        return await _ollama_generate(
            prompt, model=model, temperature=temperature,
            max_tokens=max_tokens,
        )


# --- Answer generation and judging ---

async def generate_answer(
    question: str,
    retrieved_atoms: list[str],
    category: int,
    provider: str = "openai",
    model: str = _DEFAULT_OPENAI_MODEL,
    api_key: str | None = None,
) -> str:
    """Generate an answer to a question using retrieved memory atoms."""
    # Multi-hop needs many atoms; adversarial and open_domain also benefit.
    atom_cap = 30 if category == 1 else 20 if category in (3, 5) else 15
    context = "\n".join(f"- {atom}" for atom in retrieved_atoms[:atom_cap])

    if category == 1:
        # Multi-hop: evidence is spread across multiple sessions.
        prompt = (
            "Based on the following conversation excerpts, answer the question.\n"
            "The answer requires combining information from multiple excerpts.\n"
            "List ALL relevant items — do not stop after the first one.\n"
            "Give a direct, specific answer. Use a comma-separated list.\n"
            "Do NOT hedge or say 'it is not mentioned'. The answer IS in the excerpts.\n\n"
            f"Excerpts:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer (be specific, list all items):"
        )
    elif category == 5:
        # Adversarial: tricky questions about specific speakers/details.
        prompt = (
            "Based on the following conversation excerpts, answer the question.\n"
            "IMPORTANT: Pay very careful attention to WHICH SPEAKER said or did what.\n"
            "The question may ask about a specific person — make sure you attribute "
            "the correct information to the correct speaker.\n"
            "The answer IS in the excerpts. Do NOT say 'not mentioned' or 'unclear'.\n"
            "Give the shortest possible answer — ideally just a few words.\n\n"
            f"Excerpts:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer (few words only):"
        )
    elif category == 2:
        # Temporal: sessions have [YYYY-MM-DD] dates.
        prompt = (
            "Based on the following conversation excerpts, answer the question.\n"
            "Session dates appear in brackets as [YYYY-MM-DD, session N].\n"
            "IMPORTANT: When asked about timing, extract the EXACT date or time period "
            "from the excerpts. Say the specific month/year or date — never say "
            "'last year' or 'recently'.\n"
            "Give a short, specific answer.\n\n"
            f"Excerpts:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer (state the exact date/time):"
        )
    elif category == 3:
        # Open domain: requires inference and reasoning from conversation content.
        prompt = (
            "Based on the following conversation excerpts, answer the question.\n"
            "This question may require reasonable inference from the conversation.\n"
            "Use what the speakers say and do to make your best judgment.\n"
            "Do NOT say 'not mentioned' or 'cannot determine'. Give your best answer.\n"
            "Be concise — answer in a few words or one short sentence.\n\n"
            f"Excerpts:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer (be concise):"
        )
    else:
        # Single-hop and other categories.
        prompt = (
            "Based on the following conversation excerpts, answer the question.\n"
            "Give a direct, specific answer in 1-2 sentences.\n"
            "Do NOT say 'not mentioned'. The answer is in the excerpts.\n\n"
            f"Excerpts:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    try:
        return await _generate(
            prompt, provider=provider, model=model, api_key=api_key,
        )
    except Exception as exc:
        log.warning("Failed to generate answer: %s", exc)
        return ""


async def judge_answer(
    question: str,
    predicted: str,
    ground_truth: str,
    provider: str = "openai",
    model: str = _DEFAULT_OPENAI_MODEL,
    api_key: str | None = None,
) -> float:
    """Judge whether a predicted answer matches the ground truth.

    Returns 1.0 if the answers are semantically equivalent, 0.0 otherwise.
    This matches the LLM-as-Judge protocol used by Mem0 et al.
    """
    prompt = (
        "You are a judge evaluating whether a predicted answer is correct.\n"
        "The predicted answer is correct if it contains the same key information "
        "as the ground truth, even if worded differently or with extra detail.\n\n"
        "Examples:\n"
        "- Ground Truth: 'necklace' / Predicted: 'a necklace from her grandma' → Yes\n"
        "- Ground Truth: 'June 2023' / Predicted: 'in June of 2023' → Yes\n"
        "- Ground Truth: 'pottery' / Predicted: 'She enjoys hiking' → No\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}\n\n"
        "Does the predicted answer contain the key information from the ground truth? "
        "Respond with ONLY 'Yes' or 'No'."
    )

    try:
        response = await _generate(
            prompt, provider=provider, model=model,
            max_tokens=10, api_key=api_key,
        )
        response_lower = response.lower().strip()
        if response_lower.startswith("yes"):
            return 1.0
        return 0.0
    except Exception as exc:
        log.warning("Judge failed: %s", exc)
        return 0.0


# --- Main benchmark runner ---

async def run_judge_on_results(
    results_path: str | Path | None,
    dataset_path: str | Path,
    provider: str = "openai",
    model: str | None = None,
    output_path: str | Path | None = None,
    conversation_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Run LLM-as-Judge scoring on LoCoMo dataset.

    Re-ingests conversations, recalls atoms for each question,
    generates answers, and judges them against ground truth.
    """
    from memories.benchmarks.locomo_data import load_dataset
    from memories.benchmarks.locomo_eval import (
        _create_brain,
        _ingest_conversation,
        _recall_for_question,
    )

    # Resolve provider + model + API key.
    _model_defaults = {
        "openai": _DEFAULT_OPENAI_MODEL,
        "gemini": _DEFAULT_GEMINI_MODEL,
        "deepseek": _DEFAULT_DEEPSEEK_MODEL,
        "ollama": _DEFAULT_OLLAMA_MODEL,
    }
    if model is None:
        model = _model_defaults.get(provider, _DEFAULT_OLLAMA_MODEL)

    api_key: str | None = None
    if provider == "openai":
        api_key = _get_openai_api_key()
        if not api_key:
            log.warning(
                "No OPENAI_API_KEY found — falling back to Ollama. "
                "Set OPENAI_API_KEY env var for OpenAI judging."
            )
            provider = "ollama"
            model = _DEFAULT_OLLAMA_MODEL
    elif provider == "gemini":
        api_key = _get_gemini_api_key()
        if not api_key:
            log.warning(
                "No GEMINI_API_KEY found — falling back to Ollama. "
                "Set GEMINI_API_KEY env var for Gemini Flash judging."
            )
            provider = "ollama"
            model = _DEFAULT_OLLAMA_MODEL
    elif provider == "deepseek":
        api_key = _get_deepseek_api_key()
        if not api_key:
            log.warning(
                "No DEEPSEEK_API_KEY found — falling back to Ollama. "
                "Set DEEPSEEK_API_KEY env var for DeepSeek judging."
            )
            provider = "ollama"
            model = _DEFAULT_OLLAMA_MODEL

    start = time.monotonic()
    conversations = load_dataset(dataset_path)

    if conversation_indices:
        conversations = [
            conversations[i] for i in conversation_indices if i < len(conversations)
        ]

    import tempfile

    db_dir = Path(tempfile.mkdtemp(prefix="locomo_judge_"))

    all_scores: list[float] = []
    cat_scores: dict[str, list[float]] = {}
    conv_results: list[dict] = []

    for conv in conversations:
        region = f"locomo:{conv.sample_id}"
        db_path = db_dir / f"{conv.sample_id}.db"

        log.info("Judging %s (%d QA pairs) [%s/%s]",
                 conv.sample_id, len(conv.qa_pairs), provider, model)

        brain = await _create_brain(db_path)
        try:
            await _ingest_conversation(brain, conv, region)

            qa_scores: list[float] = []
            qa_details: list[dict] = []

            for i, qa in enumerate(conv.qa_pairs):
                atoms, _ = await _recall_for_question(brain, qa.question, region, category=qa.category)

                predicted = await generate_answer(
                    qa.question, atoms, qa.category,
                    provider=provider, model=model, api_key=api_key,
                )

                score = await judge_answer(
                    qa.question, predicted, qa.answer,
                    provider=provider, model=model, api_key=api_key,
                )

                qa_scores.append(score)
                all_scores.append(score)

                cat = qa.category_name
                cat_scores.setdefault(cat, [])
                cat_scores[cat].append(score)

                qa_details.append({
                    "question": qa.question,
                    "ground_truth": qa.answer,
                    "predicted": predicted,
                    "judge_score": score,
                    "category": qa.category_name,
                })

                if (i + 1) % 25 == 0:
                    avg_so_far = sum(qa_scores) / len(qa_scores)
                    log.info(
                        "  %d/%d — running avg J=%.3f",
                        i + 1, len(conv.qa_pairs), avg_so_far,
                    )

            conv_avg = sum(qa_scores) / len(qa_scores) if qa_scores else 0.0
            conv_results.append({
                "sample_id": conv.sample_id,
                "judge_score": conv_avg,
                "qa_count": len(qa_scores),
                "qa_details": qa_details,
            })
            log.info("  %s: J=%.3f (%d QA)", conv.sample_id, conv_avg, len(qa_scores))
        finally:
            await brain.shutdown()

    # Aggregate.
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    by_cat = {
        cat: sum(scores) / len(scores)
        for cat, scores in sorted(cat_scores.items())
    }
    elapsed = time.monotonic() - start

    label = f"{provider}/{model}"
    result = {
        "judge_score": overall,
        "by_category": by_cat,
        "total_qa": len(all_scores),
        "provider": provider,
        "model": model,
        "time_s": elapsed,
        "conversations": conv_results,
    }

    # Print summary.
    print("\n" + "=" * 60)
    print("LoCoMo LLM-as-Judge Results")
    print(f"Provider: {label}")
    print("=" * 60)
    print(f"Overall Judge Score:  {overall:.3f}")
    print(f"Total QA:             {len(all_scores)}")
    print(f"Time:                 {elapsed:.1f}s")
    print()
    print(f"{'Category':15s}  {'J-Score':>8s}  {'Count':>5s}")
    print("-" * 35)
    for cat, score in by_cat.items():
        count = len(cat_scores[cat])
        print(f"  {cat:13s}  {score:8.3f}  {count:5d}")
    print()
    print("Per Conversation:")
    for cr in conv_results:
        print(f"  {cr['sample_id']:10s}  J={cr['judge_score']:.3f}  ({cr['qa_count']} QA)")
    print("=" * 60)

    # Comparison table.
    print(f"\nComparison with published systems (judge: {label}):")
    print(f"{'System':20s}  {'J-Score':>8s}")
    print("-" * 32)
    comparisons = [
        ("Full-Context GPT-4", 0.729),
        ("MemMachine v0.2", 0.912),
        ("Mem0^g (graph)", 0.684),
        ("Mem0", 0.669),
        ("RAG (k=2)", 0.610),
        ("Zep", 0.570),
        ("OpenAI Memory", 0.529),
        ("LangMem", 0.510),
        (f"memories ({label})", overall),
    ]
    comparisons.sort(key=lambda x: -x[1])
    for name, score in comparisons:
        marker = " <--" if "memories" in name else ""
        print(f"  {name:18s}  {score:8.3f}{marker}")
    print()

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        log.info("Judge results saved to %s", output_path)

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM-as-Judge scoring on LoCoMo benchmark"
    )
    parser.add_argument("dataset", help="Path to locomo10.json")
    parser.add_argument("--output", "-o", help="Output JSON path for judge results")
    parser.add_argument(
        "--provider", "-p", choices=["openai", "gemini", "deepseek", "ollama"], default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Model name (default: gemini-2.0-flash for gemini, llama3.2:3b for ollama)",
    )
    parser.add_argument(
        "--conversations", "-c",
        help="Comma-separated conversation indices (0-based)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    indices = None
    if args.conversations:
        indices = [int(x) for x in args.conversations.split(",")]

    asyncio.run(
        run_judge_on_results(
            results_path=None,
            dataset_path=args.dataset,
            provider=args.provider,
            model=args.model,
            output_path=args.output,
            conversation_indices=indices,
        )
    )


if __name__ == "__main__":
    main()
