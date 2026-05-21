"""TwoStageProposer — split the LLM call into Planner + Executor.

A small (sub-1.5B) instruction-tuned model has trouble doing two
things at once: deciding WHICH verb fits the task AND filling in the
right slot values. Splitting the decision drops accuracy on each
step:

  Stage 1 (Planner):   given (task, atoms, history) -> pick ONE verb
                       (a 12-way classification — easy for a small model).
  Stage 2 (Executor):  given (task, atoms, picked-verb) -> emit valid JSON
                       for THAT verb only (much narrower schema, far fewer
                       degrees of freedom).

Both stages can share the same underlying model — that's the
"RecursiveMAS lightweight" footprint (paper: https://recursivemas.github.io,
sub-1.5B agents). For better quality, plug in two different models —
e.g. Qwen 0.5B for the Planner (fast classification), Qwen 1.5B for
the Executor (better slot-filling). The architecture stays the same.

State sharing between stages is via the existing atom store and the
ObservationContext — the "telepathic" channel of the paper, here
realised concretely as written atoms + hint rendering rather than
hidden-state transfer (that requires open-weight model surgery).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from pydantic import ValidationError

from lattice.actions import Action, parse_action
from lattice.propose.base import ObservationContext, Proposer, ProposerError
from lattice.propose.local import (
    _DEFAULT_MODEL,
    _coerce_loose,
    _extract_json,
    _render_observation,
)


_VERB_CHOICES: tuple[str, ...] = (
    "AddImport",
    "RenameSymbol",
    "AddField",
    "AddParameter",
    "WrapInTry",
    "AddTest",
    "RecallMore",
    "RevealBody",
    "MarkBlocked",
    "MarkDone",
    "Branch",
    "Research",
)


_VERB_HINTS: dict[str, str] = {
    "AddImport": "Add an import to a Python file.",
    "RenameSymbol": "Rename a function or class (and its references).",
    "AddField": "Add an attribute to a class body.",
    "AddParameter": "Add a parameter to a function or method.",
    "WrapInTry": "Wrap a line span in try/except.",
    "AddTest": "Add a pytest test function for a target.",
    "RecallMore": "Ask the brain for more context (non-mutating).",
    "RevealBody": "Reveal the source of a known symbol (non-mutating).",
    "MarkBlocked": "Cannot proceed; needs human input.",
    "MarkDone": "Task complete; the next instruction would be a no-op.",
    "Branch": "Try an alternative branch (no-op in linear loops).",
    "Research": "Fetch a documentation URL into the brain right now.",
}


_PLANNER_SYSTEM = (
    "You pick the next VERB the agent should emit. Read the TASK, the "
    "RECENT HISTORY, and the brain HINTS. Output exactly one word — the "
    "verb name — and nothing else. No JSON, no punctuation, no commentary.\n\n"
    "Available verbs (pick exactly one):\n"
    + "\n".join(f"  {v} — {_VERB_HINTS[v]}" for v in _VERB_CHOICES)
    + "\n\nRules:\n"
    "- If FILES ALREADY EDITED contains the targets and the task is done, "
    "answer MarkDone.\n"
    "- If the task mentions a library you don't see described in HINTS, "
    "answer Research.\n"
    "- Otherwise pick the verb that most directly accomplishes the next step."
)


_EXECUTOR_SYSTEM_TEMPLATE = (
    "You are filling in a JSON action for the LATTICE coding harness.\n"
    "The verb has ALREADY been chosen for you: {verb}.\n"
    "Emit exactly one JSON object matching this verb's schema. "
    "No prose, no markdown fences.\n\n"
    "Schema (use these exact field names):\n{schema}\n\n"
    "Use the file paths from WORKSPACE FILES verbatim."
)


_VERB_SCHEMAS: dict[str, str] = {
    "AddImport": (
        '{"verb":"AddImport","file":{"path":"<rel.py>"},'
        '"module":"<dotted.python.identifier>",'
        '"names":["A","B"]?,"alias":"<id>"?,"confidence":0..1}'
    ),
    "RenameSymbol": (
        '{"verb":"RenameSymbol","symbol":{"file":"<rel.py>","name":"<DottedName>"},'
        '"new_name":"<NewName>","confidence":0..1}'
    ),
    "AddField": (
        '{"verb":"AddField","cls":{"file":"<rel.py>","name":"<ClassName>"},'
        '"name":"<field>","type":{"expr":"<type>"},"default":{"code":"<expr>"}?,'
        '"confidence":0..1}'
    ),
    "AddParameter": (
        '{"verb":"AddParameter","function":{"file":"<rel.py>","name":"<DottedFnName>"},'
        '"name":"<param>","type":{"expr":"<type>"},"default":{"code":"<expr>"}?,'
        '"position":<int>?,"keyword_only":<bool>,"confidence":0..1}'
    ),
    "WrapInTry": (
        '{"verb":"WrapInTry","span":{"file":"<rel.py>","start_line":<n>,"end_line":<m>},'
        '"exception_type":{"expr":"<TypeName>"},"handler_body":[],"confidence":0..1}'
    ),
    "AddTest": (
        '{"verb":"AddTest","target":{"file":"<rel.py>","name":"<DottedFnName>"},'
        '"test_name":"test_<...>",'
        '"given":{"code":"<setup>"},"when":{"code":"<action>"},"then":{"code":"<assert ...>"},'
        '"confidence":0..1}'
    ),
    "RecallMore": '{"verb":"RecallMore","query":"<short query>","confidence":0..1}',
    "RevealBody": (
        '{"verb":"RevealBody","symbol":{"file":"<rel.py>","name":"<DottedFnName>"},'
        '"confidence":0..1}'
    ),
    "MarkBlocked": (
        '{"verb":"MarkBlocked","reason_code":'
        '"ambiguous_intent|missing_context|incompatible_types|external_dependency|needs_human",'
        '"detail":"<short>","confidence":0..1}'
    ),
    "MarkDone": '{"verb":"MarkDone","summary":"<short summary>","confidence":0..1}',
    "Branch": '{"verb":"Branch","rationale":"<short>","confidence":0..1}',
    "Research": (
        '{"verb":"Research","url":"https://...","reason":"<why>","confidence":0..1}'
    ),
}


_VERB_PARSE_RE = re.compile(r"\b(" + "|".join(_VERB_CHOICES) + r")\b")


class TwoStageProposer:
    """Planner-then-Executor proposer.

    Reuses a single LocalLLMProposer-style transformers backend; each
    call to `propose` makes two model invocations (verb + slots). A
    failed parse on stage 2 retries up to `retries` times with the
    validation error appended.
    """

    def __init__(
        self,
        *,
        planner_model: str | None = None,
        executor_model: str | None = None,
        retries: int = 2,
    ) -> None:
        self.planner_model = planner_model or os.environ.get(
            "LATTICE_PLANNER_MODEL", _DEFAULT_MODEL
        )
        self.executor_model = executor_model or os.environ.get(
            "LATTICE_EXECUTOR_MODEL", self.planner_model
        )
        self.retries = retries
        self._planner: Any = None
        self._executor: Any = None
        self._tokenizer: Any = None  # shared when planner==executor

    # ----- model loading -----

    def _load(self) -> None:
        if self._planner is not None and self._executor is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProposerError(
                "transformers + torch not installed. Run: uv pip install -e '.[llm]'"
            ) from exc

        self._planner_tok = AutoTokenizer.from_pretrained(self.planner_model)
        self._planner = AutoModelForCausalLM.from_pretrained(
            self.planner_model, torch_dtype="auto", device_map="cpu"
        )
        self._planner.eval()

        if self.executor_model == self.planner_model:
            self._executor_tok = self._planner_tok
            self._executor = self._planner
        else:
            self._executor_tok = AutoTokenizer.from_pretrained(self.executor_model)
            self._executor = AutoModelForCausalLM.from_pretrained(
                self.executor_model, torch_dtype="auto", device_map="cpu"
            )
            self._executor.eval()

    # ----- generation primitives -----

    def _generate(
        self,
        model: Any,
        tokenizer: Any,
        system: str,
        user: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        import torch

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

    # ----- stage 1: pick a verb -----

    def _plan(self, obs: ObservationContext, *, temperature: float) -> str | None:
        observation_msg = _render_observation(obs)
        out = self._generate(
            self._planner,
            self._planner_tok,
            _PLANNER_SYSTEM,
            observation_msg + "\n\nVerb:",
            max_new_tokens=12,
            temperature=temperature,
        )
        if os.environ.get("LATTICE_LLM_DEBUG"):
            sys.stderr.write(f"[planner] raw: {out.strip()[:60]!r}\n")
        match = _VERB_PARSE_RE.search(out)
        return match.group(1) if match else None

    # ----- stage 2: fill slots -----

    def _execute(
        self,
        obs: ObservationContext,
        verb: str,
        *,
        temperature: float,
    ) -> Action | None:
        schema = _VERB_SCHEMAS.get(verb)
        if not schema:
            return None
        system = _EXECUTOR_SYSTEM_TEMPLATE.format(verb=verb, schema=schema)
        observation_msg = _render_observation(obs)

        correction = ""
        for attempt in range(self.retries + 1):
            user_msg = observation_msg
            if correction:
                user_msg += "\n\n" + correction
            user_msg += "\n\nJSON action:"

            raw = self._generate(
                self._executor,
                self._executor_tok,
                system,
                user_msg,
                max_new_tokens=240,
                temperature=temperature if attempt == 0 else max(temperature, 0.4),
            )
            if os.environ.get("LATTICE_LLM_DEBUG"):
                sys.stderr.write(f"[executor:{verb}] raw: {raw.strip()[:120]!r}\n")
            payload = _extract_json(raw)
            if payload is None:
                correction = (
                    "Your previous response had no JSON object. Emit ONE JSON "
                    "object for verb " + verb + " — no prose, no fences."
                )
                continue
            # Force the verb to match the planner's pick — otherwise the
            # executor sometimes drifts to a different verb under sampling.
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["verb"] = verb
            try:
                return parse_action(_coerce_loose(payload))
            except ValidationError as exc:
                correction = (
                    f"Your JSON failed validation for verb {verb}:\n{exc}\n"
                    "Fix the fields and try again."
                )
                continue
        return None

    # ----- public Proposer API -----

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        self._load()

        actions: list[Action] = []
        seen: set[str] = set()
        # First candidate at temp=0 (most likely verb); subsequent at
        # increasing temps to give the orchestrator's pre-flight diversity.
        plans: list[tuple[float, str | None]] = []
        plans.append((0.0, self._plan(obs, temperature=0.0)))
        for i in range(1, n):
            plans.append((0.5 + 0.15 * (i - 1), self._plan(obs, temperature=0.5 + 0.15 * (i - 1))))

        for temp, verb in plans:
            if verb is None:
                continue
            action = self._execute(obs, verb, temperature=temp)
            if action is None:
                continue
            dump = action.model_dump_json()
            if dump in seen:
                continue
            seen.add(dump)
            actions.append(action)
            if len(actions) >= n:
                break

        if not actions:
            raise ProposerError("two-stage proposer produced no valid action after retries")
        return actions


# Structural Proposer-protocol check (instantiation is deferred; this
# satisfies static checkers without forcing model load at import time).
def _selfcheck() -> None:  # pragma: no cover
    p: Proposer = TwoStageProposer()
    _ = p
