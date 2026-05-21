"""LocalLLMProposer — a small instruct model emits typed Actions.

Uses transformers + a small Qwen2.5 instruct (~0.5B params) on CPU.
Prompts the model with a compact human-readable action vocabulary,
extracts the first JSON object from the response, validates it with
Pydantic. On parse failure, retries once with the validation error
appended as a correction signal.

Heavy dependencies (torch, transformers) are imported lazily so the
base lattice install stays small. Install the optional extras:

    uv pip install -e ".[llm]"

The default model is Qwen/Qwen2.5-0.5B-Instruct, chosen because it
runs on a 4-core CPU in seconds and is well-trained on instruction
following. Override via the LATTICE_LLM_MODEL env var or constructor.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from lattice.actions import Action, parse_action
from lattice.propose.base import ObservationContext, Proposer, ProposerError

if TYPE_CHECKING:
    pass


_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_MAX_NEW_TOKENS = 220
_RETRIES = 2


_VOCAB_DOC = """\
Action vocabulary (emit exactly ONE of these as a single JSON object):

AddImport     - {"verb":"AddImport", "file":{"path":"<rel.py>"}, "module":"<mod>", "names":["a","b"]?, "alias":"<a>"?, "confidence":0.5}
RenameSymbol  - {"verb":"RenameSymbol", "symbol":{"file":"<rel.py>","name":"<dotted>"}, "new_name":"<new>", "confidence":0.5}
AddField      - {"verb":"AddField", "cls":{"file":"<rel.py>","name":"<ClassName>"}, "name":"<field>", "type":{"expr":"<type>"}, "default":{"code":"<expr>"}?, "confidence":0.5}
AddParameter  - {"verb":"AddParameter", "function":{"file":"<rel.py>","name":"<dotted>"}, "name":"<param>", "type":{"expr":"<type>"}, "default":{"code":"<expr>"}?, "position":<int>?, "keyword_only":<bool>, "confidence":0.5}
WrapInTry     - {"verb":"WrapInTry", "span":{"file":"<rel.py>","start_line":<int>,"end_line":<int>}, "exception_type":{"expr":"<TypeName>"}, "handler_body":[], "confidence":0.5}
AddTest       - {"verb":"AddTest", "target":{"file":"<rel.py>","name":"<dotted>"}, "test_name":"test_<...>", "given":{"code":"<expr>"}, "when":{"code":"<expr>"}, "then":{"code":"<assert expr>"}, "confidence":0.5}
RecallMore    - {"verb":"RecallMore", "query":"<short query>", "confidence":0.5}
RevealBody    - {"verb":"RevealBody", "symbol":{"file":"<rel.py>","name":"<dotted>"}, "confidence":0.5}
MarkBlocked   - {"verb":"MarkBlocked", "reason_code":"ambiguous_intent|missing_context|incompatible_types|external_dependency|needs_human", "detail":"<short>", "confidence":0.5}
Branch        - {"verb":"Branch", "rationale":"<short>", "confidence":0.5}

Rules:
- Emit ONE JSON object on a single line.
- No markdown code fences. No commentary. JSON only.
- Use confidence in [0.0, 1.0].
- File paths must be repository-relative (no leading /, no ..).
"""


def _render_observation(obs: ObservationContext) -> str:
    parts = [f"TASK: {obs.task}"]
    if obs.symbols:
        parts.append("RELEVANT SYMBOLS:")
        for sym in obs.symbols[:20]:
            parts.append(f"  - {sym.file}::{sym.name} ({sym.kind.value}, line {sym.line})")
    if obs.hints:
        parts.append("HINTS:")
        for h in obs.hints[:6]:
            parts.append(f"  - {h}")
    return "\n".join(parts)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of *text*. None if nothing parses."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    candidates = [stripped]
    match = _JSON_OBJECT_RE.search(stripped)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        depth = 0
        start = None
        for i, ch in enumerate(candidate):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    blob = candidate[start : i + 1]
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        break
    return None


class LocalLLMProposer:
    """Run a small instruct model on CPU, validate output with Pydantic.

    The model is loaded lazily on the first `propose` call. Subsequent
    calls reuse the loaded model. Thread-unsafe; instantiate one per
    orchestrator loop.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = _MAX_NEW_TOKENS,
        retries: int = _RETRIES,
    ) -> None:
        self.model_name = model_name or os.environ.get("LATTICE_LLM_MODEL", _DEFAULT_MODEL)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.retries = retries
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProposerError(
                "transformers + torch not installed. Run: uv pip install -e '.[llm]'"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="cpu",
        )
        self._model.eval()

    def _generate(self, prompt: str, *, sticky_correction: str | None = None) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an action emitter for the LATTICE coding harness. "
                    "Read the task and emit exactly one JSON action from the vocabulary."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if sticky_correction is not None:
            messages.append({"role": "user", "content": sticky_correction})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        import torch

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        self._load()

        prompt = f"{_VOCAB_DOC}\n\n{_render_observation(obs)}\n\nJSON action:"

        correction: str | None = None
        last_error: str | None = None
        for _ in range(self.retries + 1):
            raw = self._generate(prompt, sticky_correction=correction)
            payload = _extract_json(raw)
            if payload is None:
                last_error = "no JSON object found in model output"
                correction = (
                    "Your previous response did not contain valid JSON. "
                    "Emit ONE JSON object only, on a single line, no fences, no prose."
                )
                continue
            try:
                action = parse_action(payload)
            except ValidationError as exc:
                last_error = str(exc)
                correction = (
                    "Your JSON failed schema validation. "
                    f"Errors:\n{exc}\nFix the fields and try again."
                )
                continue
            return [action]

        raise ProposerError(f"local LLM produced no valid action after retries: {last_error}")


# Static check that LocalLLMProposer satisfies the Proposer protocol.
_: Proposer = LocalLLMProposer()
