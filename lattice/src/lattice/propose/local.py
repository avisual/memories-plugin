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

AddImport — add an import statement.
  Three forms:
  (a) plain 'import X':
      task: 'add an import of os to src/main.py'
      → {"verb":"AddImport","file":{"path":"src/main.py"},"module":"os","confidence":0.9}
  (b) 'from X import Y, Z':
      task: 'add an import of request from flask to src/app.py'
      → {"verb":"AddImport","file":{"path":"src/app.py"},"module":"flask","names":["request"],"confidence":0.9}
  (c) 'import X as Y':
      task: 'import numpy as np in src/calc.py'
      → {"verb":"AddImport","file":{"path":"src/calc.py"},"module":"numpy","alias":"np","confidence":0.9}
  Rules: 'module' is a dotted Python identifier (e.g. 'os', 'os.path', 'flask').
  It is NEVER a file path. If the task says 'X from Y', put Y in module and
  X in names. If the task names exactly ONE thing to import, that goes in
  module (plain form) — do NOT also invent a second import.

RenameSymbol — {"verb":"RenameSymbol","symbol":{"file":"src/x.py","name":"OldName"},"new_name":"NewName","confidence":0.7}
AddField — {"verb":"AddField","cls":{"file":"src/x.py","name":"ClassName"},"name":"field_name","type":{"expr":"int"},"default":{"code":"0"},"confidence":0.7}
AddParameter — {"verb":"AddParameter","function":{"file":"src/x.py","name":"ClassName.method"},"name":"param","type":{"expr":"bool"},"default":{"code":"False"},"keyword_only":true,"confidence":0.7}
WrapInTry — {"verb":"WrapInTry","span":{"file":"src/x.py","start_line":10,"end_line":14},"exception_type":{"expr":"ValueError"},"handler_body":[],"confidence":0.7}
AddTest — {"verb":"AddTest","target":{"file":"src/x.py","name":"func"},"test_name":"test_func","given":{"code":"x = 1"},"when":{"code":"y = func(x)"},"then":{"code":"assert y == 2"},"confidence":0.7}
AddStatement — insert a module-level statement (one or more lines of Python).
  {"verb":"AddStatement","file":{"path":"src/app.py"},"code":"cors = CORS(app)","position":"end","confidence":0.8}
  Use "position":"top_after_imports" to land it just after imports.
AddFunction — insert a complete function definition (with optional decorators).
  {"verb":"AddFunction","file":{"path":"src/app.py"},
   "source":"@app.route(\"/health\")\\ndef health() -> dict:\\n    return {\"ok\": True}\\n",
   "position":"end","confidence":0.8}
RecallMore — {"verb":"RecallMore","query":"rate-limit middleware","confidence":0.5}
RevealBody — {"verb":"RevealBody","symbol":{"file":"src/x.py","name":"func"},"confidence":0.5}
MarkBlocked — {"verb":"MarkBlocked","reason_code":"missing_context","detail":"need to see User model","confidence":0.6}
MarkDone — {"verb":"MarkDone","summary":"added stripe import","confidence":0.9}
Branch — {"verb":"Branch","rationale":"try alternative approach","confidence":0.5}
Research — fetch a documentation URL into the brain at runtime.
  {"verb":"Research","url":"https://flask-cors.readthedocs.io/en/latest/","reason":"need exact Flask-CORS init pattern","confidence":0.7}
  Use when the task mentions a library/API the brain doesn't already know.
  Next cycle's HINTS will include the fetched content.

Rules:
- Emit ONE JSON object on a single line.
- No markdown code fences. No commentary. JSON only.
- File paths must be repository-relative (no leading slash, no '..').
- Use the EXACT file paths from WORKSPACE FILES — do not invent paths.
- Use confidence in [0.0, 1.0].
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


_STRING_TO_REF_FIELDS = {
    "file": "path",
    "type": "expr",
    "default": "code",
    "given": "code",
    "when": "code",
    "then": "code",
    "exception_type": "expr",
    "intent": "label",
}


def _normalize_path(p: str) -> str:
    """Strip leading slashes and leading './'. Repository-relative."""
    while p.startswith("/"):
        p = p[1:]
    while p.startswith("./"):
        p = p[2:]
    return p


def _coerce_loose(payload: dict[str, Any]) -> dict[str, Any]:
    """Forgive common small-model JSON-shape mistakes.

    The strict Pydantic schema is preserved — this only normalizes
    obvious shorthand before validation runs. Mutates and returns a
    shallow copy of *payload*.
    """
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)

    for field, inner_key in _STRING_TO_REF_FIELDS.items():
        if isinstance(out.get(field), str):
            value = out[field]
            if inner_key == "path":
                value = _normalize_path(value)
            out[field] = {inner_key: value}

    for sym_field in ("symbol", "cls", "function", "target"):
        v = out.get(sym_field)
        if isinstance(v, str):
            # Accept "file::name" or "file:name" or "name" with a separate "file" field nearby.
            for sep in ("::", ":"):
                if sep in v:
                    f, _, n = v.partition(sep)
                    out[sym_field] = {"file": _normalize_path(f), "name": n}
                    break
        elif isinstance(v, dict) and isinstance(v.get("file"), str):
            v = dict(v)
            v["file"] = _normalize_path(v["file"])
            out[sym_field] = v

    span = out.get("span")
    if isinstance(span, dict) and isinstance(span.get("file"), str):
        span = dict(span)
        span["file"] = _normalize_path(span["file"])
        out["span"] = span

    if isinstance(out.get("file"), dict) and isinstance(out["file"].get("path"), str):
        f = dict(out["file"])
        f["path"] = _normalize_path(f["path"])
        out["file"] = f

    return out


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

    def _generate(
        self,
        user_msg: str,
        *,
        sticky_correction: str | None = None,
        temperature: float | None = None,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the action emitter for the LATTICE coding harness. "
                    "On each turn you receive: a TASK, RELEVANT SYMBOLS, and HINTS "
                    "(which include the history of edits already applied this task). "
                    "Emit exactly ONE JSON action from the vocabulary below.\n\n"
                    "RULES:\n"
                    "- Look at HINTS. If FILES ALREADY EDITED contains your target "
                    "  and the task says it's done, emit MarkDone.\n"
                    "- Do NOT repeat an edit listed under 'added' in a previous "
                    "  step — it is already applied.\n"
                    "- Make progress on each turn: pick an edit that is NOT YET in "
                    "  the history.\n"
                    "- 'module' in AddImport is a Python package (e.g. 'stripe'), "
                    "  NEVER a file path.\n"
                    "- Output ONLY one JSON object. No markdown fences, no prose.\n\n"
                    + _VOCAB_DOC
                ),
            },
            {"role": "user", "content": user_msg},
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
                do_sample=temp > 0,
                temperature=temp if temp > 0 else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        self._load()

        prompt = _render_observation(obs) + "\n\nJSON action:"
        if os.environ.get("LATTICE_LLM_DEBUG"):
            import sys

            sys.stderr.write("\n--- prompt ---\n" + prompt + "\n--- end ---\n")

        actions: list[Action] = []
        seen_dumps: set[str] = set()

        # First candidate at the proposer's configured temperature
        # (default 0 = most likely). Subsequent candidates use sampling
        # so we get genuinely different proposals for pre-flight selection.
        candidate_temps: list[float] = [self.temperature]
        for i in range(max(0, n - 1)):
            candidate_temps.append(0.5 + 0.15 * i)  # 0.5, 0.65, 0.8, ...

        for temp in candidate_temps:
            action = self._propose_one(prompt, temperature=temp)
            if action is None:
                continue
            dump = action.model_dump_json()
            if dump in seen_dumps:
                continue
            seen_dumps.add(dump)
            actions.append(action)
            if len(actions) >= n:
                break

        if not actions:
            raise ProposerError("local LLM produced no valid action after retries")
        return actions

    def _propose_one(self, prompt: str, *, temperature: float) -> Action | None:
        correction: str | None = None
        for _ in range(self.retries + 1):
            raw = self._generate(prompt, sticky_correction=correction, temperature=temperature)
            payload = _extract_json(raw)
            if payload is None:
                correction = (
                    "Your previous response did not contain valid JSON. "
                    "Emit ONE JSON object only, on a single line, no fences, no prose."
                )
                continue
            try:
                return parse_action(_coerce_loose(payload))
            except ValidationError as exc:
                correction = (
                    "Your JSON failed schema validation. "
                    f"Errors:\n{exc}\nFix the fields and try again."
                )
                continue
        return None


# Static check that LocalLLMProposer satisfies the Proposer protocol.
_: Proposer = LocalLLMProposer()
