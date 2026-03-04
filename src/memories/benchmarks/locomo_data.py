"""Data loader for the LoCoMo benchmark dataset.

Parses locomo10.json into typed dataclasses for use by the evaluation harness.
Download the dataset from:
    https://github.com/snap-research/locomo/blob/main/data/locomo10.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Turn:
    speaker: str
    text: str
    dialog_id: str
    image_caption: str | None = None


@dataclass
class Session:
    session_num: int
    date: datetime
    turns: list[Turn]


@dataclass
class QAPair:
    question: str
    answer: str
    category: int  # 1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop, 5=adversarial
    evidence: list[str] = field(default_factory=list)

    CATEGORY_NAMES = {
        1: "multi_hop",
        2: "temporal",
        3: "open_domain",
        4: "single_hop",
        5: "adversarial",
    }

    @property
    def category_name(self) -> str:
        return self.CATEGORY_NAMES.get(self.category, f"unknown_{self.category}")


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa_pairs: list[QAPair]


def _parse_datetime(s: str) -> datetime:
    """Parse LoCoMo's varied datetime formats."""
    # Formats seen: "1:56 pm on 8 May, 2023", "2023-05-07", etc.
    s = s.strip()

    # Try ISO format first.
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    # "1:56 pm on 8 May, 2023" → strip the time prefix, parse the date.
    m = re.search(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", s)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        return datetime.strptime(f"{day} {month} {year}", "%d %B %Y")

    # Fallback.
    return datetime(2023, 1, 1)


def _parse_conversation(raw: dict) -> Conversation:
    conv = raw["conversation"]
    speaker_a = conv.get("speaker_a", "A")
    speaker_b = conv.get("speaker_b", "B")

    # Extract sessions.
    session_keys = sorted(
        (k for k in conv if re.match(r"session_\d+$", k)),
        key=lambda k: int(k.split("_")[1]),
    )

    sessions: list[Session] = []
    for sk in session_keys:
        num = int(sk.split("_")[1])
        date_key = f"{sk}_date_time"
        date = _parse_datetime(conv.get(date_key, "2023-01-01"))
        turns = [
            Turn(
                speaker=t["speaker"],
                text=t["text"],
                dialog_id=t["dia_id"],
                image_caption=t.get("blip_caption"),
            )
            for t in conv[sk]
        ]
        sessions.append(Session(session_num=num, date=date, turns=turns))

    # Parse QA pairs — cat 5 uses 'adversarial_answer' key.
    qa_pairs: list[QAPair] = []
    for qa in raw.get("qa", []):
        answer = qa.get("answer") or qa.get("adversarial_answer", "")
        qa_pairs.append(
            QAPair(
                question=qa["question"],
                answer=str(answer),
                category=qa["category"],
                evidence=qa.get("evidence", []),
            )
        )

    return Conversation(
        sample_id=raw["sample_id"],
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=sessions,
        qa_pairs=qa_pairs,
    )


def load_dataset(path: str | Path) -> list[Conversation]:
    """Load locomo10.json and return parsed conversations."""
    with open(path) as f:
        raw = json.load(f)
    return [_parse_conversation(sample) for sample in raw]
