"""Tests for LoCoMo benchmark harness — data loading, scoring, retrieval metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from memories.benchmarks.locomo_data import (
    Conversation,
    QAPair,
    Session,
    Turn,
    _parse_datetime,
    load_dataset,
)
from memories.benchmarks.locomo_eval import (
    _compute_answer_token_recall,
    _compute_evidence_recall,
    _compute_reciprocal_rank,
)
from memories.benchmarks.locomo_scoring import (
    normalize_answer,
    score_qa,
    token_f1,
)


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------


class TestParseDatetime:
    def test_iso_format(self):
        assert _parse_datetime("2023-05-07") == datetime(2023, 5, 7)

    def test_locomo_format(self):
        dt = _parse_datetime("1:56 pm on 8 May, 2023")
        assert dt.day == 8
        assert dt.month == 5
        assert dt.year == 2023

    def test_locomo_format_no_comma(self):
        dt = _parse_datetime("2:30 pm on 15 June 2023")
        assert dt.day == 15
        assert dt.month == 6

    def test_fallback(self):
        dt = _parse_datetime("garbage")
        assert dt == datetime(2023, 1, 1)


class TestLoadDataset:
    @pytest.fixture
    def sample_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal locomo10.json for testing."""
        data = [
            {
                "sample_id": "conv-test",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1": [
                        {
                            "speaker": "Alice",
                            "dia_id": "D1:1",
                            "text": "Hey Bob! How are you?",
                            "img_url": "",
                            "blip_caption": "",
                        },
                        {
                            "speaker": "Bob",
                            "dia_id": "D1:2",
                            "text": "Good! I just started learning guitar.",
                            "img_url": "",
                            "blip_caption": "",
                        },
                    ],
                    "session_1_date_time": "2023-05-01",
                    "session_2": [
                        {
                            "speaker": "Alice",
                            "dia_id": "D2:1",
                            "text": "How's the guitar going?",
                            "img_url": "",
                            "blip_caption": "",
                        },
                    ],
                    "session_2_date_time": "2023-05-15",
                },
                "qa": [
                    {
                        "question": "What instrument did Bob start learning?",
                        "answer": "guitar",
                        "category": 4,
                        "evidence": ["D1:2"],
                    },
                    {
                        "question": "When did Alice and Bob first talk?",
                        "answer": "1 May 2023",
                        "category": 2,
                        "evidence": ["D1:1"],
                    },
                    {
                        "question": "What did Bob research?",
                        "category": 5,
                        "adversarial_answer": "music theory",
                        "evidence": [],
                    },
                    {
                        "question": "What instrument did Bob learn and when?",
                        "answer": "guitar, May 2023",
                        "category": 1,
                        "evidence": ["D1:2"],
                    },
                ],
            }
        ]
        path = tmp_path / "locomo_test.json"
        path.write_text(json.dumps(data))
        return path

    def test_loads_conversations(self, sample_dataset: Path):
        convs = load_dataset(sample_dataset)
        assert len(convs) == 1

    def test_conversation_metadata(self, sample_dataset: Path):
        conv = load_dataset(sample_dataset)[0]
        assert conv.sample_id == "conv-test"
        assert conv.speaker_a == "Alice"
        assert conv.speaker_b == "Bob"

    def test_sessions_parsed(self, sample_dataset: Path):
        conv = load_dataset(sample_dataset)[0]
        assert len(conv.sessions) == 2
        assert conv.sessions[0].session_num == 1
        assert conv.sessions[1].session_num == 2
        assert len(conv.sessions[0].turns) == 2
        assert len(conv.sessions[1].turns) == 1

    def test_turns_parsed(self, sample_dataset: Path):
        conv = load_dataset(sample_dataset)[0]
        turn = conv.sessions[0].turns[1]
        assert turn.speaker == "Bob"
        assert "guitar" in turn.text
        assert turn.dialog_id == "D1:2"

    def test_qa_pairs_parsed(self, sample_dataset: Path):
        conv = load_dataset(sample_dataset)[0]
        assert len(conv.qa_pairs) == 4

    def test_adversarial_qa_uses_adversarial_answer(self, sample_dataset: Path):
        conv = load_dataset(sample_dataset)[0]
        cat5 = [qa for qa in conv.qa_pairs if qa.category == 5]
        assert len(cat5) == 1
        assert cat5[0].answer == "music theory"

    def test_category_names(self, sample_dataset: Path):
        conv = load_dataset(sample_dataset)[0]
        names = {qa.category_name for qa in conv.qa_pairs}
        assert names == {"single_hop", "temporal", "adversarial", "multi_hop"}

    def test_session_dates(self, sample_dataset: Path):
        conv = load_dataset(sample_dataset)[0]
        assert conv.sessions[0].date == datetime(2023, 5, 1)
        assert conv.sessions[1].date == datetime(2023, 5, 15)


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Guitar") == "guitar"

    def test_removes_articles(self):
        assert normalize_answer("the guitar") == "guitar"

    def test_removes_punctuation(self):
        assert normalize_answer("guitar!") == "guitar"

    def test_removes_commas(self):
        assert normalize_answer("guitar, piano") == "guitar piano"

    def test_whitespace(self):
        assert normalize_answer("  guitar   lessons  ") == "guitar lessons"


class TestTokenF1:
    def test_exact_match(self):
        assert token_f1("guitar", "guitar") == 1.0

    def test_no_overlap(self):
        assert token_f1("piano", "guitar") == 0.0

    def test_partial_overlap(self):
        f1 = token_f1("Bob plays guitar", "guitar")
        assert 0.3 < f1 < 0.8  # Some but not full overlap

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("", "guitar") == 0.0
        assert token_f1("guitar", "") == 0.0

    def test_case_insensitive(self):
        assert token_f1("Guitar", "guitar") == 1.0


class TestScoreQA:
    def test_single_hop(self):
        score = score_qa("guitar", "guitar", category=4)
        assert score == 1.0

    def test_temporal(self):
        score = score_qa("May 2023", "May 2023", category=2)
        assert score == 1.0

    def test_multi_hop_comma_separated(self):
        # Multi-hop averages F1 of full prediction vs each ground truth part.
        score = score_qa("guitar, May 2023", "guitar, May 2023", category=1)
        assert score > 0.5  # Partial match per part is expected
        # Perfect per-part scores.
        score2 = score_qa("guitar", "guitar, piano", category=1)
        assert score2 > 0.4  # Matches "guitar" perfectly, "piano" is 0

    def test_adversarial_unanswerable(self):
        score = score_qa("no information available", "music theory", category=5)
        assert score == 1.0

    def test_adversarial_wrong_answer(self):
        score = score_qa("cooking classes", "music theory", category=5)
        assert score == 0.0

    def test_adversarial_not_mentioned(self):
        score = score_qa("This is not mentioned in the conversation", "music theory", category=5)
        assert score == 1.0

    def test_adversarial_cannot_be_determined(self):
        score = score_qa("cannot be determined from context", "music theory", category=5)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Full dataset loading (requires downloaded locomo10.json)
# ---------------------------------------------------------------------------


class TestRealDataset:
    @pytest.fixture
    def locomo_path(self) -> Path | None:
        p = Path("data/locomo10.json")
        if not p.exists():
            pytest.skip("locomo10.json not downloaded — run: curl -sL -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json")
        return p

    def test_loads_10_conversations(self, locomo_path: Path):
        convs = load_dataset(locomo_path)
        assert len(convs) == 10

    def test_all_have_qa(self, locomo_path: Path):
        convs = load_dataset(locomo_path)
        for conv in convs:
            assert len(conv.qa_pairs) > 0

    def test_total_qa_count(self, locomo_path: Path):
        convs = load_dataset(locomo_path)
        total = sum(len(c.qa_pairs) for c in convs)
        assert total > 1900  # Should be ~1986

    def test_all_categories_present(self, locomo_path: Path):
        convs = load_dataset(locomo_path)
        all_cats = set()
        for conv in convs:
            for qa in conv.qa_pairs:
                all_cats.add(qa.category)
        assert all_cats == {1, 2, 3, 4, 5}

    def test_sessions_have_dates(self, locomo_path: Path):
        convs = load_dataset(locomo_path)
        for conv in convs:
            for session in conv.sessions:
                assert session.date.year >= 2022


# ---------------------------------------------------------------------------
# Retrieval metric tests
# ---------------------------------------------------------------------------


class TestEvidenceRecall:
    def test_all_evidence_found(self):
        atoms = ["[D1:3] Caroline went to group", "[D2:5] Melanie painted"]
        assert _compute_evidence_recall(atoms, ["D1:3", "D2:5"]) == 1.0

    def test_partial_evidence(self):
        atoms = ["[D1:3] Caroline went to group"]
        assert _compute_evidence_recall(atoms, ["D1:3", "D2:5"]) == 0.5

    def test_no_evidence_found(self):
        atoms = ["[D3:1] Something else"]
        assert _compute_evidence_recall(atoms, ["D1:3"]) == 0.0

    def test_empty_evidence(self):
        assert _compute_evidence_recall([], []) == 1.0

    def test_empty_atoms(self):
        assert _compute_evidence_recall([], ["D1:3"]) == 0.0


class TestAnswerTokenRecall:
    def test_exact_match(self):
        atoms = ["guitar lessons are fun"]
        assert _compute_answer_token_recall(atoms, "guitar") == 1.0

    def test_partial_match(self):
        atoms = ["Bob plays guitar"]
        score = _compute_answer_token_recall(atoms, "guitar piano")
        assert score == 0.5

    def test_no_match(self):
        atoms = ["Bob went home"]
        assert _compute_answer_token_recall(atoms, "guitar") == 0.0

    def test_empty_answer(self):
        assert _compute_answer_token_recall(["anything"], "") == 1.0


class TestReciprocalRank:
    def test_first_position(self):
        atoms = ["[D1:3] evidence here", "[D2:1] other"]
        assert _compute_reciprocal_rank(atoms, ["D1:3"]) == 1.0

    def test_second_position(self):
        atoms = ["[D2:1] other", "[D1:3] evidence here"]
        assert _compute_reciprocal_rank(atoms, ["D1:3"]) == 0.5

    def test_not_found(self):
        atoms = ["[D2:1] other", "[D3:1] nope"]
        assert _compute_reciprocal_rank(atoms, ["D1:3"]) == 0.0

    def test_empty_evidence(self):
        assert _compute_reciprocal_rank(["anything"], []) == 1.0
