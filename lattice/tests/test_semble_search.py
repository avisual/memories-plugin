"""Tests for the SembleCodeSearch wrapper.

Live indexing+search test is gated on LATTICE_LLM_SMOKE=1 because it
downloads the Model2Vec encoder (~30MB) on first run.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from lattice.sense.semble_search import (
    CodeChunk,
    SembleCodeSearch,
    maybe_code_search,
)


def test_maybe_code_search_with_semble_installed(tmp_path: Path):
    try:
        import semble  # noqa: F401
    except ImportError:
        pytest.skip("semble not installed; cannot test the happy path")

    out = maybe_code_search(tmp_path)
    assert isinstance(out, SembleCodeSearch)


def test_construct_requires_existing_dir():
    with pytest.raises(ValueError, match="must be an existing directory"):
        SembleCodeSearch("/nonexistent/path/abcxyz")


def test_search_empty_query_returns_empty(tmp_path: Path):
    # Skip if semble not present; the [] short-circuit happens before
    # _load() so it works in either env.
    try:
        from lattice.sense.semble_search import SembleCodeSearch  # noqa: F401
    except ImportError:
        pytest.skip("semble not installed")
    s = SembleCodeSearch(tmp_path)
    assert s.search("") == []
    assert s.search("   ") == []


# Live indexing test.

_SMOKE = os.environ.get("LATTICE_LLM_SMOKE") == "1"


@pytest.mark.skipif(not _SMOKE, reason="LATTICE_LLM_SMOKE not set")
def test_indexes_workspace_and_returns_chunks(tmp_path: Path):
    """Live: index a tiny workspace, query, expect a relevant chunk."""
    pytest.importorskip("semble", reason="semble not installed")

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "billing.py").write_text(
        textwrap.dedent("""\
            \"\"\"Billing module.\"\"\"

            def charge_customer(amount: int) -> bool:
                \"\"\"Take a credit card payment.\"\"\"
                return True

            def refund(charge_id: str) -> bool:
                return True
        """)
    )

    s = SembleCodeSearch(tmp_path)
    results = s.search("how to take a credit card payment", top_k=3)
    assert results
    assert isinstance(results[0], CodeChunk)
    paths = [r.file for r in results]
    assert any("billing.py" in p for p in paths)
