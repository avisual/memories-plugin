"""Tests for the web fetcher (SSRF guards, HTML stripping)."""

from __future__ import annotations

import os

import pytest

from lattice.atoms.web import (
    FetchedPage,
    WebFetchBlocked,
    WebFetchError,
    _html_to_text,
    fetch_url,
)


class TestSSRFGuards:
    def test_blocks_localhost(self):
        with pytest.raises(WebFetchBlocked):
            fetch_url("http://localhost:8000/")

    def test_blocks_127(self):
        with pytest.raises(WebFetchBlocked):
            fetch_url("http://127.0.0.1/")

    def test_blocks_private_10(self):
        with pytest.raises(WebFetchBlocked):
            fetch_url("http://10.0.0.5/admin")

    def test_blocks_private_192(self):
        with pytest.raises(WebFetchBlocked):
            fetch_url("http://192.168.1.1/")

    def test_blocks_link_local(self):
        with pytest.raises(WebFetchBlocked):
            fetch_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_file_scheme(self):
        with pytest.raises(WebFetchBlocked):
            fetch_url("file:///etc/passwd")

    def test_blocks_javascript_scheme(self):
        with pytest.raises(WebFetchBlocked):
            fetch_url("javascript:alert(1)")


class TestHtmlToText:
    def test_strips_scripts_and_styles(self):
        html = "<html><head><title>Hello</title><style>x{}</style></head><body><script>bad()</script><p>Hi <b>there</b></p></body></html>"
        title, text = _html_to_text(html)
        assert title == "Hello"
        assert "bad()" not in text
        assert "x{}" not in text
        assert "Hi" in text
        assert "there" in text

    def test_handles_no_title(self):
        title, text = _html_to_text("<p>just a paragraph</p>")
        assert title == ""
        assert "just a paragraph" in text

    def test_caps_length(self):
        html = "<p>" + ("x" * 100_000) + "</p>"
        _, text = _html_to_text(html, max_chars=500)
        assert len(text) <= 500


_LIVE = os.environ.get("LATTICE_LIVE_WEB") == "1"


@pytest.mark.skipif(not _LIVE, reason="LATTICE_LIVE_WEB not set; skipping live fetch")
def test_live_fetch_example_dot_com():
    """example.com is a stable public page; cheap canary."""
    page = fetch_url("https://example.com/")
    assert page.status == 200
    assert "Example Domain" in page.text
    assert "Example Domain" in page.title or page.title == "Example Domain"
