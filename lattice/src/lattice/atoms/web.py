"""Web fetcher — the brain learns from the live web at run time.

Uses curl-cffi (browser-impersonating HTTP) so requests look like a
real Chrome session — past Cloudflare, past basic anti-bot. Strips
HTML to a readable text approximation suitable for storing as an atom.

SSRF guards:
- File / data / javascript schemes blocked.
- Localhost, link-local, loopback, private RFC1918 networks blocked.
- Per-fetch timeout cap.
- Response size cap (defaults to 256KB; bigger is almost never useful
  as atom content and is a memory/DOS risk).

The result is intentionally lossy — the atom should be small and
recallable, not a full archive. For deeper context, the LLM can
emit another Research action with a more specific URL.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse


_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_TIMEOUT_S = 8.0
_DEFAULT_MAX_BYTES = 256 * 1024
_DEFAULT_USER_AGENT = "chrome120"


class WebFetchError(Exception):
    """Raised when a fetch is blocked or fails."""


class WebFetchBlocked(WebFetchError):
    """Raised when SSRF guards reject the URL."""


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str
    status: int
    text: str  # cleaned, readable extract
    title: str = ""


# ---------------------------------------------------------------------------
# SSRF guards
# ---------------------------------------------------------------------------


def _is_private_addr(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True  # if we can't resolve, treat as unsafe
    for info in infos:
        sockaddr = info[4]
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False


def _guard(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise WebFetchBlocked(
            f"scheme {parsed.scheme!r} not allowed (only http/https)"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise WebFetchBlocked(f"no host in URL: {url!r}")
    if host in {"localhost", "ip6-localhost", "ip6-loopback"}:
        raise WebFetchBlocked(f"localhost not allowed: {url!r}")
    if _is_private_addr(host):
        raise WebFetchBlocked(f"private / loopback address not allowed: {url!r}")


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------


_SCRIPT_RE = re.compile(r"<script\b.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style\b.*?</style>", re.IGNORECASE | re.DOTALL)
_HEAD_RE = re.compile(r"<head\b.*?</head>", re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _html_to_text(html: str, *, max_chars: int = 20_000) -> tuple[str, str]:
    """Return (title, text) — both stripped and whitespace-collapsed."""
    title_match = _TITLE_RE.search(html)
    title = ""
    if title_match:
        title = _WS_RE.sub(" ", title_match.group(1)).strip()[:200]

    h = _SCRIPT_RE.sub(" ", html)
    h = _STYLE_RE.sub(" ", h)
    h = _HEAD_RE.sub(" ", h)
    h = _TAG_RE.sub(" ", h)
    h = _WS_RE.sub(" ", h).strip()
    return title, h[:max_chars]


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


def fetch_url(
    url: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    impersonate: str = _DEFAULT_USER_AGENT,
    extra_headers: dict[str, str] | None = None,
) -> FetchedPage:
    """Fetch *url* with browser impersonation, return a cleaned FetchedPage.

    SSRF-guarded. Raises WebFetchBlocked for unsafe URLs and
    WebFetchError for transport failures.
    """
    _guard(url)

    try:
        from curl_cffi import requests as cf_requests
    except ImportError as exc:
        raise WebFetchError(
            "curl-cffi not installed (it's in the base deps; reinstall lattice)"
        ) from exc

    try:
        response = cf_requests.get(
            url,
            impersonate=impersonate,
            timeout=timeout_s,
            headers={"Accept": "text/html,application/xhtml+xml,*/*;q=0.8", **(extra_headers or {})},
            allow_redirects=True,
        )
    except Exception as exc:  # curl_cffi raises a family of typed errors
        raise WebFetchError(f"fetch failed for {url!r}: {exc}") from exc

    final_url = response.url or url
    # Re-guard the final URL in case of redirects to private addresses.
    try:
        _guard(final_url)
    except WebFetchBlocked as exc:
        raise WebFetchBlocked(f"redirect target rejected: {exc}") from exc

    body = response.content
    if not isinstance(body, (bytes, bytearray)):
        body = body.encode("utf-8", errors="replace")
    if len(body) > max_bytes:
        body = body[:max_bytes]

    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type or body.lstrip().lower().startswith(b"<!doctype") or b"<html" in body[:2000].lower():
        title, text = _html_to_text(body.decode("utf-8", errors="replace"))
    else:
        title = ""
        text = body.decode("utf-8", errors="replace")[:20_000]

    return FetchedPage(
        url=url,
        final_url=final_url,
        status=int(response.status_code),
        text=text,
        title=title,
    )


def fetch_many(urls: Iterable[str], **kwargs) -> list[FetchedPage]:
    """Fetch sequentially (no concurrency yet). Skips fetches that fail or are blocked."""
    out: list[FetchedPage] = []
    for url in urls:
        try:
            out.append(fetch_url(url, **kwargs))
        except WebFetchError:
            continue
    return out
