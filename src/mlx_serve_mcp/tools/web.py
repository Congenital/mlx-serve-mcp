"""Web tools: search the web and read web pages.

``web_search`` queries DuckDuckGo's HTML endpoint (no API key) and returns the
top results. ``browse`` is a fetch-based page reader: it can navigate to a URL
and extract visible text or a CSS-selected subtree. Actions that require a real
browser (``click``, ``executeJS``, ``screenshot``) are accepted but report that
a browser backend is not bundled, so clients get a clear signal instead of a hang.

Only the selectolax calls we're confident about are used (``.css`` / ``.text`` /
``.decompose`` / ``.attributes`` / ``.html``), so the parser API surface stays
small and stable across selectolax versions.
"""

from __future__ import annotations

import httpx
from selectolax.parser import HTMLParser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from . import Deps

USER_AGENT = "mlx-serve-mcp/0.2 (+mcp)"
MAX_CHARS = 20000
_NOISE = "script, style, noscript, template, iframe, svg, canvas"


def _clean(text: str | None) -> str:
    """Collapse whitespace in extracted text."""
    return " ".join((text or "").split())


def _fetch_text(html: str) -> str:
    """Extract visible text from an HTML document."""
    tree = HTMLParser(html)
    for tag in tree.css(_NOISE):
        tag.decompose()
    candidates = tree.css("body") or tree.css("html")
    if not candidates:
        return ""
    return _clean(candidates[0].text())


def register(mcp: "FastMCP", deps: Deps) -> None:
    # Per-session "current page" for the browse tool.
    page: dict[str, Any] = {"url": None, "html": None}

    async def _get(url: str) -> tuple[int, str]:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30.0
        ) as client:
            resp = await client.get(url)
            return resp.status_code, resp.text

    @mcp.tool()
    async def web_search(query: str, max_results: int = 8) -> str:
        """Search the web using DuckDuckGo and return the top results.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
        """
        url = "https://html.duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30.0
            ) as client:
                resp = await client.post(url, data={"q": query})
                if resp.status_code >= 400:
                    return f"error: search failed (HTTP {resp.status_code})"
                tree = HTMLParser(resp.text)
        except httpx.HTTPError as exc:
            return f"error: search failed: {exc}"

        results = []
        for result in tree.css(".result")[:max_results]:
            titles = result.css(".result__title")
            if not titles:
                continue
            title_node = titles[0]
            title = _clean(title_node.text())
            href = title_node.attributes.get("href", "")
            snippets = result.css(".result__snippet")
            snippet = _clean(snippets[0].text()) if snippets else ""
            results.append(f"{title}\n  {href}\n  {snippet}")
        if not results:
            return f"no results for {query!r}"
        return "\n\n".join(results)

    @mcp.tool()
    async def browse(
        action: str,
        url: str | None = None,
        selector: str | None = None,
        script: str | None = None,
    ) -> str:
        """Browse a web page.

        Actions:
        * ``navigate``   — load a URL (requires ``url``).
        * ``readText``   — visible text of the current page (or ``url``).
        * ``extractText``— innerText of ``selector`` on the page.
        * ``readHTML``   — raw HTML of the page (or ``selector``).
        * ``click`` / ``executeJS`` / ``screenshot`` — require a real browser
          backend, which is not bundled; these return a clear notice.

        Args:
            action: One of navigate, readText, extractText, readHTML, click, executeJS, screenshot.
            url: URL to load (for navigate, or to read a specific page).
            selector: CSS selector (for extractText / readHTML / click).
            script: JavaScript to run (for executeJS; requires a browser backend).
        """
        if action in ("click", "executeJS", "screenshot"):
            return (
                f"error: browse action {action!r} requires a real browser backend "
                "(JavaScript execution / rendering), which this server does not bundle. "
                "Use navigate / readText / extractText / readHTML for fetch-based reading."
            )

        if action == "navigate":
            if not url:
                return "error: navigate requires a url"
            code, html = await _get(url)
            page["url"], page["html"] = url, html
            return f"navigated to {url} (HTTP {code}, {len(html)} chars)"

        # Resolve which page to read from.
        if url:
            code, html = await _get(url)
            page["url"], page["html"] = url, html
        elif page["html"] is None:
            return "error: no page loaded; call browse(navigate, url=...) first"
        else:
            html = page["html"]

        tree = HTMLParser(html)
        if action == "readText":
            return _fetch_text(html)[:MAX_CHARS]
        if action == "extractText":
            if not selector:
                return "error: extractText requires a selector"
            texts = [_clean(n.text()) for n in tree.css(selector) if n.text()]
            return "\n".join(texts)[:MAX_CHARS] or "(empty)"
        if action == "readHTML":
            if selector:
                return "\n".join(n.html for n in tree.css(selector))[:MAX_CHARS]
            return html[:MAX_CHARS]
        return f"error: unknown browse action: {action!r}"