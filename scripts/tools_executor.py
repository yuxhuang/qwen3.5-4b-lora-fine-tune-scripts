#!/usr/bin/env python3
"""
Agent tools executor: fetch_url, save_page, extract_dom_content, validate_content.
Used by run_agent_eval.py. Optional url_to_html map for mocking fetch_url (no network).
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

# Optional: use requests for real fetch, BeautifulSoup for DOM
try:
    import requests
except ImportError:
    requests = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def fetch_url(url: str, url_to_html: Optional[dict[str, str]] = None, timeout: int = 10) -> str:
    """Fetch HTML from URL. If url_to_html is provided, return that (mock, no network)."""
    if url_to_html is not None and url in url_to_html:
        return url_to_html[url]
    if requests is None:
        raise RuntimeError("Install requests to use fetch_url without url_to_html mock.")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def save_page(content: str, path: str, base_dir: Optional[Path] = None) -> str:
    """Save HTML content to local path. If base_dir is set, path is relative to it."""
    if base_dir is not None:
        path = str(base_dir / path.lstrip("/"))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Saved successfully to {path}"


def extract_dom_content(path: str, selector: str, base_dir: Optional[Path] = None) -> str:
    """Extract text at DOM selector from HTML file. Uses BeautifulSoup (CSS selector)."""
    if BeautifulSoup is None:
        raise RuntimeError("Install beautifulsoup4 to use extract_dom_content.")
    if base_dir is not None:
        path = str(base_dir / path.lstrip("/"))
    html = Path(path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    el = soup.select_one(selector)
    if el is None:
        return ""
    return el.get_text(strip=True)


def validate_content(extracted_content: str, expected_text: str) -> str:
    """Check if expected_text appears in extracted_content (substring)."""
    if expected_text in extracted_content:
        return "Match."
    return "No match."


def parse_action_and_input(assistant_text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse ReAct-style output for 'Action:' and 'Action Input:' (JSON). Returns list of (action_name, args)."""
    actions = []
    pattern = re.compile(
        r"Action:\s*(\w+)\s*\n\s*Action Input:\s*(\{[^}]+\}|[^\n]+)",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in pattern.finditer(assistant_text):
        name = m.group(1).strip()
        raw_input = m.group(2).strip()
        try:
            args = json.loads(raw_input)
        except json.JSONDecodeError:
            args = {}
        actions.append((name, args))
    return actions


def run_tool(name: str, args: dict[str, Any], url_to_html: Optional[dict] = None, base_dir: Optional[Path] = None) -> str:
    """Dispatch to the right tool and return observation string."""
    if name == "fetch_url":
        return fetch_url(args.get("url", ""), url_to_html=url_to_html)
    if name == "save_page":
        return save_page(args.get("content", ""), args.get("path", ""), base_dir=base_dir)
    if name == "extract_dom_content":
        return extract_dom_content(args.get("path", ""), args.get("selector", ""), base_dir=base_dir)
    if name == "validate_content":
        return validate_content(args.get("extracted_content", ""), args.get("expected_text", ""))
    return f"Unknown tool: {name}"
