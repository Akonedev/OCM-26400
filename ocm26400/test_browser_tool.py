"""Tests browser interactif (Playwright) — OCM-26400 — audit H1."""
import pytest
from ocm26400.browser_tool import InteractiveBrowser, demo

PW = InteractiveBrowser.available()
skip_no_pw = pytest.mark.skipif(not PW, reason="Playwright non installé")


def test_available_flag():
    assert isinstance(InteractiveBrowser.available(), bool)


@skip_no_pw
def test_navigate_real():
    with InteractiveBrowser(headless=True, timeout=20000) as b:
        nav = b.navigate("https://example.com")
    assert nav["ok"] is True
    assert nav["status"] == 200
    assert "example" in nav["title"].lower()


@skip_no_pw
def test_ssrf_blocked():
    """Le browser refuse les schemes non-HTTP (SSRF)."""
    with InteractiveBrowser(headless=True) as b:
        with pytest.raises(ValueError):
            b.navigate("file:///etc/passwd")


@skip_no_pw
def test_extract_text_and_links():
    with InteractiveBrowser(headless=True, timeout=20000) as b:
        b.navigate("https://example.com")
        text = b.extract_text()
        links = b.extract_links()
    assert text and len(text) > 0
    assert isinstance(links, list)


@skip_no_pw
def test_context_manager_closes():
    b = InteractiveBrowser(headless=True)
    with b:
        b.navigate("https://example.com")
    # après close, _page est None
    assert b._page is None


@skip_no_pw
def test_fill_validation_rejects_huge_value():
    """SÉCURITÉ : une value énorme est rejetée."""
    with InteractiveBrowser(headless=True) as b:
        b.navigate("https://example.com")
        r = b.fill("input", "x" * 20000)
    assert r.get("error") or r.get("filled") is False


def test_demo_runs_or_reports_missing():
    """demo() tourne (si PW) ou signale l'absence proprement."""
    out = demo()
    assert "available" in out
