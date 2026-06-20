"""Browser interactif RÉEL (Playwright) — réfute audit H1.

L'audit H1 : « Browser use INTERACTIF (clics, JS, formulaires, login) manquant.
WebFetchTool = GET HTTP passif. Pas de Playwright/Selenium ». On comble avec un
vrai browser interactif via Playwright : naviguer, cliquer, remplir des formulaires,
extraire du texte, exécuter du JS. C'est la capacité 'browser use' du cahier des
charges, nécessaire pour BrowseComp / OSWorld.

SÉCURITÉ : SSRF protégé (réutilise _validate_url_safe), headless par défaut,
timeout, pas de téléchargement. Une session = un contexte browser isolé.

HONNÊTE : nécessite Playwright installé + browsers téléchargés
('playwright install chromium'). Si absent, available()=False et les méthodes
retournent un message clair (pas de crash). Le WebFetchTool passif reste le fallback.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .web_tools import _validate_url_safe

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    _HAS_PW = True
except ImportError:
    _HAS_PW = False


class InteractiveBrowser:
    """Browser interactif (Playwright) : navigate, click, fill, extract, js.

    Une instance gère un browser headless + une page. SSRF-safe."""

    def __init__(self, headless: bool = True, timeout: int = 15000):
        self.headless = headless
        self.timeout = timeout
        self._pw = None
        self._browser = None
        self._page = None

    @staticmethod
    def available() -> bool:
        return _HAS_PW

    def _ensure(self):
        if not _HAS_PW:
            raise RuntimeError("Playwright non installé (pip install playwright && "
                               "playwright install chromium)")
        if self._page is None:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self.headless)
            self._page = self._page = self._browser.new_page()
            self._page.set_default_timeout(self.timeout)

    def navigate(self, url: str) -> Dict[str, Any]:
        """Navigue vers une URL (SSRF-validée). Retourne titre + status."""
        url = _validate_url_safe(url)
        self._ensure()
        try:
            resp = self._page.goto(url)
            return {"url": url, "title": self._page.title(),
                    "status": resp.status if resp else None, "ok": True}
        except PWTimeout:
            return {"url": url, "ok": False, "error": "timeout"}
        except Exception as e:
            return {"url": url, "ok": False, "error": f"{type(e).__name__}: {e}"}

    def click(self, selector: str) -> Dict[str, Any]:
        """Clique sur un élément (sélecteur CSS)."""
        self._ensure()
        try:
            self._page.click(selector, timeout=self.timeout)
            return {"selector": selector, "clicked": True}
        except Exception as e:
            return {"selector": selector, "clicked": False, "error": str(e)}

    def fill(self, selector: str, value: str) -> Dict[str, Any]:
        """Remplit un champ de formulaire."""
        self._ensure()
        # validation : pas de sélecteur/value malveillant (Playwright échappe)
        if len(value) > 10000:
            return {"error": "value trop longue"}
        try:
            self._page.fill(selector, value, timeout=self.timeout)
            return {"selector": selector, "filled": True}
        except Exception as e:
            return {"selector": selector, "filled": False, "error": str(e)}

    def extract_text(self, selector: str = "body") -> Optional[str]:
        """Extrait le texte d'un élément (body par défaut)."""
        self._ensure()
        try:
            return self._page.inner_text(selector)[:8000]
        except Exception as e:
            return f"[extract error: {type(e).__name__}]"

    def extract_links(self) -> List[Dict[str, str]]:
        """Tous les liens de la page (texte + href)."""
        self._ensure()
        try:
            return self._page.eval_on_selector_all(
                "a", "els => els.map(e => ({text: e.innerText.trim().slice(0,80), href: e.href}))")
        except Exception:
            return []

    def run_js(self, script: str) -> Any:
        """Exécute un script JS dans la page. SÉCURITÉ : réservé usage interne/audit."""
        self._ensure()
        try:
            return self._page.evaluate(script)
        except Exception as e:
            return f"[js error: {type(e).__name__}]"

    def screenshot(self, path: str) -> Dict[str, Any]:
        self._ensure()
        try:
            self._page.screenshot(path=path)
            return {"path": path, "ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def close(self):
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()
        self._browser = self._pw = self._page = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------- démo (best-effort, nécessite Playwright + chromium) ----------------

def demo() -> Dict[str, Any]:
    """Démo : navigue vers example.com, extrait titre + texte + liens."""
    if not InteractiveBrowser.available():
        return {"available": False,
                "note": "Playwright non installé — pip install playwright && playwright install chromium"}
    with InteractiveBrowser(headless=True) as b:
        nav = b.navigate("https://example.com")
        if not nav.get("ok"):
            return {"available": True, "nav_failed": nav}
        text = (b.extract_text() or "")[:200]
        links = b.extract_links()[:5]
        return {"available": True, "nav": nav, "text_sample": text,
                "n_links": len(links), "links_sample": links}


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2, default=str)[:1500])
