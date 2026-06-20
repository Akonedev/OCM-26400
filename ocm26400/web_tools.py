"""Outils web RÉELS — apprentissage depuis URLs / browser use (OCM-26400, cahier des charges).

Le cahier des charges demande : « je dois pouvoir donner au model une mission
auto-apprentissage : ex je lui donne une URL, et il doit trouver la page, lire le
contenu et apprendre ce que je lui demande », « browser use », « RAG », « utiliser
intégralement le web ». Le réseau étant accessible, on implémente de VRAIS outils :

* fetch_url(url)        : GET HTTP réel (urllib) -> texte (HTML débarrassé des balises).
* WebFetchTool(Tool)    : Tool (backend réel) ; query(url) -> contenu texte de la page.
* URLMemory.learn(url)  : fetch la page, stocke son contenu -> désormais 'connu'
                          (rétention). C'est l'apprentissage depuis une URL réelle.

HONNÊTE : c'est un VRAI fetch HTTP (pas un stub). Le contenu est stocké tel quel
(extraction d'un fait précis nécessiterait un module NLP supplémentaire — on stocke
le texte débarrassé de ses balises). Un BrowserTool plein (clics, JS, computer-use
GUI complet) est une intégration runtime externe (selenium/playwright) ; le WebFetchTool
couvre le cas « lire une page web et l'apprendre » du spec.
"""
import re
import html as _html
import urllib.request
from typing import Optional, Dict

from .tools import Tool


def strip_html(raw: str) -> str:
    """Débarrasse le HTML de ses balises -> texte propre (réutilisable pour apprentissage)."""
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)                  # balises
    text = _html.unescape(raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text


import ipaddress

def _validate_url_safe(url: str) -> str:
    """Valide l'URL contre SSRF : scheme HTTP/HTTPS uniquement, bloque IPs privées."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Scheme interdit (SSRF): {parsed.scheme}")
    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError("Hostname manquant")
    # bloquer IPs privées/loopback
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"IP privée bloquée (SSRF): {hostname}")
    except ipaddress.AddressValueError:
        pass  # hostname DNS (pas une IP), OK
    return url


def fetch_url(url: str, timeout: int = 15, max_chars: int = 4000) -> str:
    """GET HTTP réel -> texte propre (HTML strippé). Tronqué à max_chars.
    SSRF protégé: scheme HTTP/HTTPS + IP privées bloquées."""
    url = _validate_url_safe(url)
    req = urllib.request.Request(url, headers={"User-Agent": "OCM-26400/1.0 (learning agent)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    ctype = r.headers.get("Content-Type", "")
    if "html" in ctype.lower():
        raw = strip_html(raw)
    elif raw.lstrip().startswith("{"):   # JSON (ex: API REST Wikipedia)
        import json
        try:
            obj = json.loads(raw)
            raw = obj.get("extract") or obj.get("title") or raw[:max_chars]
        except Exception:
            pass
    return raw[:max_chars]


class WebFetchTool:
    """Tool backend RÉEL : query(url) -> contenu texte de la page web."""

    def __init__(self, timeout: int = 15, max_chars: int = 4000):
        self.timeout = timeout
        self.max_chars = max_chars

    def query(self, url: str) -> Optional[str]:
        try:
            return fetch_url(url, timeout=self.timeout, max_chars=self.max_chars)
        except Exception as e:
            return f"[fetch error: {type(e).__name__}]"


class URLMemory:
    """Mémoire de contenus appris depuis des URLs réelles (rétention)."""

    def __init__(self, tool: Tool = None):
        self.tool = tool or WebFetchTool()
        self.learned: Dict[str, str] = {}

    def learn(self, url: str) -> str:
        """Fetch la page réelle et l'apprend (stocke). Retourne le contenu appris."""
        content = self.tool.query(url)
        if content:
            self.learned[url] = content
        return content

    def knows(self, url: str) -> bool:
        return url in self.learned

    def retrieve(self, url: str) -> Optional[str]:
        return self.learned.get(url)


def parse_pdf(path: str, max_pages: int = 10, max_chars: int = 8000) -> str:
    """Parse un PDF → texte (PyMuPDF/fitz si dispo, sinon message)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        text = ""
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars]
    except ImportError:
        return f"[PDF parser: PyMuPDF (fitz) non installé — pip install pymupdf]"
    except Exception as e:
        return f"[erreur PDF: {type(e).__name__}]"
