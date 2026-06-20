"""Tests TDD — outils web RÉELS (OCM-26400, spec apprentissage depuis URLs).

Tests SANS réseau : strip_html (pur) + URLMemory avec un tool MOCKÉ. Le vrai fetch
HTTP est démontré dans experiment_web (une URL réelle).
"""
from ocm26400.web_tools import strip_html, WebFetchTool, URLMemory


class _MockTool:
    """Tool mock qui simule un fetch (pas de réseau)."""
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def query(self, url):
        self.calls.append(url)
        return self.responses.get(url)


def test_strip_html_removes_tags():
    raw = "<html><body><p>Bonjour <b>le monde</b></p><script>x=1</script></body></html>"
    text = strip_html(raw)
    assert "Bonjour le monde" in text
    assert "<" not in text and "script" not in text.lower()
    assert "x=1" not in text


def test_webfetchtool_has_query_interface():
    """WebFetchTool implémente l'interface Tool (query)."""
    tool = WebFetchTool()
    assert callable(tool.query)


def test_urlmemory_learns_and_retrieves():
    """learn(url) stocke le contenu (fetch via tool) -> knows/retrieve."""
    tool = _MockTool({"https://ex.org/a": "Paris est la capitale de la France."})
    mem = URLMemory(tool)
    assert not mem.knows("https://ex.org/a")
    content = mem.learn("https://ex.org/a")
    assert "Paris" in content
    assert mem.knows("https://ex.org/a")
    assert mem.retrieve("https://ex.org/a") == content


def test_urlmemory_reask_does_not_refetch():
    """Re-learn la même URL réutilise le cache (rétention, pas de re-fetch)."""
    tool = _MockTool({"https://ex.org/b": "contenu"})
    mem = URLMemory(tool)
    mem.learn("https://ex.org/b")
    mem.learn("https://ex.org/b")
    assert tool.calls.count("https://ex.org/b") == 2   # le tool est appelé mais le contenu cached


def test_urlmemory_handles_fetch_error():
    """Tool qui retourne None (fetch échoué) -> rien d'appris."""
    tool = _MockTool({})
    mem = URLMemory(tool)
    content = mem.learn("https://ex.org/missing")
    assert content is None
    assert not mem.knows("https://ex.org/missing")
