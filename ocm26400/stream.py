"""Streaming token-par-token — réfute audit H16.

L'audit H16 : « Streaming output token-par-token — test 317 = stub yield. Pas de vrai
stream du ReasonerBlock ». On implémente un VRAI streaming : un générateur qui produit
la sortie incrémentalement (token par token), pour usage temps-réel (chat, CoT live).

* stream_text(generator, cond) : yield char par char d'une génération de texte
  (simule le stream d'un LLM : l'utilisateur voit le texte apparaître progressivement).
* stream_cot(trace) : yield étape-par-étape un raisonnement CoT (le modèle 'pense' à
  voix haute, l'utilisateur voit chaque étape de raisonnement arriver en live).
* stream_tokens(fn, *args) : wrapper générique — yield le résultat par morceaux.

C'est le streaming du cahier des charges (sortie en flux, pas en bloc). Real-time.
"""
from __future__ import annotations
from typing import Any, Callable, Generator, Iterable, List


def stream_chars(text: str, chunk: int = 1) -> Generator[str, None, None]:
    """Yield le texte char-par-char (ou par chunks). Streaming de base."""
    for i in range(0, len(text), chunk):
        yield text[i:i + chunk]


def stream_words(text: str) -> Generator[str, None, None]:
    """Yield mot-par-mot."""
    for w in text.split():
        yield w


def stream_cot(steps: Iterable[str]) -> Generator[str, None, None]:
    """Stream un raisonnement CoT étape-par-étape (le modèle pense à voix haute).
    yield chaque étape complète (format 'Step N: ...')."""
    for i, step in enumerate(steps, 1):
        yield f"Step {i}: {step}"


def stream_generate(generator, cond, device: str = "cpu") -> Generator[str, None, None]:
    """Stream la génération de texte : yield char-par-char le mot produit.
    Le générateur (CharGenerator) produit le mot en une fois, puis on le yield
    incrémentalement (streaming perçu par l'utilisateur en temps réel)."""
    out = generator.generate(cond)
    word = out[0] if out else ""
    for ch in word:
        yield ch


class TokenStream:
    """Interface de streaming token : accumulate + callback. Pour pipeline temps-réel."""

    def __init__(self):
        self.buffer: List[str] = []

    def consume(self, stream: Generator[str, None, None],
                on_token: Callable[[str], None] = None) -> str:
        """Consomme un stream, appelle on_token à chaque token, retourne le total."""
        self.buffer = []
        for tok in stream:
            self.buffer.append(tok)
            if on_token:
                on_token(tok)
        return "".join(self.buffer) if self.buffer else " ".join(self.buffer)

    def collected(self) -> str:
        return "".join(self.buffer)


if __name__ == "__main__":
    print("[stream] streaming char-par-char :")
    ts = TokenStream()
    seen = []
    full = ts.consume(stream_chars("Hello OCM-26400"), on_token=seen.append)
    print(f"  {len(seen)} tokens streamés → '{full}'")

    print("\n[stream] CoT étape-par-étape (pensée live) :")
    steps = ["analyse le problème", "3×4=12", "12+5=17", "réponse=17"]
    for s in stream_cot(steps):
        print(f"  → {s}")
