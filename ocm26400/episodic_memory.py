"""Mémoire épisodique neurale — L3, EX-B182 (audit final HAUTE).

La mémoire épisodique stocke les EXPÉRIENCES vécues (qui, quoi, où, quand).
Contrairement à la mémoire sémantique (règles abstraites) ou procédurale (comment faire),
l'épisodique est SPECIFIQUE : "j'ai vu X à T moment dans le contexte Y".

* EpisodicMemory : stocke des épisodes (timestamp, content, context, embedding).
* recall(query) : retrouve l'épisode le + similaire (retrieval cosinus).
* consolidate : épisodique → sémantique (extrait la règle, cf sleep).
* replay : rejoue les épisodes pour consolidation (expérience replay).

Le SpectralCoreBlock peut encoder chaque épisode en AMV (le "contexte" de l'épisode).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class Episode:
    """Un épisode = une expérience vécue mémorisée."""
    id: int
    timestamp: float
    content: str                    # ce qui s'est passé
    context: Dict[str, Any]         # où, qui, conditions
    embedding: Optional[torch.Tensor] = None  # AMV de l'épisode
    outcome: Optional[str] = None   # résultat/action
    reward: Optional[float] = None  # récompense reçue


class EpisodicMemory:
    """Mémoire épisodique : stocke, retrieve, consolide, rejoue."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.episodes: List[Episode] = []
        self._next_id = 0

    def store(self, content: str, context: Dict[str, Any] = None,
              outcome: str = None, reward: float = None,
              embedding: torch.Tensor = None) -> Episode:
        """Stocke un nouvel épisode."""
        ep = Episode(
            id=self._next_id, timestamp=time.time(),
            content=content, context=context or {}, outcome=outcome,
            reward=reward,
            embedding=embedding.detach().flatten()[:self.dim] if embedding is not None
                      else torch.zeros(self.dim),
        )
        self._next_id += 1
        self.episodes.append(ep)
        return ep

    def recall(self, query_embedding: torch.Tensor, top_k: int = 1
               ) -> List[Tuple[Episode, float]]:
        """Retrouve les top_k épisodes les + similaires (cosinus)."""
        if not self.episodes:
            return []
        q = query_embedding.flatten()[:self.dim]
        qn = q.norm() + 1e-8
        scored = []
        for ep in self.episodes:
            if ep.embedding is None:
                continue
            en = ep.embedding.norm() + 1e-8
            sim = float((q @ ep.embedding[:self.dim]) / (qn * en))
            scored.append((ep, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def recall_by_content(self, query: str, top_k: int = 3) -> List[Episode]:
        """Retrouve par matching textuel (fallback sans embedding)."""
        q = query.lower()
        scored = []
        for ep in self.episodes:
            score = sum(1 for w in q.split() if w in ep.content.lower())
            scored.append((ep, score))
        scored.sort(key=lambda x: -x[1])
        return [ep for ep, _ in scored[:top_k]]

    def consolidate_to_rule(self) -> Optional[str]:
        """Consolide les épisodes → règle sémantique (extraction de pattern).
        Si tous les épisodes suivent un pattern, l'extrait comme règle."""
        if len(self.episodes) < 3:
            return None
        # pattern simple : si même outcome → règle
        outcomes = [ep.outcome for ep in self.episodes if ep.outcome]
        if len(outcomes) >= 3 and len(set(outcomes)) == 1:
            return f"règle extraite : {self.episodes[0].content[:30]}... → {outcomes[0]}"
        return None

    def replay(self, n: int = 10) -> List[Episode]:
        """Rejoue n épisodes (expérience replay pour consolidation)."""
        import random
        rng = random.Random(42)
        if len(self.episodes) <= n:
            return self.episodes[:]
        return rng.sample(self.episodes, n)

    def size(self) -> int:
        return len(self.episodes)

    def forget_oldest(self, n: int = 1) -> None:
        """Oublie les n épisodes les + anciens (gestion mémoire)."""
        self.episodes = self.episodes[n:]


if __name__ == "__main__":
    mem = EpisodicMemory(dim=64)
    # stocke des épisodes
    mem.store("chat vu dans le jardin", {"lieu": "jardin"}, outcome="observé")
    mem.store("chien vu dans le parc", {"lieu": "parc"}, outcome="observé")
    mem.store("repas servi à midi", {"lieu": "cuisine", "heure": "12h"}, outcome="mangé")
    mem.store("chat vu dans le jardin encore", {"lieu": "jardin"}, outcome="observé")
    mem.store("chat vu dans le jardin toujours", {"lieu": "jardin"}, outcome="observé")
    print(f"[episodic] {mem.size()} épisodes stockés")
    # recall par contenu
    found = mem.recall_by_content("chat jardin", top_k=2)
    print(f"[episodic] recall 'chat jardin' : {len(found)} résultats")
    for ep in found:
        print(f"  #{ep.id} : {ep.content} (outcome={ep.outcome})")
    # consolidation
    rule = mem.consolidate_to_rule()
    print(f"[episodic] règle consolidée : {rule}")
    # replay
    replayed = mem.replay(3)
    print(f"[episodic] replay : {len(replayed)} épisodes rejoués")
