"""RAG multi-hop + re-ranking — réfute audit M11.

M11. RAG multi-hop : la réponse nécessite de CHAINER plusieurs retrievals
(query → doc1 → entité liée → doc2 → ... → réponse). Complète le document_learner
(1-hop). Plus re-ranking (ordonne les candidats par pertinence).

* multi_hop_retrieve(learner, query, hops) : chaîne k retrievals en suivant les
  entités mentionnées. Retourne la chaîne de preuves (citations multiples).
* rerank(candidates, query) : re-classement par pertinence (cosinus + marge).

Ex : « Qui a fondé l'entreprise qui a créé l'iPhone ? »
  hop1: « iPhone créé par Apple » → hop2: « Apple fondée par Steve Jobs » → réponse.
"""
from __future__ import annotations
import re
from typing import List, Optional, Tuple

from .document_learner import DocumentLearner


def rerank(learner: DocumentLearner, query: str, candidates: List[str],
           top_k: int = 3) -> List[Tuple[str, float]]:
    """Re-classement des candidats par similarité cosinus à la requête."""
    from .document_learner import text_embedding
    q = text_embedding(query)
    scored = []
    for c in candidates:
        cv = text_embedding(c)
        n1, n2 = q.norm(), cv.norm()
        sim = float((q @ cv) / (n1 * n2)) if n1 > 1e-9 and n2 > 1e-9 else 0.0
        scored.append((c, sim))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def extract_entities(text: str) -> List[str]:
    """Extrait les entités candidates (mots capitalisés ou > 5 lettres) pour le hop suivant."""
    words = re.findall(r"[A-ZÉÈ][a-zéèàç]{4,}|\b[a-z]{6,}\b", text)
    return list(dict.fromkeys(words))[:5]      # unique, top 5


def multi_hop_retrieve(learner: DocumentLearner, query: str, hops: int = 2
                       ) -> dict:
    """RAG multi-hop : chaîne k retrievals. Chaque hop suit une entité du résultat précédent.
    Retourne la chaîne de preuves (chunks + sources) et la réponse fusionnée."""
    chain: List[dict] = []
    current_query = query
    seen_sources = set()
    for h in range(hops):
        res = learner.retrieve(current_query, top_k=1)
        if not res:
            break
        chunk, source, conf = res[0]
        if source in seen_sources:           # évite les boucles
            break
        seen_sources.add(source)
        chain.append({"hop": h + 1, "query": current_query, "chunk": chunk,
                      "source": source, "confidence": conf})
        # hop suivant : une entité du chunk devient la requête
        ents = extract_entities(chunk)
        if not ents:
            break
        current_query = ents[0]
    return {
        "n_hops": len(chain),
        "chain": chain,
        "final_answer": chain[-1]["chunk"] if chain else None,
        "sources": list(seen_sources),
    }


def demo() -> dict:
    """Démo : KB à 2 docs liés par entité. Question nécessite 2 hops.
    Seuil abaissé (hash embedding faible sur peu de chunks — cf audit H17)."""
    dl = DocumentLearner(threshold=0.3, margin=0.02)
    dl.learn_text("L'iPhone a été créé par Apple en 2007. Apple est une entreprise.",
                  source="tech:iphone")
    dl.learn_text("Apple a été fondée par Steve Jobs en 1976. Jobs est le cofondateur.",
                  source="bio:jobs")
    return multi_hop_retrieve(dl, "Qui a créé l'iPhone ?", hops=2)


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2, ensure_ascii=False)[:800])
