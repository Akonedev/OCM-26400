"""Apprentissage depuis documents — cycle PDF/URL → chunk → KB → retrieval + citations.

Réfute les gaps audit C7/C8/H10 :
* C7  : parse_pdf existait mais n'écrivait PAS dans la KB → aucun apprentissage.
* C8  : URLMemory stockait le texte brut, pas de retrieval sémantique exploitable.
* H10 : KnowledgeBase retrieval cosinus sur mots-id, pas de chunking texte + citations.

Ici on câble les briques existantes (parse_pdf, fetch_url, chunk_document,
KnowledgeBase, LearnedVocab) en un VRAI cycle d'apprentissage :
    document (PDF/URL/texte) → chunks → embedding (hash stable) → KB.store
    → retrieve(query) retourne le chunk + sa source (citation) + confiance.

L'embedding texte = hash de n-grammes stable (pas un BERT — honnête, cf audit H17
qui demande des embeddings sémantiques réels ; ici on fournit le PIPELINE complet,
l'embedding est remplaçable par un vrai encodeur sans changer l'API).

Abstention conservée : si aucun chunk n'atteint le seuil → "je ne sais pas" (épistémique).
"""
from __future__ import annotations
import hashlib
from typing import List, Optional, Tuple

import torch

from .knowledge_base import chunk_document
from .learned_vocab import LearnedVocab


def text_embedding(text: str, dim: int = 64) -> torch.Tensor:
    """Embedding texte stable par hash de n-grammes de caractères (dim-d).
    Honnête : hash déterministe (pas sémantique) — remplaçable par un vrai encodeur.
    Suffisant pour retrieval exact/near-exact sur chunks appris."""
    v = torch.zeros(dim)
    s = text.lower().strip()
    if not s:
        return v
    # n-grammes de caractères (3,4,5) + mots
    for n in (3, 4, 5):
        for i in range(max(0, len(s) - n + 1)):
            gram = s[i:i + n]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
    for w in s.split():
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 2.0      # mots pèsent +
    nrm = v.norm() + 1e-8
    return v / nrm


class DocumentLearner:
    """Cycle complet : apprend des documents (PDF/URL/texte), retrieve + citations.

    Memoire = liste de (chunk_text, source, embedding). Retrieval = plus proche
    voisin cosinus. Abstention si confiance < seuil."""

    def __init__(self, threshold: float = 0.55, chunk_size: int = 200, overlap: int = 40,
                 margin: float = 0.08):
        """threshold (absolu) + margin (cos1−cos2) : double critère d'abstention.
        HONNÊTE : le hash n-gramme est peu discriminant sémantiquement (audit H17) ;
        la marge+seuil filtre les OOD grossiers mais un vrai encodeur sémantique
        (mini-BERT/word2vec) reste nécessaire pour un retrieval de qualité. Le
        PIPELINE (learn→retrieve→citation→abstention) est complet et l'embedding
        est remplaçable sans changer l'API."""
        self.threshold = threshold
        self.margin = margin
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.embeddings: List[torch.Tensor] = []

    # ---- apprentissage (écrit dans la mémoire) ----
    def learn_text(self, text: str, source: str = "text") -> int:
        """Découpe un texte en chunks et les apprend. Retourne le nb de chunks appris."""
        if not text or text.startswith("[") and "error" in text.lower():
            return 0
        n_before = len(self.chunks)
        for ch in chunk_document(text, self.chunk_size, self.overlap):
            self.chunks.append(ch)
            self.sources.append(source)
            self.embeddings.append(text_embedding(ch))
        return len(self.chunks) - n_before

    def learn_pdf(self, path: str, max_pages: int = 10) -> int:
        """C7 : apprend depuis un PDF réel (PyMuPDF) → chunks → mémoire."""
        from .web_tools import parse_pdf
        text = parse_pdf(path, max_pages=max_pages)
        if not text or text.startswith("["):
            return 0
        return self.learn_text(text, source=f"pdf:{path}")

    def learn_url(self, url: str) -> int:
        """C8 : apprend depuis une URL réelle (HTTP fetch) → chunks → mémoire."""
        from .web_tools import fetch_url
        try:
            text = fetch_url(url)
        except Exception:
            return 0
        return self.learn_text(text, source=f"url:{url}")

    # ---- retrieval (avec abstention + citation) ----
    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 1
                 ) -> List[Tuple[str, str, float]]:
        """Retourne les top_k chunks pertinents (chunk, source, confiance).
        Double critère d'abstention : cos1 ≥ threshold ET (cos1−cos2) ≥ margin.
        → robuste aux faux-positifs du hash embedding (OOD rejeté)."""
        if not self.chunks:
            return []
        q = text_embedding(query)
        sims = [(float((q @ e).item()), i) for i, e in enumerate(self.embeddings)]
        sims.sort(reverse=True)
        out = []
        for rank, (sim, i) in enumerate(sims):
            if rank >= top_k:
                break
            # marge vs 2e meilleur (cos1−cos2) : distingue match net de match ambigu
            second = sims[rank + 1][0] if rank + 1 < len(sims) else 0.0
            if sim >= self.threshold and (sim - second) >= self.margin:
                out.append((self.chunks[i], self.sources[i], sim))
        return out

    def answer(self, query: str) -> Tuple[Optional[str], Optional[str], float]:
        """RAG : retourne (meilleur_chunk, source_citation, confiance) ou abstention."""
        res = self.retrieve(query, top_k=1)
        if not res:
            return None, None, 0.0        # abstention : "je ne sais pas"
        chunk, source, conf = res[0]
        return chunk, source, conf

    def knows(self, query: str) -> bool:
        return len(self.retrieve(query)) > 0

    def size(self) -> int:
        return len(self.chunks)


# ---------------- démo / self-test ----------------

def _demo() -> dict:
    dl = DocumentLearner()
    n1 = dl.learn_text(
        "La photosynthèse convertit l'énergie lumineuse en énergie chimique. "
        "Les plantes utilisent le dioxyde de carbone et l'eau pour produire du glucose. "
        "L'oxygène est libéré comme sous-produit de la photosynthèse.",
        source="bio:textbook")
    n2 = dl.learn_text(
        "La deuxième loi de Newton stipule que la force égale la masse fois l'accélération. "
        "F=ma est l'équation fondamentale de la mécanique classique.",
        source="phys:textbook")
    a1 = dl.answer("Que produit la photosynthèse ?")
    a2 = dl.answer("Quelle est la deuxième loi de Newton ?")
    a3 = dl.answer("Quelle est la capitale du Brésil ?")   # OOD → abstention
    return {
        "chunks_appris": dl.size(),
        "photosynthese": {"source": a1[1], "confiance": round(a1[2], 3),
                          "contient_glucose": "glucose" in (a1[0] or "")},
        "newton": {"source": a2[1], "confiance": round(a2[2], 3),
                   "contient_Fma": "F=ma" in (a2[0] or "")},
        "ood_bresil": {"abstention": a3[0] is None, "confiance": round(a3[2], 3)},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(_demo(), indent=2, ensure_ascii=False))
