"""Base de connaissance + retrieval avec abstention (OCM-26400, cahier des charges).

Implémente la brique « recherche dans la base de connaissance » + « si le model ne
sait pas alors activer son mode apprentissage » du cahier des charges.

La base indexe des concepts (clés = indices LearnedVocab, valeurs = contenu
associé : fait, définition, étiquette). Le retrieval se fait par plus proche
voisin cosinus dans la table E (LearnedVocab, P2). Si la similarité de la meilleure
match est SOUS un seuil, le système S'ABSTIENT (retourne UNKNOWN) — c'est le signal
« je ne sais pas » qui, dans le spec, déclenche le mode apprentissage (recherche
externe puis stockage du nouveau concept).

Intègre :
* P2 LearnedVocab : l'index de la base (E ∈ R^{V×64}).
* P3 abstention : seuil de confiance -> refuser de répondre si incertain.

UNKNOWN n'est pas une erreur : c'est le signal honnête d'incertitude épistémique
qui distingue « savoir » (retrieve confiant) de « ne pas savoir » (abstention ->
apprentissage). C'est l'opposé d'un LLM qui hallucine une réponse.
"""
from typing import Optional, Tuple, Any
import torch

from .learned_vocab import LearnedVocab

UNKNOWN = None          # sentinel : retrieval non confiant -> abstention


class KnowledgeBase:
    """Base de connaissance indexée par LearnedVocab. Retrieval cosinus + abstention."""

    def __init__(self, vocab: LearnedVocab, threshold: float = 0.5):
        self.vocab = vocab
        self.threshold = threshold
        self.values = {}                       # idx -> contenu (fait/définition/étiquette)

    def store(self, idx: int, value: Any) -> None:
        """Stocke un contenu associé au concept idx (apprentissage : on SAIT maintenant)."""
        self.values[idx] = value

    @torch.no_grad()
    def retrieve(self, query_vec: torch.Tensor, threshold: Optional[float] = None
                 ) -> Tuple[Optional[int], float]:
        """Retourne (idx, confidence) du concept le plus proche, ou (None, conf) si
        la confiance est sous le seuil => ABSTENTION (je ne sais pas)."""
        thr = self.threshold if threshold is None else threshold
        q = query_vec[: self.vocab.dim].to(torch.float32)
        q = q / (q.norm() + 1e-8)
        M = self.vocab._matrix().to(q.device)
        cos = q @ M.T                          # (V,) similarités
        cos1, idx = torch.max(cos, dim=0)
        conf = float(cos1.item())
        idx = int(idx.item())
        if conf < thr:
            return UNKNOWN, conf               # abstention
        return idx, conf

    def answer(self, query_vec: torch.Tensor, threshold: Optional[float] = None
               ) -> Tuple[Any, float]:
        """Renvoie (valeur stockée, confiance) ou (None, conf) si abstention.

        Le None est le signal 'je ne sais pas' -> dans le spec, déclenche le mode
        apprentissage (recherche externe + store du nouveau concept). Un concept
        retrouvé mais SANS valeur stockée est aussi une abstention (non appris)."""
        idx, conf = self.retrieve(query_vec, threshold=threshold)
        if idx is UNKNOWN or idx not in self.values:
            return None, conf
        return self.values[idx], conf

    def knows(self, query_vec: torch.Tensor, threshold: Optional[float] = None) -> bool:
        """Le système connaît-il une réponse confiant pour cette requête ?"""
        idx, _ = self.retrieve(query_vec, threshold=threshold)
        return idx is not UNKNOWN
