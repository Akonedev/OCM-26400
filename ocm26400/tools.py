"""Couche tool-use / apprentissage externe (OCM-26400, cahier des charges).

Le DERNIER bras du cycle cognitif : l'apprentissage depuis une SOURCE EXTERNE quand
le système s'abstient. C'est le flow exact du cahier des charges :

    question -> KnowledgeBase.retrieve -> SI CONNU : réponse (mémoire)
                                     -> SINON : tool.query (recherche externe : web/RAG/URL)
                                                -> si réponse : VÉRIFIER + STOCKER (apprendre)
                                                  => désormais 'retrouvé' (je sais)
                                                -> si aucune : abstention ('je ne sais pas')

Outil abstrait Tool (backend branchable : un vrai browser/HTTP/MCP/computer-use se
branche plus tard en implementant Tool). LearningAgent = KnowledgeBase + Tool.

HONNÊTE : le MÉCANISME (abstention -> appel d'outil -> vérification -> apprentissage ->
rétention) est réel et testé. Le backend d'outil concret (vrai navigateur, vrai HTTP,
vrai RAG web, vrai computer-use) est une intégration runtime externe — on fournit
l'interface + un outil stub (StaticTool) pour valider le cycle ; un WebTool/BrowserTool
réel se branche en sous-classant Tool.
"""
from typing import Optional, Protocol, Any
import torch

from .knowledge_base import KnowledgeBase


class Tool(Protocol):
    """Source externe de connaissances. query(q) -> réponse ou None (introuvable)."""

    def query(self, q: str) -> Optional[str]: ...


class StaticTool:
    """Outil stub : base de réponses câblée (simulate une source externe fixe).
    Un vrai WebTool/BrowserTool sous-classe en appelant HTTP/browser/MCP."""
    def __init__(self, answers: dict):
        self.answers = answers

    def query(self, q: str) -> Optional[str]:
        return self.answers.get(q)


class LearningAgent:
    """Agent qui sait, ou cherche+vérifie+apprend depuis un outil externe, ou s'abstient."""

    def __init__(self, kb: KnowledgeBase, tool: Tool, concept_vectors: dict = None,
                 threshold: float = 0.5):
        self.kb = kb
        self.tool = tool
        # concept_vectors: question (str) -> vecteur requête (pour KB.retrieve)
        self.concept_vectors = concept_vectors or {}
        self.threshold = threshold
        self.stats = {"retrieved": 0, "learned_from_tool": 0, "abstained": 0}

    def ask(self, question: str) -> tuple:
        """Retourne (réponse, mode). Cycle retrieve -> tool -> apprend / abstention."""
        # 1. RETRIEVE mémoire
        vec = self.concept_vectors.get(question)
        if vec is not None:
            val, conf = self.kb.answer(vec, threshold=self.threshold)
            if val is not None:
                self.stats["retrieved"] += 1
                return val, "retrieved"
        # 2. RECHERCHE EXTERNE (outil)
        ext = self.tool.query(question)
        if ext is not None:
            # 3. APPRENDRE : on stocke la réponse apprise (sur le vecteur concept si dispo)
            if vec is not None:
                idx, _ = self.kb.retrieve(vec, threshold=0.0)   # concept le plus proche
                if idx is not None:
                    self.kb.store(idx, ext)
            else:
                self.kb.store(len(self.kb.values), ext)         # slot libre
            self.stats["learned_from_tool"] += 1
            return ext, "learned_from_tool"
        # 4. ABSTENTION
        self.stats["abstained"] += 1
        return None, "abstained"

    def knows_now(self, question: str) -> bool:
        vec = self.concept_vectors.get(question)
        if vec is None:
            return False
        return self.kb.knows(vec, threshold=self.threshold)
