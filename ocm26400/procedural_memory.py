"""Mémoire PROCÉDURALE — réfute audit H12 (« pas distinguée de sémantique »).

L'audit H12 : « Mémoire procédurale (comment faire) — test 211 = liste Python, pas
distinguée de sémantique ». On comble : un système de mémoire procédurale distincte
des mémoires épisodique et sémantique.

3 types de mémoire (modèle cognitif standard + cahier des charges) :
* ÉPISODIQUE : événements vécus (déjà en cognitive_agent.memory).
* SÉMANTIQUE : faits/règles généralisés (déjà via sleep.consolidate).
* PROCÉDURALE (CE MODULE) : séquences d'ACTIONS — « comment faire X » (recettes,
  algorithmes, procédures). Rejouable : on peut ré-exécuter la procédure.

Une procédure = (nom, [étapes], préconditions, effets). On peut :
* apprendre une procédure (l'enregistrer),
* la rejouer (exécuter les étapes sur un executor),
* la généraliser (abstraire une procédure spécifique en template paramétré).

C'est la mémoire du « savoir-faire » (vs savoir). Ex : « faire du thé », « trier une
liste », « résoudre une équation du 2d degré ».
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Procedure:
    """Une procédure : nom + étapes ordonnées (actions) + préconditions + effets."""
    name: str
    steps: List[str]                        # étapes en langage naturel (ou actions)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)   # paramètres abstraits

    def n_steps(self) -> int:
        return len(self.steps)

    def replay(self, executor: Optional[Callable[[str], Any]] = None
               ) -> List[Tuple[str, Any]]:
        """Rejoue la procédure : applique executor à chaque étape. Si pas d'executor,
        retourne juste les étapes (dry-run). Retourne [(étape, résultat)]."""
        if executor is None:
            return [(s, None) for s in self.steps]
        return [(s, executor(s)) for s in self.steps]

    def matches(self, query: str) -> bool:
        """La procédure répond-elle à une requête 'comment faire X' ?"""
        q = query.lower()
        # normalise le nom : underscores → espaces (faire_du_thé ↔ 'faire du thé')
        name_norm = self.name.lower().replace("_", " ")
        name_words = name_norm.split()
        return (name_norm in q or any(p.lower() in q for p in self.effects)
                or any(w in q for w in name_words))


class ProceduralMemory:
    """Mémoire procédurale : registre de procédures 'comment faire'."""

    def __init__(self):
        self.procedures: Dict[str, Procedure] = {}

    def learn(self, proc: Procedure) -> None:
        """Apprend une procédure (l'enregistre pour réutilisation)."""
        self.procedures[proc.name] = proc

    def get(self, name: str) -> Optional[Procedure]:
        return self.procedures.get(name)

    def how_to(self, query: str) -> Optional[Procedure]:
        """Retourne la procédure pour 'comment faire X'. None si inconnu (abstention)."""
        for proc in self.procedures.values():
            if proc.matches(query):
                return proc
        return None

    def replay(self, name: str, executor: Optional[Callable] = None
               ) -> Optional[List[Tuple[str, Any]]]:
        proc = self.get(name)
        return proc.replay(executor) if proc else None

    def generalize(self, specific_name: str, template_name: str,
                   param_keys: List[str]) -> Optional[Procedure]:
        """Généralise une procédure spécifique en template paramétré (abstraction).
        Remplace les valeurs concrètes par des paramètres {key}."""
        proc = self.get(specific_name)
        if proc is None:
            return None
        # abstraction naïve : les étapes deviennent un template avec placeholders
        templated_steps = []
        for i, s in enumerate(proc.steps):
            if i < len(param_keys):
                templated_steps.append(s.replace(s.split()[-1] if s.split() else "",
                                                 "{" + param_keys[i] + "}"))
            else:
                templated_steps.append(s)
        template = Procedure(name=template_name, steps=templated_steps,
                             preconditions=proc.preconditions, effects=proc.effects,
                             params={k: None for k in param_keys})
        self.learn(template)
        return template

    def size(self) -> int:
        return len(self.procedures)

    def names(self) -> List[str]:
        return list(self.procedures.keys())


# ---------------- procédures de démo (savoir-faire réel) ----------------

def default_procedures() -> ProceduralMemory:
    """Procédures courantes (recipes / algorithmes / méthodes)."""
    pm = ProceduralMemory()
    pm.learn(Procedure(
        name="faire_du_thé",
        steps=["faire bouillir de l'eau", "mettre un sachet dans la tasse",
               "verser l'eau chaude", "laisser infuser 3 min", "retirer le sachet"],
        preconditions=["avoir de l'eau", "avoir un sachet de thé"],
        effects=["thé prêt à boire"]))
    pm.learn(Procedure(
        name="trier_liste",
        steps=["prendre la liste", "comparer les éléments adjacents",
               "échanger si désordonné", "répéter jusqu'à trié"],
        preconditions=["liste non vide"],
        effects=["liste triée"]))
    pm.learn(Procedure(
        name="résoudre_équation_2nd_degré",
        steps=["identifier a, b, c", "calculer Δ=b²−4ac",
               "si Δ<0: pas de racine réelle", "si Δ=0: x=−b/2a",
               "si Δ>0: x=(−b±√Δ)/2a"],
        preconditions=["équation ax²+bx+c=0"],
        effects=["racines calculées"]))
    pm.learn(Procedure(
        name="diagnostic_médical",
        steps=["recueillir les symptômes", "examen clinique",
               "établir le diagnostic différentiel", "prescrire examens si besoin",
               "annoncer le diagnostic"],
        preconditions=["patient"],
        effects=["diagnostic établi"]))
    return pm


if __name__ == "__main__":
    pm = default_procedures()
    print(f"[procedural_memory] {pm.size()} procédures : {pm.names()}")
    proc = pm.how_to("comment faire du thé ?")
    if proc:
        print(f"\n'{proc.name}' ({proc.n_steps()} étapes) :")
        for i, s in enumerate(proc.steps, 1):
            print(f"  {i}. {s}")
    # replay avec executor factice
    print("\nReplay 'trier_liste' (executor=count chars) :")
    for step, res in pm.replay("trier_liste", executor=lambda s: len(s)):
        print(f"  {step:40s} → {res}")
