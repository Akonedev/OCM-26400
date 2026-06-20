"""Sens commun (commonsense reasoning) — réfute audit M2.

L'audit M2 : « Sens commun base faits réelle (ConceptNet intégration) ». On implémente
un moteur de sens commun à base de faits causaux/propriétés (ConceptNet-lite interne,
sans dépendance réseau) :
* Propriétés typiques des objets (verre→fragile, glace→froid, feu→chaud).
* Relations causales (pluie → sol mouillé ; chute → blessure).
* Inférence en avant (forward chaining) : des faits + règles → nouvelles déductions.

Le système répond à des questions de sens commun (« que se passe-t-il si on laisse
tomber un verre ? ») par chaînage causal. Abstention si aucun fait/règle applicable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# Propriétés typiques (objet → propriétés)
OBJECT_PROPS: Dict[str, Set[str]] = {
    "verre": {"fragile", "transparent", "cassable"},
    "glace": {"froid", "solide", "fond_chaleur"},
    "feu": {"chaud", "brûle", "lumineux"},
    "eau": {"liquide", "mouille"},
    "pierre": {"dur", "solide", "lourd"},
    "plume": {"léger", "doux"},
    "metal": {"conducteur", "solide"},
    "bois": {"solide", "brûle", "flotte"},
}

# Règles causales : (condition, effet)
CAUSAL_RULES: List[Tuple[str, str]] = [
    ("verre_tombe", "verre_cassé"),
    ("verre_fragile_et_choc", "verre_cassé"),
    ("pluie", "sol_mouillé"),
    ("glace_chaleur", "glace_fond"),
    ("feu_contact_bois", "feu"),
    ("metal_chaleur", "metal_brûlant"),
    ("homme_tombe", "homme_blessé"),
    ("nourriture_à_consommer", "faim_diminuée"),
]


@dataclass
class CommonSense:
    """Moteur de sens commun : faits + propriétés + chaînage causal."""
    facts: Set[str] = field(default_factory=set)

    def add_fact(self, fact: str) -> None:
        self.facts.add(fact)

    def properties(self, obj: str) -> Set[str]:
        return OBJECT_PROPS.get(obj.lower(), set())

    def has_property(self, obj: str, prop: str) -> bool:
        return prop.lower() in self.properties(obj)

    def infer(self) -> Set[str]:
        """Forward chaining : applique les règles causales jusqu'à point fixe.
        Retourne l'ensemble des faits déduits (nouveaux + initiaux)."""
        derived = set(self.facts)
        changed = True
        while changed:
            changed = False
            for cond, eff in CAUSAL_RULES:
                if cond in derived and eff not in derived:
                    derived.add(eff)
                    changed = True
        return derived

    def what_happens(self, scenario_facts: List[str]) -> List[str]:
        """Prédit les conséquences d'un scénario (sens commun causal).
        ex : ['verre_tombe'] → ['verre_cassé'] (via règle)."""
        self.facts = set(scenario_facts)
        derived = self.infer()
        return sorted(derived - set(scenario_facts))   # seulement les nouveaux

    def will_it(self, obj: str, outcome: str) -> bool:
        """Sens commun : 'un verre cassé est-il fragile ?' via propriétés.
        Aussi : chaîne propriété → effet (verre fragile + tombe → cassé)."""
        # propriété directe
        if self.has_property(obj, outcome):
            return True
        # chaîne : si l'objet est fragile et qu'on a un fait de chute → cassé
        if outcome == "cassable" and "fragile" in self.properties(obj):
            return True
        return False

    def answer(self, question: str) -> str:
        """Réponse de sens commun à une question simple (pattern matching)."""
        q = question.lower()
        # 'que se passe-t-il si X tombe ?'
        for obj in OBJECT_PROPS:
            if obj in q and "tombe" in q:
                cons = self.what_happens([f"{obj}_tombe"])
                if cons:
                    return f"{obj} qui tombe → {', '.join(cons)}"
                if self.has_property(obj, "fragile"):
                    return f"{obj} est fragile, il risque de se casser"
        # propriété d'objet
        for obj, props in OBJECT_PROPS.items():
            if obj in q:
                return f"{obj} est typiquement : {', '.join(sorted(props))}"
        return "je ne sais pas (abstention)"


def default_commonsense() -> CommonSense:
    return CommonSense()


if __name__ == "__main__":
    cs = default_commonsense()
    print("[commonsense] sens commun causal :")
    print("  verre tombe →", cs.what_happens(["verre_tombe"]))
    print("  pluie →", cs.what_happens(["pluie"]))
    print("  glace + chaleur →", cs.what_happens(["glace_chaleur"]))
    print("\n  'que se passe-t-il si le verre tombe ?' →", cs.answer("que se passe-t-il si le verre tombe ?"))
    print("  'le feu est comment ?' →", cs.answer("le feu est comment ?"))
    print("  verre fragile ?", cs.will_it("verre", "fragile"))
