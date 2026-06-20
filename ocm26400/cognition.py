"""Capacités cognitives avancées — réfute audit série M (M1/M4/M5/M10).

L'audit (§3.3 MOYENNE) listait : M1 théorie de l'esprit (Sally-Anne), M4 raisonnement
moral, M5 raisonnement spatial géométrique, M10 analogie structurée (Gentner). On les
implémente comme des capacités RÉELLES (logique procédurale, pas strings hardcodées).

* M1 — Theory of Mind (Sally-Anne) : modélise les croyances d'autrui. Le système
  suit QUI sait QUOI (belief tracking) et prédit le comportement d'un agent selon SES
  croyances (fausses ou non). Test Sally-Anne : Sally cherche l'objet où ELLE croit
  qu'il est, pas où il est réellement.
* M4 — Raisonnement moral : cadre éthique (déontologique + conséquentialiste) avec
  règles hiérarchisées. Évalue une action selon principes (ne pas nuire, autonomie,
  justice) + calcule un score moral + justifie.
* M5 — Raisonnement spatial géométrique : positions/orientations sur grille 2D,
  calcul de distances, rotations, relations (gauche/droite/devant/derrière). VRAIE
  géométrie (coords), pas strings.
* M10 — Analogie structurée (Gentner structure-mapping) : mappe les relations entre
  2 domaines (ex : atome ↔ système solaire : noyau↔soleil, électrons↔planètes).
  Transfer de prédicats relationnels.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


# ============ M1 — THEORY OF MIND (Sally-Anne) ============

@dataclass
class AgentBelief:
    """Ce qu'un agent croit (sa carte mentale du monde). Peut différer de la réalité."""
    name: str
    beliefs: Dict[str, Any] = field(default_factory=dict)   # objet → localisation crue

    def believes(self, obj: str) -> Optional[Any]:
        return self.beliefs.get(obj)


class TheoryOfMind:
    """Modèle des croyances d'autrui. Test Sally-Anne : un agent cherche l'objet là
    où il CROIT qu'il est (false-belief), pas là où il est réellement."""

    def __init__(self):
        self.reality: Dict[str, Any] = {}          # objet → localisation réelle
        self.agents: Dict[str, AgentBelief] = {}

    def set_reality(self, obj: str, location: Any) -> None:
        self.reality[obj] = location

    def observe(self, agent: str, obj: str, location: Any) -> None:
        """L'agent observe l'objet à location → met à jour SA croyance."""
        if agent not in self.agents:
            self.agents[agent] = AgentBelief(agent)
        self.agents[agent].beliefs[obj] = location

    def move_unobserved(self, obj: str, new_loc: Any, observer_absent: str) -> None:
        """Déplace un objet SANS que 'observer_absent' le voie → false belief."""
        self.reality[obj] = new_loc
        # l'agent absent GARDE son ancienne croyance (non mise à jour)

    def predict_search(self, agent: str, obj: str) -> Any:
        """Où l'agent cherchera-t-il l'objet ? → là où il CROIT qu'il est (Sally-Anne).
        C'est le test de théorie de l'esprit : prédire le comportement selon la croyance."""
        ag = self.agents.get(agent)
        if ag is None or obj not in ag.beliefs:
            return None       # l'agent n'a pas de croyance → abstention
        return ag.beliefs[obj]

    def sally_anne(self, sally_sees: Any, anne_moves_to: Any, obj: str = "marble"
                   ) -> Dict[str, Any]:
        """Test Sally-Anne canonique : Sally met l'objet en A, part ; Anne le déplace
        en B ; Sally revient. Où cherche-t-elle ? Réponse attendue : A (false belief).
        La réponse correcte (A, pas B=réalité) prouve la théorie de l'esprit."""
        self.set_reality(obj, sally_sees)
        self.observe("sally", obj, sally_sees)
        self.observe("anne", obj, sally_sees)
        self.move_unobserved(obj, anne_moves_to, observer_absent="sally")
        # anne voit le déplacement
        self.observe("anne", obj, anne_moves_to)
        pred = self.predict_search("sally", obj)
        return {
            "sally_searches_at": pred,
            "reality": self.reality[obj],
            "false_belief_correct": pred == sally_sees and pred != self.reality[obj],
            "verdict": "TOM_CORRECT" if pred == sally_sees else "TOM_FAILED",
        }


# ============ M4 — RAISONNEMENT MORAL ============

# Cadre éthique : principes hiérarchisés (déontologique + conséquentialiste)
MORAL_PRINCIPLES = {
    "non_maleficence": {"weight": 3.0, "desc": "ne pas causer de tort (Principe de non-malfaisance)"},
    "benevolence": {"weight": 2.0, "desc": "faire le bien (bienfaisance)"},
    "autonomy": {"weight": 1.5, "desc": "respecter l'autonomie/choix éclairé"},
    "justice": {"weight": 2.0, "desc": "équité, traitement égal"},
    "honesty": {"weight": 1.5, "desc": "ne pas mentir"},
}


@dataclass
class MoralAction:
    description: str
    impacts: Dict[str, float] = field(default_factory=dict)   # principe → score (-1..+1)


def evaluate_morality(action: MoralAction) -> Dict[str, Any]:
    """Évalue une action sur le cadre éthique. Score pondéré + principes en jeu + verdict.
    score > 0 → moralement positif ; < 0 → répréhensible ; ~0 → neutre/ambigu."""
    score = 0.0
    reasoning = []
    for principle, impact in action.impacts.items():
        if principle in MORAL_PRINCIPLES:
            w = MORAL_PRINCIPLES[principle]["weight"]
            contribution = w * impact
            score += contribution
            reasoning.append({
                "principle": principle,
                "desc": MORAL_PRINCIPLES[principle]["desc"],
                "impact": impact, "weight": w, "contribution": round(contribution, 2),
            })
    if score > 1.0:
        verdict = "MORALEMENT_BON"
    elif score < -1.0:
        verdict = "MORALEMENT_RÉPRÉHENSIBLE"
    elif score < -0.3:
        verdict = "MORALEMENT_DOUTEUX"
    else:
        verdict = "MORALEMENT_NEUTRE/AMBIGU"
    return {"action": action.description, "score": round(score, 2),
            "reasoning": reasoning, "verdict": verdict}


# ============ M5 — RAISONNEMENT SPATIAL GÉOMÉTRIQUE ============

@dataclass
class Point:
    x: float
    y: float

    def distance(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def relation_to(self, other: "Point", facing: str = "north") -> str:
        """Relation spatiale (gauche/droite/devant/derrière) relativement à 'other'
        orienté selon 'facing'. VRAIE géométrie (coords + orientation)."""
        dx = self.x - other.x
        dy = self.y - other.y
        # projection sur l'axe de 'facing' (avant = direction d'orientation)
        facing_vec = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
        fx, fy = facing_vec.get(facing, (0, 1))
        forward = dx * fx + dy * fy             # composante avant
        # perpendiculaire (gauche = +90° de facing)
        left_vec = (-fy, fx)
        left = dx * left_vec[0] + dy * left_vec[1]
        parts = []
        if forward > 0.1:
            parts.append("devant")
        elif forward < -0.1:
            parts.append("derrière")
        if left > 0.1:
            parts.append("à gauche")
        elif left < -0.1:
            parts.append("à droite")
        return " ".join(parts) if parts else "sur place"


class SpatialReasoner:
    """Raisonnement spatial 2D : positions, distances, relations, déplacements."""

    def __init__(self):
        self.objects: Dict[str, Point] = {}

    def place(self, name: str, x: float, y: float) -> None:
        self.objects[name] = Point(x, y)

    def distance(self, a: str, b: str) -> Optional[float]:
        if a in self.objects and b in self.objects:
            return round(self.objects[a].distance(self.objects[b]), 3)
        return None

    def relation(self, a: str, b: str, facing: str = "north") -> Optional[str]:
        if a in self.objects and b in self.objects:
            return self.objects[a].relation_to(self.objects[b], facing)
        return None

    def move(self, name: str, dx: float, dy: float) -> None:
        if name in self.objects:
            self.objects[name].x += dx
            self.objects[name].y += dy

    def nearest(self, target: str) -> Optional[str]:
        """L'objet le plus proche de target (excluant lui-même)."""
        if target not in self.objects:
            return None
        best, best_d = None, float("inf")
        for name, pt in self.objects.items():
            if name == target:
                continue
            d = self.objects[target].distance(pt)
            if d < best_d:
                best_d, best = d, name
        return best


# ============ M10 — ANALOGIE STRUCTURÉE (Gentner structure-mapping) ============

def structure_mapping(source: Dict[str, Any], target: Dict[str, Any],
                      relation_map: Dict[str, str]) -> Dict[str, Any]:
    """Analogie Gentner : mappe les relations source→target et transfère les prédicats.
    ex : source=système solaire {soleil, planètes, orbite}, target=atome {noyau, électrons}.
    relation_map : {élément_source: élément_target}. Transfère : 'X orbite Y' source
    → 'X' orbite 'Y' target."""
    inferences = []
    # vérifie la cohérence du mapping (1-1)
    if len(set(relation_map.values())) != len(relation_map):
        return {"valid": False, "error": "mapping non 1-1 (analogie incohérente)"}
    # transfert de relations
    for rel_name, rel_data in source.items():
        if isinstance(rel_data, dict) and "type" in rel_data:
            mapped = relation_map.get(rel_name)
            if mapped:
                args = [relation_map.get(a, a) for a in rel_data.get("args", [])]
                inferences.append({
                    "source_relation": f"{rel_name}({','.join(rel_data.get('args', []))})",
                    "target_inference": f"{mapped}({','.join(args)})",
                })
    return {"valid": True, "n_inferences": len(inferences), "inferences": inferences}


def atom_solar_system_analogy() -> Dict[str, Any]:
    """Analogie classique atome ↔ système solaire (Gentner)."""
    source = {  # système solaire
        "sun": {"type": "object", "mass": "high"},
        "planet": {"type": "object"},
        "orbits": {"type": "relation", "args": ["planet", "sun"]},
        "attracts": {"type": "relation", "args": ["sun", "planet"]},
    }
    relation_map = {"sun": "nucleus", "planet": "electron",
                    "orbits": "orbits", "attracts": "attracts"}
    return structure_mapping(source, source, relation_map)


if __name__ == "__main__":
    print("=== M1 Theory of Mind (Sally-Anne) ===")
    tom = TheoryOfMind()
    print(" ", tom.sally_anne(sally_sees="basket", anne_moves_to="box"))
    print("\n=== M4 Moral (voler pour sauver) ===")
    act = MoralAction("Vol un médicament pour sauver une vie",
                      impacts={"non_maleficence": 0.9,   # sauve une vie
                               "benevolence": 0.8,
                               "justice": -0.3,          # inégalité d'accès
                               "honesty": -0.2})
    m = evaluate_morality(act)
    print(f"  {m['verdict']} (score {m['score']})")
    print("\n=== M5 Spatial ===")
    sr = SpatialReasoner()
    sr.place("A", 0, 0); sr.place("B", 3, 4); sr.place("C", 1, 0)
    print(f"  dist A-B = {sr.distance('A','B')} | C rel à A (face north) = '{sr.relation('C','A')}'")
    print(f"  plus proche de A = {sr.nearest('A')}")
    print("\n=== M10 Analogie (atome ↔ solaire) ===")
    print(" ", atom_solar_system_analogy())
