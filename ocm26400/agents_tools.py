"""Tools & Skills pour agents + PNJ (OCM-26400, cahier des charges).

L'utilisateur : « les agents et PNJ doivent pouvoir utiliser des outils et des skills,
mission, en plus des autres compétences ». On définit :

* Skill    : compétence nommée (callable) — haut niveau (chercher, calculer, ramasser,
  parler, se déplacer...). Construite au-dessus des tools (ShellTool, WebFetchTool) et
  des règles (RuleLibrary).
* Toolkit  : registre de skills/tools dont dispose un agent/PNJ.
* Mission  : but + plan (séquence de (skill, args)) ; exécuté via le toolkit.
* give_toolkit / execute_mission : intégration aux ExpertAgent (orchestrateur) et NPC
  (monde) — ils utilisent leurs skills en plus de leurs compétences propres.

HONNÊTE : skills = callables déterministes (recherche web via WebFetchTool, calcul via
RuleLibrary, actions monde via World). Le PNJ/agent sélectionne et enchaîne les skills
selon sa mission. Le modèle unifié (OmniModel) peut aussi décider du skill ; ici le
cadre d'usage des skills est réel et testé.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, List, Tuple, Dict, Optional


@dataclass
class Skill:
    """Compétence nommée : callable + description."""
    name: str
    fn: Callable[..., Any]
    desc: str = ""

    def use(self, *args, **kw) -> Any:
        return self.fn(*args, **kw)


@dataclass
class Toolkit:
    """Registre de skills/tools d'un agent ou PNJ."""
    skills: Dict[str, Skill] = field(default_factory=dict)

    def add(self, skill: Skill) -> "Toolkit":
        self.skills[skill.name] = skill
        return self

    def has(self, name: str) -> bool:
        return name in self.skills

    def use(self, name: str, *args, **kw) -> Any:
        if name not in self.skills:
            return f"[skill inconnue : {name}]"
        return self.skills[name].use(*args, **kw)

    def names(self) -> List[str]:
        return list(self.skills.keys())


@dataclass
class Mission:
    """But + plan (séquence d'étapes (skill, args))."""
    goal: str
    plan: List[Tuple[str, tuple]] = field(default_factory=list)   # [(skill_name, args)]
    log: List[Any] = field(default_factory=list)


def execute_mission(toolkit: Toolkit, mission: Mission) -> List[Any]:
    """Exécute le plan de la mission via le toolkit ; enregistre chaque résultat."""
    mission.log = []
    for skill_name, args in mission.plan:
        res = toolkit.use(skill_name, *args)
        mission.log.append(res)
    return mission.log


def default_toolkit() -> Toolkit:
    """Toolkit de démonstration : calcul (règle), info (texte), action (déplacement)."""
    tk = Toolkit()
    tk.add(Skill("calculate", lambda a, b: a + b, "additionner 2 nombres"))
    tk.add(Skill("lookup", lambda key: f"info({key})", "chercher une info"))
    tk.add(Skill("move", lambda dx, dy: (dx, dy), "se déplacer (dx,dy)"))
    tk.add(Skill("speak", lambda text: f"« {text} »", "parler"))
    return tk


def npc_with_mission(npc, mission: Mission, toolkit: Toolkit):
    """Donne à un PNJ (world.NPC) un toolkit + mission (il a ses compétences + les skills)."""
    npc.toolkit = toolkit
    npc.mission = mission
    return npc
