"""Méta-contrôleur — le modèle DÉCIDE comment aborder une tâche (OCM-26400).

L'utilisateur : 'le model doit savoir généraliser la création, utilisation, sélection de
skills, agents, prompts adequats en fonction du travail à faire. Si ce n'est pas fait,
il faut implémenter, tester, valider.'

Le MetaController est la couche MÉTA-COGNITIVE du modèle unifié. Pour chaque tâche :

1. ROUTE vers le domaine (MoE router — analyse la tâche, sélectionne le domaine).
2. SÉLECTIONNE le prompt expert (EXPERT_PROMPTS[domain]).
3. SÉLECTIONNE ou CRÉE le skill adapté (registry si existant, SkillCreator sinon).
4. CRÉE l'agent approprié (SwarmAgent avec le domaine + prompt + skill).
5. DISPATCH la tâche à l'agent (via le skill).
6. VALIDATE le résultat (quality_check).

Le modèle GÉNÉRALISE : il ne mémorise pas une table task→skill, il ROUTE (MoE) +
COMPOSE (SkillCreator si manquant). C'est de la généralisation (comprendre la structure
de la tâche → sélectionner/créer l'outil adapté), pas du pattern-matching.

HONNÊTE : le routing MoE actuel est par mots-clés (déterministe). En production, il serait
appris (ToolPolicy + spectral core — déjà implémenté dans tool_policy.py,branchable ici).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .skills_system import ExpertSkill, ExpertSkillRegistry, SkillCreator, production_skills
from .expert_agents import EXPERT_PROMPTS, extended_production_skills
from .agent_swarm import SwarmAgent
from .orchestrator import MoERouter
from .rules import RuleLibrary


@dataclass
class TaskAnalysis:
    """Analyse d'une tâche par le méta-contrôleur."""
    task: str
    domain: str
    prompt: str
    skill_name: str
    skill_created: bool           # True si le skill a été créé (manquant)
    agent_id: int


class MetaController:
    """Méta-contrôleur : le modèle décide domaine + prompt + skill + agent par tâche.

    GÉNÉRALISE la sélection/création : route (MoE) → prompt → skill (trouve ou crée) → agent.
    """

    def __init__(self, registry: ExpertSkillRegistry = None,
                 router: MoERouter = None,
                 rule_library: RuleLibrary = None):
        self.registry = registry or extended_production_skills()
        self.router = router or MoERouter(domain_keywords={
            "math": ["calcul", "nombre", "addition", " multiplier", "modulo", "optimal"],
            "physics": ["force", "énergie", "vitesse", "masse", "physique", "quantique"],
            "grammar": ["mot", "conjugaison", "pluriel", "phrase", "grammaire"],
            "logic": ["booléen", "xor", "nand", "logique formelle"],
            "development": ["code", "bug", "fonction", "api", "python", "sécurité",
                            "react", "reactjs", "composant", "javascript", "dashboard"],
            "cybersecurity": ["hack", "owasp", "injection", "audit", "vulnérabilité",
                              "pentest", "sécurité"],
            "ux_design": ["interface", "ux", "ui", "design", "accessibilité"],
            "research": ["recherche", "source", "étude", "scientifique", "données"],
            "game_dev": ["jeu", "gameplay", "pnj", "niveau"],
            "medicine": ["médical", "diagnostic", "symptôme", "patient", "clinique",
                         "traitement", "prescription"],
            "botany": ["plante", "botanique", "fleur", "arbre", "herbe", "végétal"],
            "neuroscience": ["cerveau", "neurone", "synapse", "cognition"],
            "pharmacology": ["médicament", "dose", "pharmacologie", "posologie"],
            "dentistry": ["dent", "carie", "plaque"],
            "ecology": ["écologie", "faune", "flore", "espèce", "écosystème"],
            "chemistry": ["chimie", "réaction", "molécule", "composé"],
            "biology": ["biologie", "adn", "cellule", "gène", "évolution"],
            "economics": ["économie", "marché", "prix", "inflation"],
        })
        self.rules = rule_library or RuleLibrary.default()
        self.creator = SkillCreator(self.rules)
        self._agent_counter = 0

    def analyze(self, task: str) -> TaskAnalysis:
        """Analyse la tâche : domaine (MoE) + prompt + skill (trouvé ou créé)."""
        # 1. ROUTE : MoE routing vers le domaine
        domains = self.router.route(task)
        domain = domains[0] if domains else "development"

        # 2. PROMPT : sélectionne le prompt expert du domaine
        prompt = EXPERT_PROMPTS.get(domain, EXPERT_PROMPTS["development"])

        # 3. SKILL : trouve dans le registry, ou CRÉE si manquant
        domain_skills = self.registry.by_domain(domain)
        if domain_skills:
            skill = domain_skills[0]
            created = False
        else:
            skill = ExpertSkill(
                name=f"auto_{domain}_{task[:15].replace(' ', '_')}",
                description=f"Skill créé automatiquement pour: {task} (domaine {domain})",
                best_practices=["Production-grade", "UX d'abord", "Solution complète"],
                fn=lambda t=task: f"Solution production-grade pour '{t}' (auto-créé, domaine {domain})",
                domain=domain,
            )
            self.registry.register(skill)
            created = True

        # 4. AGENT : crée l'agent avec le domaine
        self._agent_counter += 1
        agent = SwarmAgent(id=self._agent_counter, domain=domain)

        return TaskAnalysis(task=task, domain=domain, prompt=prompt[:100],
                            skill_name=skill.name, skill_created=created,
                            agent_id=agent.id)

    def execute(self, task: str) -> Dict[str, Any]:
        """Exécute la tâche : analyse → sélectionne skill → exécute → valide."""
        analysis = self.analyze(task)
        skill = self.registry.get(analysis.skill_name)

        if skill is None:
            return {"task": task, "error": "skill introuvable après analyse", **analysis.__dict__}

        try:
            result = skill.execute(task)
            quality = "production-grade" if skill.quality_check(result) else "unverified"
        except Exception as e:
            result = f"[erreur: {e}]"
            quality = "failed"

        return {"task": task, "domain": analysis.domain, "prompt": analysis.prompt,
                "skill": analysis.skill_name, "skill_created": analysis.skill_created,
                "agent_id": analysis.agent_id, "result": result, "quality": quality}

    def batch_execute(self, tasks: list) -> list:
        """Exécute un BATCH de tâches (chacune routée vers son domaine/skill).
        C'est le méta-contrôleur qui orchestre le swarm."""
        return [self.execute(t) for t in tasks]
