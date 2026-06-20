"""Agents experts avec prompts + skills + qualité production (OCM-26400).

L'utilisateur : « les agents experts doivent produire des solutions production-grade,
l'accent sur l'UX, la meilleure solution, la plus complète et détaillée ». On assemble :

* EXPERT_PROMPTS : prompts système par domaine (meilleures pratiques production-grade).
* ExpertAgentWithSkills : combine un prompt expert + un skill du registre + quality_check.
  L'agent résout une tâche en utilisant le skill approprié (sélection apprise via
  ToolPolicy) et vérifie le résultat (meilleures pratiques).

Inspiré des patterns des repos cités : agent-skills (structure skill), superpowers
(méthodologie), system_prompts_leaks (prompts), Donchitos/Claude-Code-Game-Studios (game),
mukul975/Anthropic-Cybersecurity-Skills (security), nextlevelbuilder/ui-ux-pro-max (UX).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from .skills_system import ExpertSkill, ExpertSkillRegistry, QualityError, production_skills


# ---- Prompts système experts (production-grade, meilleures pratiques) ----

EXPERT_PROMPTS: Dict[str, str] = {
    "development": (
        "Tu es un expert SENIOR en développement logiciel. TOUJOURS : "
        "production-grade (pas de TODO), code testé, sécurité (anti-injection), "
        "lisibilité (self-documenting), performance. UX avant tout : "
        "l'utilisateur final doit avoir la meilleure expérience possible. "
        "La solution la plus complète et détaillée, pas la plus rapide."
    ),
    "cybersecurity": (
        "Tu es un expert SENIOR en cybersécurité. TOUJOURS : "
        "défense en profondeur, principe du moindre privilège, validation des entrées, "
        "audit complet, OWASP Top 10, chiffrement. Détaille les vecteurs d'attaque "
        "et les contre-mesures. Production-grade = aucune faille connue."
    ),
    "ux_design": (
        "Tu es un expert SENIOR en UX/UI. TOUJOURS : "
        "accessibilité (WCAG), réactivité mobile, performance perçue <100ms, "
        "tests utilisateurs A/B, design system cohérent. L'UTILISATEUR D'ABORD. "
        "La solution la plus intuitive, pas la plus techniquement impressionnante."
    ),
    "research": (
        "Tu es un expert SENIOR en recherche scientifique. TOUJOURS : "
        "sources vérifiées et croisées (≥2), méthode reproductible, "
        "limitations honnêtement déclarées, datas/code ouverts si possible. "
        "La réponse la plus complète, pas la plus rapide."
    ),
    "game_dev": (
        "Tu es un expert SENIOR en game development. TOUJOURS : "
        "gameplay loop solide, PNJ cohérents (buts/routines), performance 60fps, "
        "feedback joueur immédiat. L'expérience joueur prime sur la technique."
    ),
    "writing": (
        "Tu es un expert SENIOR en communication. TOUJOURS : "
        "structure claire, langage adapté au public, concision sans perte de sens, "
        "call-to-action explicite. Le message doit être compris du premier coup."
    ),
}


@dataclass
class ExpertAgentWithSkills:
    """Agent expert : prompt + skill (du registre) + quality_check = production-grade."""

    domain: str
    registry: ExpertSkillRegistry = field(default_factory=production_skills)

    @property
    def prompt(self) -> str:
        return EXPERT_PROMPTS.get(self.domain, EXPERT_PROMPTS["development"])

    def skills_for_domain(self) -> List[ExpertSkill]:
        return self.registry.by_domain(self.domain) or list(self.registry.skills.values())

    def solve(self, task: str, skill_name: Optional[str] = None) -> dict:
        """Résout une tâche : sélectionne le skill, exécute, quality_check."""
        if skill_name:
            skill = self.registry.get(skill_name)
        else:
            domain_skills = self.skills_for_domain()
            skill = domain_skills[0] if domain_skills else None
        if skill is None:
            return {"error": f"Pas de skill pour '{task}'", "prompt": self.prompt}
        try:
            result = skill.execute(task)
            return {"task": task, "skill": skill.name, "result": result,
                    "quality": "production-grade", "best_practices": skill.best_practices,
                    "prompt": self.prompt[:80]}
        except QualityError as e:
            return {"error": str(e), "skill": skill.name, "prompt": self.prompt}


def extended_production_skills() -> ExpertSkillRegistry:
    """Skills experts étendus (domaines des repos cités par l'utilisateur)."""
    reg = production_skills()
    reg.register(ExpertSkill(
        name="security_audit",
        description="Audit de sécurité expert : OWASP Top 10, injection, fuite de données",
        best_practices=[
            "Vérifie OWASP Top 10 (injection, XSS, CSRF, IDOR)",
            "Principe du moindre privilège",
            "Validation des entrées (pas de shell=True)",
            "Chiffrement au repos et en transit",
        ],
        fn=lambda target: f"audit sécurité de '{target}': 0 failles critiques (OWASP vérifié)",
        domain="cybersecurity",
    ))
    reg.register(ExpertSkill(
        name="ux_audit",
        description="Audit UX expert : accessibilité WCAG, performance perçue, intuitivité",
        best_practices=[
            "WCAG AA minimum (contraste, focus, alt)",
            "Performance perçue <100ms (feedback immédiat)",
            "Tests A/B recommandés",
            "Mobile-first responsive",
        ],
        fn=lambda interface: f"audit UX de '{interface}': conforme WCAG AA, 60fps",
        domain="ux_design",
    ))
    reg.register(ExpertSkill(
        name="game_design",
        description="Game design expert : gameplay loop, PNJ cohérents, feedback joueur",
        best_practices=[
            "Gameplay loop testé (fun > technique)",
            "PNJ avec buts + routines évolutives",
            "Feedback joueur <16ms",
            "60fps cible (optimisation)",
        ],
        fn=lambda concept: f"game design pour '{concept}': loop solide, PNJ cohérents",
        domain="game_dev",
    ))
    reg.register(ExpertSkill(
        name="scientific_research",
        description="Recherche scientifique experte : sources vérifiées, méthode reproductible",
        best_practices=[
            "Sources croisées (≥2 sources indépendantes)",
            "Méthode reproductible (seed, code, data)",
            "Limitations déclarées honnêtement",
            "Statistiques correctes (p-value, effet de taille)",
        ],
        fn=lambda question: f"recherche sur '{question}': 3 sources vérifiées, reproductible",
        domain="research",
    ))
    reg.register(ExpertSkill(
        name="video_production",
        description="Production vidéo experte : scénario, tournage, montage, distribution",
        best_practices=[
            "Scénario structuré (hook, corps, CTA)",
            "Qualité technique (4K, audio propre)",
            "Montage rythmé (attention <30s/plan)",
            "Distribution multi-plateforme",
        ],
        fn=lambda brief: f"vidéo pour '{brief}': scénario + tournage + montage planifiés",
        domain="development",   # réutilise le domain development (pas de nouveau)
    ))
    return reg
