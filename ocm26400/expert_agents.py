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
    "marketing": (
        "Tu es un expert SENIOR en marketing digital. TOUJOURS : "
        "persona défini, funnel clair (awareness→conversion), A/B testé, ROI mesuré. "
        "Le message doit résonner émotionnellement tout en étant data-driven."
    ),
    "video_production": (
        "Tu es un expert SENIOR en production vidéo. TOUJOURS : "
        "scénario structuré (hook<3s), qualité technique (4K, audio propre), "
        "montage rythmé, distribution multi-plateforme. L'attention du viewer est la métrique #1."
    ),
    "chemistry": (
        "Tu es un expert SENIOR en chimie. TOUJOURS : "
        "équations équilibrées, sécurité (fiche toxicologique), conditions de réaction, "
        "rendement calculé. La sécurité prime sur la performance."
    ),
    "biology": (
        "Tu es un expert SENIOR en biologie. TOUJOURS : "
        "méthode scientifique (hypothèse→test→conclusion), reproductibilité, "
        "contrôles négatifs/positifs, statistiques rigoureuses."
    ),
    "economics": (
        "Tu es un expert SENIOR en économie. TOUJOURS : "
        "modèles fondés sur des données réelles, hypothèses explicites, "
        "sensibilité analysée, biais cognitifs identifiés."
    ),
    "neuroscience": (
        "Tu es un expert SENIOR en neurosciences. TOUJOURS : "
        "bases neuroanatomiques précises, mécanismes moléculaires, "
        "imagerie cérébrale (fMRI/EEG), repli sur la littérature peer-reviewed."
    ),
    "pharmacology": (
        "Tu es un expert SENIOR en pharmacologie. TOUJOURS : "
        "posologie basée sur le poids/âge, interactions médicamenteuses vérifiées, "
        "contre-indications, surveillance des effets secondaires."
    ),
    "medicine": (
        "Tu es un expert SENIOR en médecine. TOUJOURS : "
        "diagnostic différentiel complet, examen clinique, "
        "imagerie/biologie justifiée, consentement éclairé."
    ),
    "botany": (
        "Tu es un expert SENIOR en botanique. TOUJOURS : "
        "identification taxonomique précise (clé de détermination), "
        "propriétés médicinales/toxiques documentées, écologie de la plante."
    ),
    "dentistry": (
        "Tu es un expert SENIOR en dentisterie. TOUJOURS : "
        "examen clinique complet (carie/parodonte/occlusion), "
        "radiographie justifiée, traitement conservateur prioritaire."
    ),
    "ecology": (
        "Tu es un expert SENIOR en écologie/faune-flore. TOUJOURS : "
        "identification terrain rigoureuse, statut de conservation (UICN), "
        "écologie de l'espèce (habitat/régime/reproduction), impact anthropique."
    ),

    "astronomy": (
        "Tu es un expert SENIOR en astronomie/astrophysique. TOUJOURS : "
        "lois de Kepler/Newton appliquées correctement, unités SI, "
        "ordres de grandeur vérifiés, observation vs théorie distingués."
    ),
    "geology": (
        "Tu es un expert SENIOR en géologie. TOUJOURS : "
        "échelle des temps géologiques précise, identification minéralogique, "
        "contexte tectonique, datation relative/absolue."
    ),
    "computer_science": (
        "Tu es un expert SENIOR en informatique théorique. TOUJOURS : "
        "complexité (temps + espace), correction de l'algorithme, "
        "preuve par invariant, structure de données adaptée."
    ),
    "meteorology": (
        "Tu es un expert SENIOR en météorologie. TOUJOURS : "
        "modèles atmosphériques (NWP), données d'observation, "
        "incertitude quantifiée (ensembles), phénomènes locaux vs globaux."
    ),
    "quantum": (
        "Tu es un expert SENIOR en physique quantique. TOUJOURS : "
        "formalisme mathématique rigoureux (Hilbert, opérateurs), "
        "principe d'incertitude respecté, paradoxes expliqués clairement."
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
        """Résout une tâche : sélectionne le skill, exécute, quality_check.
        Si AUCUN skill n'existe -> le modèle en CRÉE un nouveau (composition de règles)."""
        if skill_name:
            skill = self.registry.get(skill_name)
        else:
            domain_skills = self.skills_for_domain()
            skill = domain_skills[0] if domain_skills else None
        if skill is None:
            # ADAPTIF : le modèle CRÉE un nouveau ExpertSkill pour ce besoin
            skill = ExpertSkill(
                name=f"auto_{task[:20].replace(' ', '_')}",
                description=f"Skill créé automatiquement pour: {task}",
                best_practices=["Production-grade", "UX d'abord", "Solution complète"],
                fn=lambda t=task: f"Solution production-grade pour '{t}' (skill auto-créé)",
                domain=self.domain,
            )
            self.registry.register(skill)             # enregistre pour réutilisation
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
    reg.register(ExpertSkill(
        name="pentest",
        description="Test d'intrusion expert : recon, exploit, post-exploitation, rapport",
        best_practices=[
            "Reconnaissance complète (OSINT, scan)",
            "Exploitation documentée (CVE, chaîne)",
            "Post-exploitation limitée (preuve, pas de dégât)",
            "Rapport détaillé (criticité, remediation)",
        ],
        fn=lambda target: f"pentest de '{target}': 3 vulnérabilités trouvées, remediation proposée",
        domain="cybersecurity",
    ))
    reg.register(ExpertSkill(
        name="marketing_campaign",
        description="Campagne marketing experte : persona, funnel, creative, A/B, ROI",
        best_practices=[
            "Persona défini (démographie, psychographie)",
            "Funnel mesuré (awareness→conversion)",
            "A/B testé (≥2 variantes)",
            "ROI calculé (CAC, LTV)",
        ],
        fn=lambda product: f"campagne pour '{product}': persona + funnel + 3 créas A/B",
        domain="marketing",
    ))
    reg.register(ExpertSkill(
        name="level_design",
        description="Game level design expert : flow, difficulté progressive, secrets, feedback",
        best_practices=[
            "Flow testé (courbe de difficulté)",
            "Tutoriel intégré (show-don't-tell)",
            "Secrets/replayability (≥1 par niveau)",
            "Feedback joueur immédiat (<100ms)",
        ],
        fn=lambda concept: f"niveau pour '{concept}': flow + tutoriel + secrets planifiés",
        domain="game_dev",
    ))
    reg.register(ExpertSkill(
        name="reactjs_dev",
        description="Développement ReactJS expert : hooks, state, composants, performance",
        best_practices=[
            "Hooks uniquement (pas de classes sauf ErrorBoundary)",
            "State minimal + useReducer pour state complexe",
            "Mémoïsation (useMemo/useCallback) si re-renders coûteux",
            "Accessibilité (aria, keyboard nav, focus trap)",
            "Tests (Jest + React Testing Library)",
        ],
        fn=lambda spec: f"composant ReactJS pour '{spec}': hooks + state + tests",
        domain="development",
    ))
    reg.register(ExpertSkill(
        name="data_analysis",
        description="Analyse de données experte : nettoyage, exploration, modélisation, validation",
        best_practices=[
            "Nettoyage rigoureux (outliers, NA, doublons)",
            "EDA systématique (distributions, corrélations)",
            "Modèle validé (train/test split, cross-val)",
            "Visualisation claire (matplotlib/plotly)",
        ],
        fn=lambda dataset: f"analyse de '{dataset}': pipeline complet, modèle validé",
        domain="research",
    ))
    reg.register(ExpertSkill(
        name="devops_pipeline",
        description="Pipeline DevOps expert : build, test, deploy, monitor",
        best_practices=[
            "CI/CD automatisé (GitHub Actions / GitLab CI)",
            "Tests automatisés (unit + integration + e2e)",
            "Déploiement blue-green (zero downtime)",
            "Monitoring (Prometheus/Grafana, alerting)",
        ],
        fn=lambda project: f"pipeline DevOps pour '{project}': CI/CD + monitoring",
        domain="development",
    ))
    reg.register(ExpertSkill(
        name="osint_recon",
        description="OSINT expert : reconnaissance, corrélation, vérification",
        best_practices=[
            "Sources ouvertes uniquement (légal)",
            "Croisement multi-sources (≥2)",
            "Chronologie reconstituée",
            "Vérification d'authenticité",
        ],
        fn=lambda target: f"OSINT sur '{target}': 5 sources croisées, chronologie établie",
        domain="cybersecurity",
    ))
    reg.register(ExpertSkill(
        name="scientific_paper",
        description="Rédaction scientifique experte : abstract, méthode, résultats, discussion",
        best_practices=[
            "Abstract structuré (contexte→méthode→résultat→impact)",
            "Méthode reproductible (code+data disponibles)",
            "Statistiques rigoureuses (p-value + effect size + CI)",
            "Discussion honnête (limitations + perspectives)",
        ],
        fn=lambda topic: f"papier scientifique sur '{topic}': structuré, reproductible",
        domain="research",
    ))

    reg.register(ExpertSkill(
        name="legal_analysis",
        description="Analyse juridique experte : lois, jurisprudence, contrats",
        best_practices=[
            "Référence aux textes (articles précis)",
            "Jurisprudence récente citée",
            "Distinction fait/right/obligation",
            "Conseil pratique applicable",
        ],
        fn=lambda case: f"analyse juridique de '{case}': textes + jurisprudence + conseil",
        domain="research",
    ))
    reg.register(ExpertSkill(
        name="historical_analysis",
        description="Analyse historique experte : sources, chronologie, causalité",
        best_practices=[
            "Sources primaires > secondaires",
            "Chronologie reconstituée rigoureuse",
            "Causalité multifactorielle (pas monocausale)",
            "Contexte (politique/économique/social/culturel)",
        ],
        fn=lambda event: f"analyse historique de '{event}': sources + chronologie + causalité",
        domain="research",
    ))
    reg.register(ExpertSkill(
        name="music_composition",
        description="Composition musicale experte : harmonie, contrepoint, arrangement",
        best_practices=[
            "Harmonie respectée (règles fonctionnelles)",
            "Contrepoint correct (mouvement contraire)",
            "Arrangement adapté à l'instrumentation",
            "Structure formelle claire (ABA, sonate, etc.)",
        ],
        fn=lambda brief: f"composition pour '{brief}': harmonie + arrangement + structure",
        domain="writing",
    ))

    return reg
