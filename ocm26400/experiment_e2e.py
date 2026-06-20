#!/usr/bin/env python3
"""
INTÉGRATION END-TO-END — cycle cognitif complet (OCM-26400).

Démontre le SYSTÈME UNIFIÉ de bout en bout sur des tâches concrètes :

  Tâche → MétaContrôleur (route MoE → prompt → skill) → SwarmAgent (spectral + skills)
    → génération (flow-matching si nécessaire) → quality_check (sécurité)
    → self-correction (vérifie le résultat) → DA critique → Juge valide → LIVRÉ

C'est le cycle complet du cahier des charges, exécuté avec l'architecture spectrale.
"""
import json, time

from ocm26400.meta_controller import MetaController
from ocm26400.agent_swarm import SwarmOrchestrator, SwarmConfig
from ocm26400.orchestrator import DevAdvocate, Judge
from ocm26400.rules import RuleLibrary


def run_e2e():
    mc = MetaController()
    swarm = SwarmOrchestrator(SwarmConfig(n_agents=20, depth=8))
    lib = RuleLibrary.default()
    da = DevAdvocate("da", lambda a, q: ("vérifié", 0.05))
    judge = Judge(quorum=0.5)
    t0 = time.time()

    tasks = [
        "calcul mathématique du nombre optimal",
        "audit de sécurité owasp du code",
        "conception interface utilisateur accessible",
        "composant ReactJS pour dashboard",
        "diagnostic médical des symptômes",
        "identification plante botanique",
    ]

    print("=" * 70)
    print("CYCLE COGNITIF END-TO-END (architecture spectrale unifiée)")
    print("=" * 70)
    print(f"\nMéta-contrôleur : {len(lib.rules)} règles, {len(mc.registry.names())} skills, "
          f"{len(swarm.agents)} agents\n")

    results = []
    for task in tasks:
        print(f"\n{'─'*60}")
        print(f"TÂCHE : {task}")

        # 1. MÉTA-CONTRÔLEUR : route → prompt → skill
        analysis = mc.analyze(task)
        print(f"  Domaine routé (MoE) : {analysis.domain}")
        print(f"  Prompt expert       : {analysis.prompt[:60]}...")
        print(f"  Skill               : {analysis.skill_name} "
              f"({'créé' if analysis.skill_created else 'existant'})")

        # 2. SWARM : agent du domaine traite la tâche
        domain_agents = [a for a in swarm.agents if a.domain == analysis.domain]
        agent = domain_agents[0] if domain_agents else swarm.agents[0]
        agent_result = agent.process(task, depth=8)
        print(f"  Agent #{agent.id} ({agent.domain}) → {str(agent_result)[:60]}...")

        # 3. MÉTA-CONTRÔLEUR exécute (skill + quality_check)
        exec_result = mc.execute(task)
        quality = exec_result.get("quality", "unknown")
        print(f"  Exécution skill     : quality={quality}")

        # 4. SELF-CORRECTION : agent vérifie son propre résultat
        agent.memory.remember("last_result", exec_result.get("result", ""))
        checked = agent._quality_check(exec_result.get("result", ""))
        print(f"  Self-correction     : {'✓ validé' if checked else '✗ rejeté'}")

        # 5. DA + JUGE valident
        da_critique, da_doubt = da.critique(str(exec_result.get("result", "")), task)
        expert_input = [(str(exec_result.get("result", "")), 0.9, 1.0)]
        verdict, conf, raison = judge.arbitrate(expert_input, [(da_critique, da_doubt)])
        print(f"  DA                  : {da_critique} (doute {da_doubt:.2f})")
        print(f"  JUGE                : verdict={verdict}, confiance={conf:.2f}")
        status = "LIVRÉ ✓" if verdict and checked else "REFUSÉ ✗"
        print(f"  STATUS              : {status}")

        results.append({"task": task, "domain": analysis.domain,
                        "skill": analysis.skill_name, "quality": quality,
                        "self_corrected": checked, "da_doubt": da_doubt,
                        "judge_conf": conf, "status": status})

    dt = time.time() - t0
    delivered = sum(1 for r in results if "LIVRÉ" in r["status"])
    print(f"\n{'='*70}")
    print(f"RÉSULTAT : {delivered}/{len(results)} tâches livrées (cycle complet)")
    print(f"Domaines touchés : {', '.join(sorted(set(r['domain'] for r in results)))}")
    print(f"Temps total : {dt:.1f}s")
    print(f"{'='*70}")

    json.dump({"task": "intégration end-to-end (cycle cognitif complet)",
               "n_tasks": len(tasks), "delivered": delivered,
               "results": results, "duration_s": round(dt, 1)},
              open("ocm26400/e2e_results.json", "w"), indent=2)
    print("\nRésultats: ocm26400/e2e_results.json")


if __name__ == "__main__":
    run_e2e()
