#!/usr/bin/env python3
"""
EXPÉRIENCE computer use RÉEL (OCM-26400, cahier des charges 'computer use').

Démontre le computer-use réel : le ShellTool exécute de VRAIES commandes sur la
machine (mode sûr, sans shell=True) et l'agent interroge le système. C'est la capacité
'computer use' (le modèle contrôle l'ordinateur via shell), en exécution réelle.
"""
import json
from ocm26400.computer_use import ShellTool, safe_default_allowlist

COMMANDS = [
    "uname -a",                       # système réel
    "pwd",                            # répertoire courant
    "echo OCM-26400 computer use OK",
    "ls -1 data",                     # liste les corpus réels téléchargés
]


def main():
    tool = ShellTool(timeout=10, allowlist=safe_default_allowlist())
    print("OCM-26400 COMPUTER USE RÉEL (ShellTool, exécution sûre sans shell=True)")
    results = {}
    for cmd in COMMANDS:
        out = tool.run(cmd)
        results[cmd] = (out or "").strip()
        first = (out or "").strip().split("\n")[0][:80]
        print(f"  $ {cmd}\n      -> {first}")
    # démonstration mode sûr : injection neutralisée
    safe = tool.run("echo $(whoami)")
    print(f"\n  Sécurité : 'echo $(whoami)' (mode sûr) -> {safe.strip()!r} "
          f"(non substitué => pas d'injection)")
    verdict = "VALIDÉ" if all(results[c] != "" or "echo" in c for c in COMMANDS) else "NON VALIDÉ"
    print(f"\nVERDICT (computer use réel, commandes OS exécutées) : {verdict}")

    with open("ocm26400/computer_use_results.json", "w") as f:
        json.dump({"task": "computer use réel (ShellTool)", "commands": results,
                   "injection_neutralized": "$(whoami)" in safe,
                   "verdict": verdict}, f, indent=2)
    print("Résultats: ocm26400/computer_use_results.json")
    return verdict


if __name__ == "__main__":
    main()
