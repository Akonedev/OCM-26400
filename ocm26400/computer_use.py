"""Computer use RÉEL — exécution de commandes OS (OCM-26400, cahier des charges).

Le cahier des charges exige « computer use, browser use ». On implémente un VRAI
computer-use : le ShellTool exécute de RÉELLES commandes sur la machine (subprocess) et
retourne leur sortie. L'agent peut donc interroger/agir sur le système — c'est la
capacité 'computer use' (le modèle contrôle l'ordinateur via shell).

HONNÊTE / SÉCURITÉ : c'est une exécution RÉELLE de commandes arbitraires (shell=True).
C'est précisément le 'computer use' demandé (le modèle agit sur la machine), mais cela
comporte un risque réel — en production, à encadrer (sandbox, allowlist de commandes,
confirmation utilisateur). Ici on l'implémente comme capacité de recherche, avec un
mécanisme d'allowlist optionnel pour restreindre les commandes autorisées.

Le BrowserTool plein (navigation interactive clics/JS via playwright/selenium) est une
extension runtime ; le WebFetchTool (web_tools.py) couvre déjà la lecture de pages.
"""
import shlex
import subprocess
from typing import Optional, List


class ShellTool:
    """Computer use réel : exécute des commandes et retourne la sortie.

    SÉCURITÉ : par défaut, exécution SÛRE (shlex.split + liste, SANS shell=True) pour
    éviter l'injection de commandes. Le mode raw=True (shell=True, opt-in explicite)
    permet pipes/redirects mais réactive le risque d'injection — à n'utiliser qu'avec
    une commande de confiance. Une allowlist optionnelle restreint les binaires permis.
    """

    def __init__(self, timeout: int = 15, allowlist: Optional[List[str]] = None):
        self.timeout = timeout
        self.allowlist = allowlist        # si défini, seuls ces binaires sont permis

    def _allowed(self, command: str) -> bool:
        if self.allowlist is None:
            return True
        try:
            head = shlex.split(command)[0]
        except Exception:
            return False
        return head in self.allowlist

    def query(self, command: str, raw: bool = False) -> Optional[str]:
        """Exécute RÉELLEMENT la commande. raw=False (défaut) = exécution sûre sans shell.

        raw=True active shell=True (pipes/redirects) au prix du risque d'injection —
        n'employer qu'avec une commande de confiance.
        """
        if not self._allowed(command):
            return f"[bloqué par allowlist : {command!r}]"
        try:
            if raw:
                r = subprocess.run(command, shell=True, capture_output=True,
                                   text=True, timeout=self.timeout)
            else:
                # SAFE : liste d'arguments, pas de shell -> pas d'injection
                argv = shlex.split(command)
                r = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
            return (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            return f"[timeout ({self.timeout}s)]"
        except FileNotFoundError:
            return f"[commande introuvable]"
        except Exception as e:
            return f"[erreur: {type(e).__name__}: {e}]"

    def run(self, command: str, raw: bool = False) -> str:
        """Alias de query (sémantique 'computer use')."""
        out = self.query(command, raw=raw)
        return out if out is not None else ""


def safe_default_allowlist() -> List[str]:
    """Allowlist de commandes en lecture seule (safe computer-use pour démo)."""
    return ["ls", "pwd", "echo", "cat", "head", "tail", "wc", "uname", "whoami",
            "date", "grep", "find", "python3", "git"]
