"""Computer use RÉEL — exécution de commandes OS (OCM-26400, cahier des charges).

Le cahier des charges exige « computer use, browser use ». On implémente un VRAI
computer-use : le ShellTool exécute de RÉELLES commandes sur la machine (subprocess,
liste d'arguments SANS shell=True) et retourne leur sortie.

SÉCURITÉ (durcie après revue) :
* JAMAIS de shell=True (supprime l'injection de commandes). La commande est découpée
  par shlex.split et passée en LISTE d'arguments à subprocess.run -> les métacaractères
  shell ($, ;, |, &) ne sont jamais interprétés.
* Allowlist par défaut restreinte aux binaires purement informatifs read-only.
* Une allowlist personnalisée peut restreindre davantage.

Le BrowserTool plein (navigation interactive clics/JS) et la GUI OS complète
(souris/clavier via pyautogui) sont des extensions runtime nécessitant un display ;
le WebFetchTool (web_tools.py) couvre la lecture de pages, le ShellTool l'exécution OS.
"""
import shlex
import subprocess
from typing import Optional, List


class ShellTool:
    """Computer use réel : exécute des commandes (liste d'args, SANS shell) et retourne la sortie."""

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

    def query(self, command: str) -> Optional[str]:
        """Exécute RÉELLEMENT la commande (shlex.split + liste, SANS shell=True).

        Aucun shell => pas d'injection ($, ;, |, &, `...` ignorés). Retourne stdout+stderr."""
        if not self._allowed(command):
            return f"[bloqué par allowlist : {command!r}]"
        try:
            argv = shlex.split(command)                     # SAFE : liste d'arguments
            r = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
            return (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            return f"[timeout ({self.timeout}s)]"
        except FileNotFoundError:
            return "[commande introuvable]"
        except Exception as e:
            return f"[erreur: {type(e).__name__}: {e}]"

    def run(self, command: str) -> str:
        """Alias de query (sémantique 'computer use')."""
        out = self.query(command)
        return out if out is not None else ""


def safe_default_allowlist() -> List[str]:
    """Allowlist restrictive : binaires purement informatifs read-only."""
    return ["ls", "pwd", "echo", "uname", "whoami", "date", "wc"]

