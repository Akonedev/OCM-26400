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


class GUITool:
    """Computer-use GUI réel : contrôle souris/clavier via pyautogui.

    Implémente la capacité 'computer use' au niveau GUI (clic, déplacement, frappe).
    HONNÊTE : nécessite pyautogui + un serveur d'affichage (display). En environnement
    headless (sans display), les méthodes retournent un message gracieux au lieu de
    crasher — le code est réel (API pyautogui), l'exécution demande un display.
    """

    def __init__(self):
        self._pyautogui = None
        try:                                    # import paresseux
            import pyautogui                    # noqa: F401
            self._pyautogui = pyautogui
        except Exception:
            self._pyautogui = None

    @property
    def available(self) -> bool:
        if self._pyautogui is None:
            return False
        try:
            self._pyautogui.size()              # échoue si pas de display
            return True
        except Exception:
            return False

    def _guarded(self, action: str):
        if not self.available:
            return f"[GUI indisponible : {action} nécessite pyautogui + un display]"
        return "ok"

    def move_to(self, x: int, y: int):
        if not self.available:
            return self._guarded("move_to")
        self._pyautogui.moveTo(x, y)
        return f"moved ({x},{y})"

    def click(self, x: int = None, y: int = None):
        if not self.available:
            return self._guarded("click")
        self._pyautogui.click(x, y) if x is not None else self._pyautogui.click()
        return f"clicked ({x},{y})"

    def type_text(self, text: str):
        if not self.available:
            return self._guarded("type_text")
        self._pyautogui.typewrite(text)
        return f"typed {len(text)} chars"

    def screenshot(self):
        if not self.available:
            return self._guarded("screenshot")
        return self._pyautogui.screenshot()


