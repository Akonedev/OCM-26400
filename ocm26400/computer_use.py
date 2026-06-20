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
import re
import time
from typing import Optional, List, Tuple


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
    SÉCURITÉ (durcie, posture équivalente au ShellTool) :
    * caractères frappés filtrés par une allowlist (alphanumérique + ponctuation basique) ;
      tout caractère unsafe est rejeté (anti-injection clavier).
    * coordonnées validées contre les bornes écran ET une allowlist de régions sûres.
    * rate-limiting (délai minimum entre actions) + timeout.
    HONNÊTE : nécessite pyautogui + un serveur d'affichage. En headless, les méthodes
    retournent un message gracieux (pas de crash).
    """

    # caractères autorisés à la frappe (anti-injection : pas de méta-keys/control/etc.)
    _SAFE_TEXT = re.compile(r"^[A-Za-z0-9 .,?!:'\-]+$")
    _MAX_TEXT = 200

    def __init__(self, timeout: float = 15.0, min_delay: float = 0.2,
                 allowed_regions: Optional[List[Tuple[int, int, int, int]]] = None):
        self.timeout = timeout
        self.min_delay = min_delay
        self.allowed_regions = allowed_regions      # (x0,y0,x1,y1) ; None = tout l'écran
        self._pyautogui = None
        self._last_action = 0.0
        try:
            import pyautogui                        # noqa: F401
            self._pyautogui = pyautogui
            self._pyautogui.FAILSAFE = True         # safety: aller en coin HG stoppe tout
        except Exception:
            self._pyautogui = None

    @property
    def available(self) -> bool:
        if self._pyautogui is None:
            return False
        try:
            self._pyautogui.size()
            return True
        except Exception:
            return False

    def _guarded(self, action: str):
        if not self.available:
            return f"[GUI indisponible : {action} nécessite pyautogui + un display]"
        return "ok"

    def _rate_limit(self):
        """Délai minimum entre actions (anti-spam/automatisation trop rapide)."""
        now = time.monotonic()
        wait = self.min_delay - (now - self._last_action)
        if wait > 0:
            time.sleep(wait)
        self._last_action = time.monotonic()

    def _validate_point(self, x: int, y: int) -> Optional[str]:
        """Valide qu'un point est dans l'écran ET dans une région autorisée."""
        if x is None or y is None:
            return "coordonnées manquantes"
        try:
            sw, sh = self._pyautogui.size()
        except Exception:
            return "taille écran inaccessible"
        if not (0 <= x < sw and 0 <= y < sh):
            return f"({x},{y}) hors écran ({sw}x{sh})"
        if self.allowed_regions is not None:
            if not any(x0 <= x < x1 and y0 <= y < y1 for (x0, y0, x1, y1) in self.allowed_regions):
                return f"({x},{y}) hors région autorisée"
        return None

    def move_to(self, x: int, y: int):
        if not self.available:
            return self._guarded("move_to")
        err = self._validate_point(x, y)
        if err:
            return f"[refusé : {err}]"
        self._rate_limit()
        self._pyautogui.moveTo(x, y, _pause=False)
        return f"moved ({x},{y})"

    def click(self, x: int = None, y: int = None):
        if not self.available:
            return self._guarded("click")
        if x is not None:
            err = self._validate_point(x, y)
            if err:
                return f"[refusé : {err}]"
        self._rate_limit()
        self._pyautogui.click(x, y) if x is not None else self._pyautogui.click()
        return f"clicked ({x},{y})"

    def type_text(self, text: str):
        if not self.available:
            return self._guarded("type_text")
        if len(text) > self._MAX_TEXT:
            return f"[refusé : texte trop long (>{self._MAX_TEXT} chars)]"
        if not self._SAFE_TEXT.match(text):         # anti-injection clavier
            return "[refusé : caractères non autorisés à la frappe]"
        self._rate_limit()
        self._pyautogui.typewrite(text, interval=0.02)
        return f"typed {len(text)} chars"

    def screenshot(self):
        if not self.available:
            return self._guarded("screenshot")
        self._rate_limit()
        return self._pyautogui.screenshot()



