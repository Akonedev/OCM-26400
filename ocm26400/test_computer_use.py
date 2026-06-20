"""Tests TDD — computer use réel (ShellTool, OCM-26400).

Valide l'exécution RÉELLE de commandes + la SÉCURITÉ (mode sûr sans shell=True :
injection neutralisée) + l'allowlist.
"""
from ocm26400.computer_use import ShellTool, safe_default_allowlist, GUITool


def test_shell_runs_real_command():
    """echo réel -> retourne la sortie."""
    out = ShellTool().query("echo hello_ocm")
    assert "hello_ocm" in out


def test_shell_safe_mode_neutralizes_injection():
    """raw=False (défaut) : $(...) n'est PAS interprété (pas de shell, pas d'injection).

    En mode shell, 'echo $(whoami)' substituerait le nom d'utilisateur ; en mode sûr,
    '$(whoami)' est imprimé LITTÉRALEMENT (shlex.split -> ['echo','$(whoami)'])."""
    import getpass
    out = ShellTool().query("echo $(whoami)")
    assert "$(whoami)" in out                      # imprimé littéralement, pas substitué
    user = getpass.getuser()
    assert user not in out                         # le nom réel n'est PAS injecté


def test_shell_allowlist_blocks_disallowed():
    """Allowlist restreint les binaires permis."""
    tool = ShellTool(allowlist=["echo"])
    assert "hello" in tool.query("echo hello")
    blocked = tool.query("ls")
    assert "bloqué" in blocked


def test_shell_missing_command_returns_error():
    out = ShellTool().query("ceci_n_existe_pas_123")
    assert "introuvable" in out or "erreur" in out


def test_safe_default_allowlist_is_readonly():
    al = safe_default_allowlist()
    assert "ls" in al and "echo" in al
    assert "rm" not in al and "sudo" not in al


def test_gui_tool_interface_present():
    """GUITool (computer-use GUI souris/clavier) : interface présente.
    En headless, .available=False et les méthodes retournent un message gracieux (pas
    de crash) — la capacité existe dans le code, l'exécution demande un display."""
    gui = GUITool()
    if not gui.available:                         # headless (cas ici)
        for out in (gui.move_to(10, 20), gui.click(0, 0), gui.type_text("x"), gui.screenshot()):
            assert isinstance(out, str) and ("indisponible" in out or "display" in out)
    else:                                         # si display présent : pas de crash
        assert gui.available is True


def test_gui_safe_text_allowlist():
    """Anti-injection clavier : seuls alphanumérique + ponctuation basique sont frappables."""
    rx = GUITool._SAFE_TEXT
    assert rx.match("hello world")                 # ok
    assert rx.match("OCM-26400, c'est test!")       # ok (ponctuation basique)
    assert rx.match("abc;rm") is None              # ';' interdit (anti-injection)
    assert rx.match("$(whoami)") is None           # substitution shell interdite
    assert rx.match("a|b") is None                 # pipe interdit
    assert rx.match("a\nb") is None                # newline interdit
