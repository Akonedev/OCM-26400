"""Tests adaptateur MCP + harnais d'évaluation (OCM-26400, SOTA benchmarks)."""
from ocm26400.mcp_adapter import (
    McpAdapter, McpTool, default_adapter, adapter_security_audit,
)
from ocm26400.eval_harness import (
    BenchmarkItem, BenchmarkRunner, EvalReport,
    compare_to_baselines, random_baseline, total_abstention,
    load_jsonl, synthetic_aime_demo, run_demo,
)


# ---------------- MCP adapter ----------------

def _adapter():
    return default_adapter()


def test_mcp_manifest_lists_tools():
    a = _adapter()
    names = [t["name"] for t in a.manifest()]
    assert "shell" in names and "web_fetch" in names
    # chaque tool a un inputSchema valide
    for t in a.manifest():
        assert "inputSchema" in t and "type" in t["inputSchema"]


def test_mcp_dispatch_known_tool():
    a = _adapter()
    r = a.dispatch({"name": "shell", "arguments": {"command": "echo abc"}})
    assert r["status"] == "ok"
    assert "abc" in r["result"]


def test_mcp_dispatch_unknown_tool():
    a = _adapter()
    r = a.dispatch({"name": "nope", "arguments": {}})
    assert r["status"] == "error" and r["kind"] == "unknown_tool"


def test_mcp_missing_required_param():
    a = _adapter()
    r = a.dispatch({"name": "shell", "arguments": {}})   # command manquant
    assert r["status"] == "error" and r["kind"] == "invalid_params"


def test_mcp_sandboxed_errors_no_traceback():
    a = _adapter()
    r = a.dispatch({"name": "shell", "arguments": {"command": "exit 1"}})
    # ne doit JAMAIS lever ; status error contrôlé (exit 1 = code retour non-zero)
    assert r["status"] in ("ok", "error")
    assert "Traceback" not in str(r)


def test_mcp_security_audit_fields():
    audit = adapter_security_audit(_adapter())
    assert audit["shell_allowlist"] is True
    assert audit["ssrf_protection"] is True
    assert audit["error_sandboxing"] is True
    assert audit["n_tools"] >= 2


def test_mcp_batch_dispatch():
    a = _adapter()
    calls = [
        {"name": "shell", "arguments": {"command": "echo x"}},
        {"name": "shell", "arguments": {"command": "echo y"}},
    ]
    out = a.batch(calls)
    assert len(out) == 2 and all(o["status"] == "ok" for o in out)


def test_mcp_custom_tool_registration():
    a = McpAdapter()
    a.register(McpTool(
        name="add", description="addition",
        input_schema={"type": "object",
                      "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                      "required": ["a", "b"]},
        handler=lambda args: args["a"] + args["b"],
    ))
    r = a.dispatch({"name": "add", "arguments": {"a": 2, "b": 3}})
    assert r["status"] == "ok" and r["result"] == 5


# ---------------- Eval harness ----------------

def test_harness_runner_accuracy():
    items = synthetic_aime_demo(10)
    import re
    def solver(it):
        m = re.search(r"\(3\*(\d+)\+2\)\s*mod\s*5", it.question)
        if not m:
            return None, True, 1
        return str((3 * int(m.group(1)) + 2) % 5), False, 3
    rep = BenchmarkRunner("aime_demo", solver, params=675_000).run(items)
    assert rep.accuracy == 1.0
    assert rep.coverage == 1.0
    assert rep.abstention_rate == 0.0
    assert rep.mean_depth == 3.0
    assert rep.params == 675_000


def test_harness_abstention_counted():
    items = synthetic_aime_demo(5)
    def solver(it):       # abstient sur la moitié
        return None, True, 0 if items.index(it) % 2 == 0 else False
    # corrige : solver booléen pur -> abstention
    def solver2(it):
        return (None, True, 0) if items.index(it) % 2 == 0 else ("0", False, 1)
    rep = BenchmarkRunner("t", solver2).run(items)
    assert rep.n_abstained == 3        # indices 0,2,4
    assert rep.coverage == 2 / 5


def test_harness_beats_random_signal():
    items = synthetic_aime_demo(10)
    import re
    def solver(it):
        m = re.search(r"\(3\*(\d+)\+2\)\s*mod\s*5", it.question)
        return (str((3 * int(m.group(1)) + 2) % 5), False, 3) if m else (None, True, 1)
    rep = BenchmarkRunner("t", solver).run(items)
    cmp = compare_to_baselines(rep, items)
    assert cmp["value_vs_random"] > 0.05
    assert cmp["verdict"] == "SIGNAL"


def test_harness_total_abstention_is_zero_accuracy():
    items = synthetic_aime_demo(5)
    rep = BenchmarkRunner("t", total_abstention(items)).run(items)
    assert rep.accuracy == 0.0 and rep.coverage == 0.0


def test_harness_save_results_json(tmp_path):
    items = synthetic_aime_demo(3)
    def solver(it):
        return ("0", False, 2)
    rep = BenchmarkRunner("save_demo", solver).run(items)
    path = rep.save(str(tmp_path))
    import json, os
    assert os.path.exists(path)
    assert path.endswith("save_demo_results.json")
    with open(path) as f:
        d = json.load(f)
    assert d["benchmark"] == "save_demo" and d["n_items"] == 3


def test_harness_score_qcm_and_numeric():
    from ocm26400.eval_harness import _score
    assert _score("B", "B", True) is True
    assert _score("b ", "B", True) is True       # insensible casse/espaces
    assert _score("42", 42, False) is True       # numérique
    assert _score("42.0", "42", False) is True
    assert _score(None, "x", False) is False


def test_harness_demo_runs():
    out = run_demo()
    assert out["report"]["accuracy"] == 1.0
    assert out["comparison"]["verdict"] == "SIGNAL"


def test_harness_qcm_choices_loaded():
    items = [
        BenchmarkItem(id="q1", question="?", gold_answer="B",
                      choices=["A", "B", "C", "D"]),
    ]
    assert items[0].choices == ["A", "B", "C", "D"]
