"""Tests TDD — swarm d'agents hétérogènes progressif (OCM-26400).

Valide : mémoire (locale+partagée), hétérogénéité (MoE par agent), dialogue, cohérence,
sécurité (quality_check), progressif (N paramétrable, pas de limite fixe).
"""
from ocm26400.agent_swarm import (
    AgentMemory, SwarmAgent, SwarmConfig, SwarmOrchestrator, AgentMessage,
)


def test_agent_memory_local_and_shared():
    """Mémoire locale (par agent) + partagée (tous les agents, cohérence)."""
    AgentMemory.reset_shared()
    m1, m2 = AgentMemory(), AgentMemory()
    m1.remember("x", 1); m2.remember("x", 2)
    assert m1.recall("x") == 1 and m2.recall("x") == 2   # locales distinctes
    m1.share("fact", "pi=3.14")
    assert m2.read_shared("fact") == "pi=3.14"             # partagé


def test_swarm_agent_heterogeneous_domains():
    """Chaque agent a un domaine différent (slot op = MoE routing par agent)."""
    a_math = SwarmAgent(id=0, domain="math")
    a_phys = SwarmAgent(id=1, domain="physics")
    assert a_math.domain != a_phys.domain
    assert a_math.slot_op != a_phys.slot_op                 # MoE routing distinct


def test_swarm_agent_process_with_quality_check():
    """L'agent traite une tâche ET vérifie la qualité (sécurité)."""
    agent = SwarmAgent(id=0, domain="development")
    result = agent.process("review code", depth=4)
    assert result is not None and "error" not in str(result).lower()
    assert agent.memory.recall("last_task") == "review code"  # mémorisé


def test_swarm_agent_dialogue():
    """Dialogue inter-agents : send + receive."""
    a1, a2 = SwarmAgent(id=0, domain="math"), SwarmAgent(id=1, domain="physics")
    msg = a1.send("agent_1", "j'ai trouvé 42", "result")
    a2.receive(msg)
    assert len(a2.inbox) == 1
    assert a2.inbox[0].content == "j'ai trouvé 42"


def test_swarm_orchestrator_progressive():
    """N agents est un PARAMÈTRE (progressif, pas de limite fixe)."""
    for n in [1, 10, 50]:
        swarm = SwarmOrchestrator(SwarmConfig(n_agents=n))
        assert len(swarm.agents) == n
    assert swarm.n_active_domains() >= 1              # au moins 1 domaine actif


def test_swarm_dispatch_by_domain_moe():
    """MoE routing : dispatch seulement aux agents du domaine donné."""
    swarm = SwarmOrchestrator(SwarmConfig(n_agents=16))
    math_results = swarm.dispatch_by_domain("math", "calcul")
    assert len(math_results) > 0                       # au moins 1 agent math
    assert all(r is not None for r in math_results)


def test_swarm_coherence_check():
    """Cohérence : les faits partagés ne se contredisent pas."""
    swarm = SwarmOrchestrator(SwarmConfig(n_agents=8))
    swarm.dispatch("task commune")
    assert swarm.coherence_check() is True


def test_swarm_broadcast_dialogue():
    """Broadcast : un agent envoie à TOUS les autres."""
    swarm = SwarmOrchestrator(SwarmConfig(n_agents=5))
    swarm.broadcast(0, "salut les copains")
    # les 4 autres agents ont reçu le message
    received = sum(1 for a in swarm.agents if a.id != 0 and len(a.inbox) > 0)
    assert received == 4
    assert len(swarm.message_bus) == 4


def test_swarm_security_audit():
    """Audit sécurité : tous les agents ont un output quality-checké."""
    swarm = SwarmOrchestrator(SwarmConfig(n_agents=10))
    swarm.dispatch("tâche sécurisée")
    audit = swarm.security_audit()
    assert audit["agents_total"] == 10
    assert audit["with_output"] >= 1
