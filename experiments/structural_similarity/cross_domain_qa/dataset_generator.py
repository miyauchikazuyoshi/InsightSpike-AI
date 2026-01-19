"""
Cross-Domain Analogy QA Dataset Generator

Creates paired knowledge graphs from different domains that share
structural patterns, along with questions that require analogical reasoning.

Usage:
    python -m experiments.structural_similarity.cross_domain_qa.dataset_generator
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class QAExample:
    """A single QA example requiring analogical reasoning."""
    id: str
    question: str
    answer: str
    source_domain: str
    target_domain: str
    structure_type: str
    source_fact: str  # The fact in source domain that helps answer
    target_context: str  # Incomplete info about target domain
    difficulty: str  # easy, medium, hard
    requires_analogy: bool = True


@dataclass
class DomainPair:
    """A pair of domains with shared structure."""
    name: str
    source_domain: str
    target_domain: str
    structure_type: str
    source_graph: nx.Graph = field(default_factory=nx.Graph)
    target_graph_complete: nx.Graph = field(default_factory=nx.Graph)
    target_graph_incomplete: nx.Graph = field(default_factory=nx.Graph)
    qa_examples: List[QAExample] = field(default_factory=list)


# =============================================================================
# Domain Pair Generators
# =============================================================================

def create_solar_atom_pair() -> DomainPair:
    """Solar system <-> Atom (hub-spoke pattern)."""
    pair = DomainPair(
        name="solar_atom",
        source_domain="astronomy",
        target_domain="physics",
        structure_type="hub_spoke",
    )

    # Source: Solar system (complete)
    G_source = nx.Graph()
    G_source.add_node("sun", domain="astronomy", role="hub",
                      properties={"mass": "large", "type": "star", "position": "center"})
    planets = [
        ("mercury", {"distance": "close", "size": "small"}),
        ("venus", {"distance": "close", "size": "medium"}),
        ("earth", {"distance": "medium", "size": "medium", "has_life": True}),
        ("mars", {"distance": "medium", "size": "small"}),
        ("jupiter", {"distance": "far", "size": "large"}),
    ]
    for planet, props in planets:
        G_source.add_node(planet, domain="astronomy", role="spoke", properties=props)
        G_source.add_edge("sun", planet, relation="orbits",
                         properties={"force": "gravity", "motion": "elliptical"})
    pair.source_graph = G_source

    # Target: Atom (complete version for ground truth)
    G_target_complete = nx.Graph()
    G_target_complete.add_node("nucleus", domain="physics", role="hub",
                               properties={"mass": "large", "charge": "positive", "position": "center"})
    electrons = [
        ("e1", {"shell": "1", "energy": "low"}),
        ("e2", {"shell": "1", "energy": "low"}),
        ("e3", {"shell": "2", "energy": "medium"}),
        ("e4", {"shell": "2", "energy": "medium"}),
    ]
    for electron, props in electrons:
        G_target_complete.add_node(electron, domain="physics", role="spoke", properties=props)
        G_target_complete.add_edge("nucleus", electron, relation="orbits",
                                   properties={"force": "electromagnetic", "motion": "orbital"})
    pair.target_graph_complete = G_target_complete

    # Target: Atom (incomplete - missing orbital info)
    G_target_incomplete = nx.Graph()
    G_target_incomplete.add_node("nucleus", domain="physics", role="hub",
                                  properties={"mass": "large", "charge": "positive", "position": "center"})
    for electron, props in electrons[:2]:  # Only partial electrons
        G_target_incomplete.add_node(electron, domain="physics", role="spoke", properties=props)
        # Edge exists but motion property is unknown
        G_target_incomplete.add_edge("nucleus", electron, relation="associated_with",
                                      properties={"force": "electromagnetic"})
    pair.target_graph_incomplete = G_target_incomplete

    # QA Examples
    pair.qa_examples = [
        QAExample(
            id="solar_atom_001",
            question="How do electrons move around the nucleus?",
            answer="Electrons orbit around the nucleus in orbital paths",
            source_domain="astronomy",
            target_domain="physics",
            structure_type="hub_spoke",
            source_fact="Planets orbit around the sun in elliptical paths due to gravity",
            target_context="Electrons are associated with the nucleus through electromagnetic force",
            difficulty="easy",
        ),
        QAExample(
            id="solar_atom_002",
            question="What keeps electrons bound to the nucleus?",
            answer="Electromagnetic force keeps electrons bound to the nucleus",
            source_domain="astronomy",
            target_domain="physics",
            structure_type="hub_spoke",
            source_fact="Gravity keeps planets bound to the sun",
            target_context="There is a force between the nucleus and electrons",
            difficulty="easy",
        ),
        QAExample(
            id="solar_atom_003",
            question="Where is most of the atom's mass concentrated?",
            answer="Most of the atom's mass is concentrated in the nucleus at the center",
            source_domain="astronomy",
            target_domain="physics",
            structure_type="hub_spoke",
            source_fact="Most of the solar system's mass is concentrated in the sun at the center",
            target_context="The atom has a nucleus and electrons",
            difficulty="medium",
        ),
        QAExample(
            id="solar_atom_004",
            question="If electrons are like planets, what property would they share in terms of motion?",
            answer="Like planets orbiting the sun, electrons would orbit the nucleus",
            source_domain="astronomy",
            target_domain="physics",
            structure_type="hub_spoke",
            source_fact="Planets orbit around the sun",
            target_context="Electrons exist around the nucleus",
            difficulty="medium",
        ),
        QAExample(
            id="solar_atom_005",
            question="By analogy with the solar system, what can we infer about the relative size of the nucleus compared to electron positions?",
            answer="The nucleus is very small compared to the total size of the atom, similar to how the sun is small compared to the solar system",
            source_domain="astronomy",
            target_domain="physics",
            structure_type="hub_spoke",
            source_fact="The sun is at the center with planets at various distances, making the solar system mostly empty space",
            target_context="The atom has a central nucleus with electrons somewhere around it",
            difficulty="hard",
        ),
    ]

    return pair


def create_company_military_pair() -> DomainPair:
    """Company organization <-> Military (hierarchy pattern)."""
    pair = DomainPair(
        name="company_military",
        source_domain="business",
        target_domain="military",
        structure_type="hierarchy",
    )

    # Source: Company (complete)
    G_source = nx.DiGraph()
    G_source.add_node("ceo", domain="business", role="root", level=0,
                      properties={"authority": "highest", "responsibility": "strategy"})
    G_source.add_node("vp_sales", domain="business", role="middle", level=1,
                      properties={"authority": "high", "responsibility": "sales"})
    G_source.add_node("vp_engineering", domain="business", role="middle", level=1,
                      properties={"authority": "high", "responsibility": "engineering"})
    G_source.add_node("manager_a", domain="business", role="middle", level=2)
    G_source.add_node("manager_b", domain="business", role="middle", level=2)
    G_source.add_node("employee_1", domain="business", role="leaf", level=3)
    G_source.add_node("employee_2", domain="business", role="leaf", level=3)

    G_source.add_edge("ceo", "vp_sales", relation="commands")
    G_source.add_edge("ceo", "vp_engineering", relation="commands")
    G_source.add_edge("vp_sales", "manager_a", relation="commands")
    G_source.add_edge("vp_engineering", "manager_b", relation="commands")
    G_source.add_edge("manager_a", "employee_1", relation="commands")
    G_source.add_edge("manager_b", "employee_2", relation="commands")
    pair.source_graph = G_source

    # Target: Military (complete)
    G_target_complete = nx.DiGraph()
    G_target_complete.add_node("general", domain="military", role="root", level=0,
                               properties={"authority": "highest", "responsibility": "strategy"})
    G_target_complete.add_node("colonel_a", domain="military", role="middle", level=1)
    G_target_complete.add_node("colonel_b", domain="military", role="middle", level=1)
    G_target_complete.add_node("captain_a", domain="military", role="middle", level=2)
    G_target_complete.add_node("captain_b", domain="military", role="middle", level=2)
    G_target_complete.add_node("soldier_1", domain="military", role="leaf", level=3)
    G_target_complete.add_node("soldier_2", domain="military", role="leaf", level=3)

    G_target_complete.add_edge("general", "colonel_a", relation="commands")
    G_target_complete.add_edge("general", "colonel_b", relation="commands")
    G_target_complete.add_edge("colonel_a", "captain_a", relation="commands")
    G_target_complete.add_edge("colonel_b", "captain_b", relation="commands")
    G_target_complete.add_edge("captain_a", "soldier_1", relation="commands")
    G_target_complete.add_edge("captain_b", "soldier_2", relation="commands")
    pair.target_graph_complete = G_target_complete

    # Target: Military (incomplete - missing command structure)
    G_target_incomplete = nx.DiGraph()
    G_target_incomplete.add_node("general", domain="military", role="root", level=0)
    G_target_incomplete.add_node("colonel_a", domain="military", role="unknown", level=1)
    G_target_incomplete.add_node("soldier_1", domain="military", role="leaf", level=3)
    G_target_incomplete.add_edge("general", "colonel_a", relation="related_to")
    pair.target_graph_incomplete = G_target_incomplete

    # QA Examples
    pair.qa_examples = [
        QAExample(
            id="company_military_001",
            question="In a military hierarchy, who does a colonel report to?",
            answer="A colonel reports to a general, similar to how a VP reports to a CEO",
            source_domain="business",
            target_domain="military",
            structure_type="hierarchy",
            source_fact="VPs report directly to the CEO in a company hierarchy",
            target_context="The military has generals, colonels, and soldiers",
            difficulty="easy",
        ),
        QAExample(
            id="company_military_002",
            question="How are orders transmitted in a military structure?",
            answer="Orders flow from top (general) down through intermediate ranks (colonel, captain) to soldiers",
            source_domain="business",
            target_domain="military",
            structure_type="hierarchy",
            source_fact="Instructions flow from CEO through VPs and managers to employees",
            target_context="The military has a chain of command",
            difficulty="medium",
        ),
        QAExample(
            id="company_military_003",
            question="If a company has CEO -> VP -> Manager -> Employee, what is the analogous structure in military?",
            answer="General -> Colonel -> Captain -> Soldier",
            source_domain="business",
            target_domain="military",
            structure_type="hierarchy",
            source_fact="Company hierarchy: CEO at top, then VPs, then Managers, then Employees",
            target_context="Military has ranks including general, colonel, captain, soldier",
            difficulty="medium",
        ),
        QAExample(
            id="company_military_004",
            question="What happens if a captain is removed from the chain of command?",
            answer="Communication between colonel and soldiers is disrupted, similar to removing a manager",
            source_domain="business",
            target_domain="military",
            structure_type="hierarchy",
            source_fact="If a manager is removed, communication between VP and employees breaks",
            target_context="Captains are between colonels and soldiers",
            difficulty="hard",
        ),
        QAExample(
            id="company_military_005",
            question="By analogy with corporate span of control, how many subordinates might a colonel typically have?",
            answer="A colonel might have 2-5 direct reports (captains), similar to a VP having 2-5 managers",
            source_domain="business",
            target_domain="military",
            structure_type="hierarchy",
            source_fact="A VP typically manages 2-5 managers directly",
            target_context="Colonels command captains",
            difficulty="hard",
        ),
    ]

    return pair


def create_blood_river_pair() -> DomainPair:
    """Blood vessels <-> River system (branching pattern)."""
    pair = DomainPair(
        name="blood_river",
        source_domain="biology",
        target_domain="geography",
        structure_type="branching",
    )

    # Source: Blood vessels (complete)
    G_source = nx.DiGraph()
    G_source.add_node("heart", domain="biology", role="source",
                      properties={"function": "pump", "flow_direction": "outward"})
    G_source.add_node("aorta", domain="biology", role="main_trunk",
                      properties={"size": "large", "carries": "oxygenated_blood"})
    G_source.add_node("artery_1", domain="biology", role="branch", properties={"size": "medium"})
    G_source.add_node("artery_2", domain="biology", role="branch", properties={"size": "medium"})
    G_source.add_node("arteriole_1a", domain="biology", role="branch", properties={"size": "small"})
    G_source.add_node("arteriole_1b", domain="biology", role="branch", properties={"size": "small"})
    G_source.add_node("capillary_1", domain="biology", role="terminal", properties={"size": "tiny"})

    G_source.add_edge("heart", "aorta", relation="flows_to")
    G_source.add_edge("aorta", "artery_1", relation="branches_to")
    G_source.add_edge("aorta", "artery_2", relation="branches_to")
    G_source.add_edge("artery_1", "arteriole_1a", relation="branches_to")
    G_source.add_edge("artery_1", "arteriole_1b", relation="branches_to")
    G_source.add_edge("arteriole_1a", "capillary_1", relation="branches_to")
    pair.source_graph = G_source

    # Target: River (complete)
    G_target_complete = nx.DiGraph()
    G_target_complete.add_node("source", domain="geography", role="source",
                               properties={"type": "spring", "flow_direction": "downstream"})
    G_target_complete.add_node("main_river", domain="geography", role="main_trunk",
                               properties={"size": "large", "carries": "water"})
    G_target_complete.add_node("tributary_1", domain="geography", role="branch")
    G_target_complete.add_node("tributary_2", domain="geography", role="branch")
    G_target_complete.add_node("stream_1a", domain="geography", role="branch")
    G_target_complete.add_node("stream_1b", domain="geography", role="branch")
    G_target_complete.add_node("delta_outlet", domain="geography", role="terminal")

    G_target_complete.add_edge("source", "main_river", relation="flows_to")
    G_target_complete.add_edge("main_river", "tributary_1", relation="branches_to")
    G_target_complete.add_edge("main_river", "tributary_2", relation="branches_to")
    G_target_complete.add_edge("tributary_1", "stream_1a", relation="branches_to")
    G_target_complete.add_edge("tributary_1", "stream_1b", relation="branches_to")
    G_target_complete.add_edge("stream_1a", "delta_outlet", relation="flows_to")
    pair.target_graph_complete = G_target_complete

    # Target: River (incomplete)
    G_target_incomplete = nx.DiGraph()
    G_target_incomplete.add_node("source", domain="geography", role="source")
    G_target_incomplete.add_node("main_river", domain="geography", role="main_trunk")
    G_target_incomplete.add_node("tributary_1", domain="geography", role="unknown")
    G_target_incomplete.add_edge("source", "main_river", relation="flows_to")
    pair.target_graph_incomplete = G_target_incomplete

    # QA Examples
    pair.qa_examples = [
        QAExample(
            id="blood_river_001",
            question="How does a river system distribute water across a landscape?",
            answer="A river branches into smaller tributaries and streams, similar to how blood vessels branch into arteries and capillaries",
            source_domain="biology",
            target_domain="geography",
            structure_type="branching",
            source_fact="Blood flows from the heart through the aorta, then branches into arteries, arterioles, and capillaries",
            target_context="Rivers have a main channel and tributaries",
            difficulty="medium",
        ),
        QAExample(
            id="blood_river_002",
            question="What happens to the size of river channels as they branch?",
            answer="River channels get smaller as they branch, just like blood vessels get smaller from aorta to capillaries",
            source_domain="biology",
            target_domain="geography",
            structure_type="branching",
            source_fact="Blood vessels decrease in diameter: aorta (large) -> arteries (medium) -> capillaries (tiny)",
            target_context="Rivers branch into tributaries",
            difficulty="easy",
        ),
        QAExample(
            id="blood_river_003",
            question="If a main river is blocked, what happens to the downstream flow?",
            answer="Downstream branches lose water supply, similar to tissue losing blood if an artery is blocked",
            source_domain="biology",
            target_domain="geography",
            structure_type="branching",
            source_fact="If an artery is blocked, tissues downstream don't receive blood",
            target_context="Rivers supply water to downstream areas",
            difficulty="hard",
        ),
    ]

    return pair


def create_supply_nerve_pair() -> DomainPair:
    """Supply chain <-> Nervous system (chain/signal pattern)."""
    pair = DomainPair(
        name="supply_nerve",
        source_domain="logistics",
        target_domain="biology",
        structure_type="chain",
    )

    # Source: Supply chain
    G_source = nx.DiGraph()
    nodes = [
        ("supplier", {"role": "origin", "function": "produce"}),
        ("manufacturer", {"role": "processor", "function": "transform"}),
        ("distributor", {"role": "relay", "function": "forward"}),
        ("retailer", {"role": "relay", "function": "forward"}),
        ("customer", {"role": "destination", "function": "consume"}),
    ]
    for node, props in nodes:
        G_source.add_node(node, domain="logistics", **props)

    edges = [
        ("supplier", "manufacturer", "supplies"),
        ("manufacturer", "distributor", "ships"),
        ("distributor", "retailer", "delivers"),
        ("retailer", "customer", "sells"),
    ]
    for src, dst, rel in edges:
        G_source.add_edge(src, dst, relation=rel, properties={"delay": "variable"})
    pair.source_graph = G_source

    # Target: Nerve signal (complete)
    G_target_complete = nx.DiGraph()
    nodes = [
        ("stimulus", {"role": "origin", "function": "trigger"}),
        ("sensory_neuron", {"role": "processor", "function": "encode"}),
        ("interneuron", {"role": "relay", "function": "process"}),
        ("motor_neuron", {"role": "relay", "function": "transmit"}),
        ("muscle", {"role": "destination", "function": "respond"}),
    ]
    for node, props in nodes:
        G_target_complete.add_node(node, domain="biology", **props)

    edges = [
        ("stimulus", "sensory_neuron", "activates"),
        ("sensory_neuron", "interneuron", "signals"),
        ("interneuron", "motor_neuron", "signals"),
        ("motor_neuron", "muscle", "triggers"),
    ]
    for src, dst, rel in edges:
        G_target_complete.add_edge(src, dst, relation=rel)
    pair.target_graph_complete = G_target_complete

    # Target: Nerve (incomplete)
    G_target_incomplete = nx.DiGraph()
    G_target_incomplete.add_node("stimulus", domain="biology", role="origin")
    G_target_incomplete.add_node("sensory_neuron", domain="biology", role="unknown")
    G_target_incomplete.add_node("muscle", domain="biology", role="destination")
    G_target_incomplete.add_edge("stimulus", "sensory_neuron", relation="connected")
    pair.target_graph_incomplete = G_target_incomplete

    # QA Examples
    pair.qa_examples = [
        QAExample(
            id="supply_nerve_001",
            question="How does a signal travel from stimulus to muscle response?",
            answer="Signal travels through a chain: stimulus -> sensory neuron -> interneuron -> motor neuron -> muscle",
            source_domain="logistics",
            target_domain="biology",
            structure_type="chain",
            source_fact="Products flow: supplier -> manufacturer -> distributor -> retailer -> customer",
            target_context="Nerves connect stimuli to muscle responses",
            difficulty="medium",
        ),
        QAExample(
            id="supply_nerve_002",
            question="What happens if an interneuron is damaged?",
            answer="The signal chain is broken, preventing the muscle from receiving the signal, like a broken supply chain link",
            source_domain="logistics",
            target_domain="biology",
            structure_type="chain",
            source_fact="If the distributor fails, products don't reach retailers or customers",
            target_context="Interneurons are part of the neural pathway",
            difficulty="hard",
        ),
    ]

    return pair


def create_sns_epidemic_pair() -> DomainPair:
    """Social network <-> Epidemic spread (network diffusion pattern)."""
    pair = DomainPair(
        name="sns_epidemic",
        source_domain="social",
        target_domain="epidemiology",
        structure_type="network",
    )

    # Source: SNS viral spread
    G_source = nx.Graph()
    G_source.add_node("influencer", domain="social", role="hub", followers=10000)
    G_source.add_node("user_a", domain="social", role="node", followers=500)
    G_source.add_node("user_b", domain="social", role="node", followers=300)
    G_source.add_node("user_c", domain="social", role="node", followers=200)
    G_source.add_node("user_d", domain="social", role="leaf", followers=50)
    G_source.add_node("user_e", domain="social", role="leaf", followers=30)

    G_source.add_edge("influencer", "user_a", relation="follows")
    G_source.add_edge("influencer", "user_b", relation="follows")
    G_source.add_edge("user_a", "user_c", relation="follows")
    G_source.add_edge("user_a", "user_d", relation="follows")
    G_source.add_edge("user_b", "user_e", relation="follows")
    G_source.add_edge("user_c", "user_d", relation="follows")  # Creates alternative path
    pair.source_graph = G_source

    # Target: Epidemic (complete)
    G_target_complete = nx.Graph()
    G_target_complete.add_node("patient_zero", domain="epidemiology", role="hub", contacts=50)
    G_target_complete.add_node("infected_a", domain="epidemiology", role="node", contacts=20)
    G_target_complete.add_node("infected_b", domain="epidemiology", role="node", contacts=15)
    G_target_complete.add_node("infected_c", domain="epidemiology", role="node", contacts=10)
    G_target_complete.add_node("infected_d", domain="epidemiology", role="leaf", contacts=5)
    G_target_complete.add_node("infected_e", domain="epidemiology", role="leaf", contacts=3)

    G_target_complete.add_edge("patient_zero", "infected_a", relation="infects")
    G_target_complete.add_edge("patient_zero", "infected_b", relation="infects")
    G_target_complete.add_edge("infected_a", "infected_c", relation="infects")
    G_target_complete.add_edge("infected_a", "infected_d", relation="infects")
    G_target_complete.add_edge("infected_b", "infected_e", relation="infects")
    G_target_complete.add_edge("infected_c", "infected_d", relation="infects")
    pair.target_graph_complete = G_target_complete

    # Target: Epidemic (incomplete)
    G_target_incomplete = nx.Graph()
    G_target_incomplete.add_node("patient_zero", domain="epidemiology", role="hub")
    G_target_incomplete.add_node("infected_a", domain="epidemiology", role="unknown")
    G_target_incomplete.add_edge("patient_zero", "infected_a", relation="contact")
    pair.target_graph_incomplete = G_target_incomplete

    # QA Examples
    pair.qa_examples = [
        QAExample(
            id="sns_epidemic_001",
            question="How does a disease spread through a population?",
            answer="Disease spreads through contact networks, similar to how viral content spreads through social connections",
            source_domain="social",
            target_domain="epidemiology",
            structure_type="network",
            source_fact="Viral content spreads from influencers to followers, then to their followers",
            target_context="Diseases spread from infected to susceptible individuals",
            difficulty="easy",
        ),
        QAExample(
            id="sns_epidemic_002",
            question="Who should be vaccinated first to stop an epidemic?",
            answer="People with many contacts (hubs) should be vaccinated first, like targeting influencers to stop viral spread",
            source_domain="social",
            target_domain="epidemiology",
            structure_type="network",
            source_fact="Blocking or removing influencer accounts can stop viral content spread",
            target_context="Some people have more contacts than others",
            difficulty="hard",
        ),
        QAExample(
            id="sns_epidemic_003",
            question="Why might some infected people spread disease to more others?",
            answer="People with more contacts (super-spreaders) infect more, like influencers reaching more followers",
            source_domain="social",
            target_domain="epidemiology",
            structure_type="network",
            source_fact="Influencers with more followers spread content to more people",
            target_context="Not all infected people spread equally",
            difficulty="medium",
        ),
    ]

    return pair


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_full_dataset() -> Dict[str, Any]:
    """Generate the complete cross-domain QA dataset."""

    pairs = [
        create_solar_atom_pair(),
        create_company_military_pair(),
        create_blood_river_pair(),
        create_supply_nerve_pair(),
        create_sns_epidemic_pair(),
    ]

    dataset = {
        "metadata": {
            "name": "Cross-Domain Analogy QA",
            "version": "1.0",
            "description": "QA dataset requiring analogical reasoning across domains",
            "num_domain_pairs": len(pairs),
            "total_examples": sum(len(p.qa_examples) for p in pairs),
        },
        "domain_pairs": [],
        "examples": [],
    }

    for pair in pairs:
        # Serialize graphs
        pair_data = {
            "name": pair.name,
            "source_domain": pair.source_domain,
            "target_domain": pair.target_domain,
            "structure_type": pair.structure_type,
            "source_graph": {
                "nodes": [
                    {"id": n, **pair.source_graph.nodes[n]}
                    for n in pair.source_graph.nodes()
                ],
                "edges": [
                    {"source": u, "target": v, **pair.source_graph.edges[u, v]}
                    for u, v in pair.source_graph.edges()
                ],
            },
            "target_graph_complete": {
                "nodes": [
                    {"id": n, **pair.target_graph_complete.nodes[n]}
                    for n in pair.target_graph_complete.nodes()
                ],
                "edges": [
                    {"source": u, "target": v, **pair.target_graph_complete.edges[u, v]}
                    for u, v in pair.target_graph_complete.edges()
                ],
            },
            "target_graph_incomplete": {
                "nodes": [
                    {"id": n, **pair.target_graph_incomplete.nodes[n]}
                    for n in pair.target_graph_incomplete.nodes()
                ],
                "edges": [
                    {"source": u, "target": v, **pair.target_graph_incomplete.edges[u, v]}
                    for u, v in pair.target_graph_incomplete.edges()
                ],
            },
            "num_examples": len(pair.qa_examples),
        }
        dataset["domain_pairs"].append(pair_data)

        # Add examples
        for ex in pair.qa_examples:
            dataset["examples"].append(asdict(ex))

    return dataset


def save_dataset(output_dir: Path):
    """Generate and save the dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_full_dataset()

    # Save full dataset
    with open(output_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Save examples only (for easier loading)
    with open(output_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(dataset["examples"], f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "total_examples": len(dataset["examples"]),
        "by_difficulty": {},
        "by_structure": {},
        "by_domain_pair": {},
    }
    for ex in dataset["examples"]:
        diff = ex["difficulty"]
        struct = ex["structure_type"]
        pair_key = f"{ex['source_domain']}_{ex['target_domain']}"

        summary["by_difficulty"][diff] = summary["by_difficulty"].get(diff, 0) + 1
        summary["by_structure"][struct] = summary["by_structure"].get(struct, 0) + 1
        summary["by_domain_pair"][pair_key] = summary["by_domain_pair"].get(pair_key, 0) + 1

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to {output_dir}")
    print(f"Total examples: {summary['total_examples']}")
    print(f"By difficulty: {summary['by_difficulty']}")
    print(f"By structure: {summary['by_structure']}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "data"
    save_dataset(output_dir)
