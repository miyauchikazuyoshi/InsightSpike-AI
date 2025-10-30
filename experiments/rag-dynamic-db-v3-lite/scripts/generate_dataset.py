#!/usr/bin/env python3
"""Generate richer synthetic RAG datasets for v3-lite experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

DocRef = Union[str, Tuple[str, str]]


def _pref(domain: str, doc_id: str) -> Tuple[str, str]:
    return (domain, doc_id)


DOMAINS: Dict[str, Dict[str, object]] = {
    "solar": {
        "episodes": {
            "sun_core": {
                "context": "Sun-Earth system at dawn over coastal waters",
                "operation": "Emit continuous visible-light photons",
                "affordance": "Photon flux reaches shallow water where algae reside",
                "salience": "High irradiance immediately after sunrise",
                "outcome": "Primary producers receive the energy needed to photosynthesise",
                "goal": "Provide an energy source for biospheric oxygen production",
                "text": "The Sun bathes Earth in photons, delivering the energy that photosynthetic life uses.",
            },
            "photosynthesis": {
                "context": "Leaf and algal cell chloroplasts",
                "operation": "Capture light energy via chlorophyll",
                "affordance": "Pigments convert light to chemical energy",
                "salience": "Efficient photon capture under adequate sunlight",
                "outcome": "Glucose and oxygen are produced from carbon dioxide and water",
                "goal": "Feed ecosystems and release oxygen to surrounding environments",
                "text": "Green plants and algae capture sunlight during photosynthesis, producing oxygen as a by-product.",
            },
            "oxygen_cycle": {
                "context": "Atmosphere and surface oceans near coasts",
                "operation": "Release oxygen into air and dissolve it into water",
                "affordance": "Mixing allows oxygen to contact gases and liquids",
                "salience": "Concentration rises near active photosynthetic zones",
                "outcome": "Available oxygen accumulates for animals and microbes",
                "goal": "Sustain aerobic life by maintaining oxygen reservoirs",
                "text": "Oxygen released by photosynthesis accumulates in the atmosphere and dissolves into surface waters.",
            },
            "marine_algae": {
                "context": "Shallow tidal pools during sunny hours",
                "operation": "Absorb sunlight using photosynthetic pigments",
                "affordance": "Clear water allows light penetration to algal mats",
                "salience": "Oxygen bubbles form as photosynthesis accelerates",
                "outcome": "Dissolved oxygen rises locally in the pools",
                "goal": "Provide habitats with oxygen for marine animals",
                "text": "Coastal algae perform intense photosynthesis when sunlight penetrates shallow water.",
            },
            "solar_flares": {
                "context": "Solar corona during magnetic reconnection events",
                "operation": "Eject energetic charged particles",
                "affordance": "Magnetic field lines accelerate plasma outward",
                "salience": "Produces bursts that can disrupt satellites",
                "outcome": "Causes geomagnetic storms rather than supporting biology",
                "goal": "Illustrate solar weather phenomena (distractor)",
                "text": "Solar flares eject charged particles that can disrupt satellite electronics.",
            },
            "sunspots": {
                "context": "Solar photosphere with intense magnetic activity",
                "operation": "Suppress convective heat flow temporarily",
                "affordance": "Magnetic fields create cooler dark patches",
                "salience": "Visible as dark spots on the Sun's surface",
                "outcome": "Affects solar irradiance only slightly",
                "goal": "Provide astronomical detail unrelated to oxygen flow (distractor)",
                "text": "Sunspots are temporary dark regions caused by strong magnetic fields on the Sun's surface.",
            },
        },
        "scenarios": [
            {
                "question": "Trace how solar energy becomes the oxygen we breathe.",
                "answer": "Solar photons drive photosynthesis in plants and algae, generating the oxygen that builds up in Earth's air.",
                "supports": ["sun_core", "photosynthesis", "oxygen_cycle"],
                "distractors": ["solar_flares"],
            },
            {
                "question": "Explain how sunlight ultimately sustains atmospheric oxygen levels.",
                "answer": "Sunlight powers photosynthesis; the resulting oxygen feeds the atmosphere and oceans.",
                "supports": ["sun_core", "photosynthesis", "oxygen_cycle"],
                "distractors": ["sunspots"],
            },
            {
                "question": "How does coastal sunlight help marine animals obtain oxygen?",
                "answer": "When sunlight reaches coastal waters, marine algae photosynthesise, releasing oxygen that dissolves for fish to use.",
                "supports": ["sun_core", "marine_algae", "oxygen_cycle"],
                "distractors": ["solar_flares"],
            },
        ],
    },
    "moon_tides": {
        "episodes": {
            "moon_gravity": {
                "context": "Gravitational interaction between the Moon and Earth's oceans",
                "operation": "Pull ocean water into two primary bulges",
                "affordance": "Differential gravity stretches water on near and far sides",
                "salience": "Creates predictable tidal bulges aligned with the Moon",
                "outcome": "Generates potential energy that drives tidal motion",
                "goal": "Explain the force responsible for tides",
                "text": "The Moon exerts differential gravitational forces that pull Earth's oceans into bulges.",
            },
            "tide_cycle": {
                "context": "Coastlines rotating through tidal bulges",
                "operation": "Carry each shoreline through bulges twice daily",
                "affordance": "Earth's rotation brings regions under tidal forces",
                "salience": "Produces two high and two low tides per day",
                "outcome": "Alternating water levels along the shore",
                "goal": "Describe the temporal rhythm of tides",
                "text": "As Earth rotates, coastlines pass through tidal bulges, producing regular high and low tides.",
            },
            "tide_mixing": {
                "context": "Shallow estuaries and tidal flats",
                "operation": "Churn water and distribute dissolved gases",
                "affordance": "Flowing currents remix stratified layers",
                "salience": "Maintains oxygen availability for marine organisms",
                "outcome": "Fresh oxygen reaches tidal habitats",
                "goal": "Highlight ecological effects of tides",
                "text": "Tidal currents continuously mix coastal waters, distributing dissolved gases like oxygen.",
            },
            "earth_rotation": {
                "context": "Earth's daily rotation relative to Moon-fixed bulges",
                "operation": "Spin coastlines through gravitational alignment",
                "affordance": "Rotation sweeps each region under the bulges",
                "salience": "Explains the twice-daily tidal cycle",
                "outcome": "Regular high/low tide sequence at a location",
                "goal": "Connect Earth's rotation to tidal frequency",
                "text": "Earth's rotation completes a full turn roughly every 24 hours, sweeping regions through tidal bulges.",
            },
            "moon_dust": {
                "context": "Regolith on the lunar surface",
                "operation": "Form a powdery layer from meteorite impacts",
                "affordance": "Records footprints of astronauts",
                "salience": "No influence on Earth's oceans",
                "outcome": "Irrelevant to tidal oxygen transport",
                "goal": "Serve as an off-topic astronomical detail (distractor)",
                "text": "Fine regolith covers the Moon, forming the powdery surface seen in photographs.",
            },
        },
        "scenarios": [
            {
                "question": "Why do coastal areas experience two high tides each day?",
                "answer": "The Moon's gravity creates two bulges in Earth's oceans, and Earth's rotation carries each coast through them twice per day.",
                "supports": ["moon_gravity", "tide_cycle", "earth_rotation"],
                "distractors": ["moon_dust"],
            },
            {
                "question": "Describe how lunar gravity keeps tidal mixing active.",
                "answer": "The Moon pulls water into bulges; as Earth rotates, those bulges shift, generating currents that mix shoreline waters.",
                "supports": ["moon_gravity", "tide_cycle", "tide_mixing"],
                "distractors": ["moon_dust"],
            },
        ],
    },
    "cell_energy": {
        "episodes": {
            "mitochondria_intro": {
                "context": "Eukaryotic cell cytoplasm containing mitochondria",
                "operation": "House enzymes for aerobic respiration",
                "affordance": "Inner membranes create surface area for ATP synthesis",
                "salience": "Mitochondria require oxygen to function efficiently",
                "outcome": "Cells gain energy from nutrient oxidation",
                "goal": "Explain why mitochondria provide power",
                "text": "Mitochondria are double-membraned organelles that house the machinery for aerobic respiration.",
            },
            "atp_role": {
                "context": "Cellular metabolic pathways",
                "operation": "Use ATP to drive energy-demanding reactions",
                "affordance": "ATP hydrolysis releases manageable energy packets",
                "salience": "Immediate energy supply for biosynthesis and transport",
                "outcome": "Cell functions proceed when ATP is abundant",
                "goal": "Make clear why ATP is the energy currency",
                "text": "Adenosine triphosphate (ATP) provides the immediate energy that powers cellular reactions.",
            },
            "respiration_chain": {
                "context": "Mitochondrial inner membrane electron transport chain",
                "operation": "Pass electrons to oxygen and pump protons",
                "affordance": "Proton gradient drives ATP synthase",
                "salience": "Consumes oxygen to produce large quantities of ATP",
                "outcome": "Cellular energy yield increases dramatically",
                "goal": "Link oxygen use with ATP output",
                "text": "Electron transport chains in mitochondria use oxygen to generate ATP efficiently.",
            },
            "cell_distractor": {
                "context": "Ribosomes on rough ER translating mRNA",
                "operation": "Link amino acids to build proteins",
                "affordance": "Ribosomes catalyse peptide bond formation",
                "salience": "Does not directly involve oxygen-driven ATP synthesis",
                "outcome": "Protein chains elongate independent of mitochondrial function",
                "goal": "Provide unrelated cellular detail (distractor)",
                "text": "Ribosomes build proteins by linking amino acids on the rough endoplasmic reticulum.",
            },
        },
        "scenarios": [
            {
                "question": "Why can mitochondria be described as a cell's power station?",
                "answer": "They oxidise nutrients through respiration, generating ATP that fuels the cell's activities.",
                "supports": ["mitochondria_intro", "respiration_chain", "atp_role"],
                "distractors": ["cell_distractor"],
            },
            {
                "question": "How does oxygen participation in mitochondria support cellular work?",
                "answer": "Oxygen accepts electrons in mitochondrial respiration, allowing ATP production that powers cellular tasks.",
                "supports": ["mitochondria_intro", "respiration_chain", "atp_role"],
                "distractors": ["cell_distractor"],
            },
        ],
    },
    "gps_navigation": {
        "episodes": {
            "gps_satellites": {
                "context": "Medium Earth orbit constellation of GPS satellites",
                "operation": "Broadcast time-stamped navigation signals",
                "affordance": "Signals travel at light speed to receivers worldwide",
                "salience": "Continuous coverage ensures availability",
                "outcome": "Receivers can collect timing data from multiple satellites",
                "goal": "Provide raw data for positioning",
                "text": "GPS satellites orbit Earth and continuously broadcast nanosecond-accurate timing signals.",
            },
            "gps_triangulation": {
                "context": "GPS receiver processing multiple satellite signals",
                "operation": "Compare travel times to solve for position",
                "affordance": "Signals arriving at different times enable trilateration",
                "salience": "Needs four satellites to solve position and clock bias",
                "outcome": "Device computes latitude, longitude, and altitude",
                "goal": "Explain how GPS receivers determine location",
                "text": "Receivers trilaterate their position by comparing travel times from multiple satellites.",
            },
            "gps_clocks": {
                "context": "Atomic clocks aboard GPS satellites",
                "operation": "Maintain precise timing standards",
                "affordance": "Stable clocks keep navigation signals synchronised",
                "salience": "Relativistic corrections keep errors small",
                "outcome": "Receivers trust the timestamps for trilateration",
                "goal": "Highlight importance of precise timing",
                "text": "On-board atomic clocks give each GPS satellite a stable frequency reference.",
            },
            "gps_distractor": {
                "context": "Weather radar monitoring the atmosphere",
                "operation": "Emit pulses that scatter off precipitation",
                "affordance": "Detects rainfall intensity but unrelated to positioning",
                "salience": "Different frequency bands and purposes",
                "outcome": "Provides meteorological data, not navigation",
                "goal": "Serve as off-topic signal technology (distractor)",
                "text": "Weather radar sweeps the atmosphere using pulses that bounce off precipitation.",
            },
        },
        "scenarios": [
            {
                "question": "How does your phone compute its GPS position?",
                "answer": "It measures the arrival time of signals from several satellites and triangulates its location.",
                "supports": ["gps_satellites", "gps_triangulation", "gps_clocks"],
                "distractors": ["gps_distractor"],
            },
            {
                "question": "Explain the timing chain that enables GPS navigation.",
                "answer": "Atomic clocks on satellites timestamp signals; receivers compare the times to locate themselves.",
                "supports": ["gps_satellites", "gps_clocks", "gps_triangulation"],
                "distractors": ["gps_distractor"],
            },
        ],
    },
    "climate_carbon": {
        "episodes": {
            "carbon_sink": {
                "context": "Mature forests and soils absorbing atmospheric CO₂",
                "operation": "Fix carbon into biomass through photosynthesis",
                "affordance": "Dense vegetation draws down carbon dioxide",
                "salience": "Key long-term buffer against greenhouse gases",
                "outcome": "Atmospheric CO₂ growth slows",
                "goal": "Explain forests as carbon sinks",
                "text": "Forests act as carbon sinks by absorbing CO₂ during photosynthesis.",
            },
            "deforestation": {
                "context": "Conversion of forest to agricultural land",
                "operation": "Remove trees and disturb carbon-rich soils",
                "affordance": "Stored carbon is released when biomass is burned or decays",
                "salience": "Reduces future carbon uptake capability",
                "outcome": "Atmospheric CO₂ levels rise faster",
                "goal": "Show climate impact of deforestation",
                "text": "Clearing forests releases stored carbon and decreases future carbon uptake.",
            },
            "reforestation": {
                "context": "Tree planting projects in cleared landscapes",
                "operation": "Establish new vegetation to absorb CO₂",
                "affordance": "Young forests sequester carbon as they grow",
                "salience": "Part of climate mitigation strategies",
                "outcome": "Additional carbon is locked into biomass",
                "goal": "Explain benefits of reforestation",
                "text": "Planting trees increases carbon sequestration and can moderate greenhouse warming.",
            },
            "policy_distractor": {
                "context": "Market mechanisms for emissions trading",
                "operation": "Allow firms to offset emissions via credits",
                "affordance": "Financial incentives rather than ecological processes",
                "salience": "Focuses on policy instruments, not biophysical cycles",
                "outcome": "Does not directly describe oxygen or carbon pathways",
                "goal": "Provide unrelated policy detail (distractor)",
                "text": "Carbon credits allow companies to offset emissions by supporting reduction projects.",
            },
        },
        "scenarios": [
            {
                "question": "How does deforestation accelerate atmospheric warming?",
                "answer": "Losing forests reduces CO₂ uptake while releasing stored carbon, strengthening the greenhouse effect.",
                "supports": ["carbon_sink", "deforestation"],
                "distractors": ["policy_distractor"],
            },
            {
                "question": "Why do climate plans emphasise reforestation?",
                "answer": "New forests sequester carbon and counterbalance emissions that would otherwise accumulate.",
                "supports": ["reforestation", "carbon_sink"],
                "distractors": ["policy_distractor"],
            },
        ],
    },
    "internet_routing": {
        "episodes": {
            "router_layer": {
                "context": "Network layer of the OSI model",
                "operation": "Inspect packet headers and forward based on IP information",
                "affordance": "Routers understand layer-three addressing",
                "salience": "Key mechanism for inter-network connectivity",
                "outcome": "Packets traverse multiple networks en route to destinations",
                "goal": "Explain router responsibilities",
                "text": "Routers operate at layer three of the OSI model, forwarding packets based on IP headers.",
            },
            "routing_tables": {
                "context": "Routing table entries inside routers",
                "operation": "Record next-hop decisions for prefixes",
                "affordance": "Routers choose efficient paths using these tables",
                "salience": "Dynamic updates keep routes current",
                "outcome": "Traffic is steered along viable paths",
                "goal": "Show how routers decide forwarding paths",
                "text": "Routing tables store next-hop decisions for various network prefixes, steering traffic efficiently.",
            },
            "packet_flow": {
                "context": "Packets travelling from source host to destination server",
                "operation": "Carry addressing information and payload",
                "affordance": "Routers read addresses to move packets along",
                "salience": "End-to-end delivery depends on correct forwarding",
                "outcome": "Data reaches the intended server",
                "goal": "Illustrate packet-level forwarding",
                "text": "Packets contain source and destination addresses that routers inspect to guide them across networks.",
            },
            "routing_distractor": {
                "context": "Local Wi-Fi access point converting wired signals",
                "operation": "Bridge wired and wireless segments on the same LAN",
                "affordance": "Enables wireless clients to connect locally",
                "salience": "Does not perform long-haul routing decisions",
                "outcome": "Provides local connectivity but not inter-network routing",
                "goal": "Serve as unrelated network detail (distractor)",
                "text": "Wi-Fi access points convert wired signals into local wireless transmissions.",
            },
        },
        "scenarios": [
            {
                "question": "Describe how routers escort web traffic across the internet.",
                "answer": "Routers read packet addresses and consult routing tables to forward traffic along viable next hops.",
                "supports": ["router_layer", "routing_tables", "packet_flow"],
                "distractors": ["routing_distractor"],
            },
            {
                "question": "What happens to a packet leaving your laptop for a remote server?",
                "answer": "Routers examine its headers and relay it through successive networks until it reaches the destination prefix.",
                "supports": ["packet_flow", "routing_tables", "router_layer"],
                "distractors": ["routing_distractor"],
            },
        ],
    },
}


CROSS_DOMAIN_SCENARIOS: List[Dict[str, object]] = [
    {
        "name": "sun_tide_oxygen",
        "question": "How does sunlight end up boosting the oxygen mixed into tidal pools?",
        "answer": (
            "Sunlight powers coastal algae to photosynthesise oxygen, and lunar-driven tides mix that oxygenated water along the shore."
        ),
        "supports": [
            _pref("solar", "marine_algae"),
            _pref("solar", "oxygen_cycle"),
            _pref("moon_tides", "tide_mixing"),
        ],
        "distractors": [
            _pref("moon_tides", "moon_dust"),
            _pref("internet_routing", "routing_distractor"),
        ],
    },
    {
        "name": "forest_ocean_link",
        "question": "Outline the chain connecting forest loss to oxygen available for marine animals.",
        "answer": (
            "Deforestation reduces oxygen-producing vegetation on land; less photosynthetic output means less oxygen dissolving into waters where tides normally distribute it."
        ),
        "supports": [
            _pref("climate_carbon", "deforestation"),
            _pref("solar", "oxygen_cycle"),
            _pref("moon_tides", "tide_mixing"),
        ],
        "distractors": [
            _pref("internet_routing", "routing_distractor"),
            _pref("gps_navigation", "gps_distractor"),
        ],
    },
    {
        "name": "energy_chain",
        "question": "In what way do oxygen, mitochondria, and GPS-guided deliveries enable city hospitals?",
        "answer": (
            "Forests supply oxygen, humans use it in mitochondrial respiration for energy, and GPS-guided logistics ensure supplies reach hospitals reliably."
        ),
        "supports": [
            _pref("climate_carbon", "carbon_sink"),
            _pref("cell_energy", "respiration_chain"),
            _pref("gps_navigation", "gps_triangulation"),
        ],
        "distractors": [
            _pref("moon_tides", "moon_dust"),
            _pref("internet_routing", "routing_distractor"),
        ],
    },
]


QUESTION_PREFIXES = [
    "From a systems perspective, ",
    "In practical terms, ",
    "Briefly outline: ",
    "Step-by-step, ",
    "As a causal chain, ",
]

QUESTION_SUFFIXES = [
    " Please explain concisely.",
    " Focus on cause and effect in your answer.",
    " Highlight each stage in order.",
    " Keep the explanation high-level but complete.",
]

SYNONYM_MAP = {
    "explain": ["describe", "clarify"],
    "how": ["in what way", "through what mechanism"],
    "why": ["for what reason", "what causes"],
    "oxygen": ["oxygen gas", "O₂"],
    "sunlight": ["solar irradiation", "sunshine"],
    "tides": ["tidal cycles", "tidal motions"],
    "routers": ["network routers", "IP routers"],
    "forests": ["woodlands", "forest ecosystems"],
}


def _resolve_episode(domain: str, episode_id: str) -> Dict[str, str]:
    episodes = DOMAINS[domain]["episodes"]  # type: ignore[index]
    if episode_id not in episodes:
        raise KeyError(f"Unknown episode {episode_id} for domain {domain}")
    episode = episodes[episode_id].copy()  # type: ignore[assignment]
    episode.setdefault("domain", domain)
    episode.setdefault("type", "episode")
    return {k: str(v) for k, v in episode.items()}


def _perturb_question(question: str, rng: random.Random) -> str:
    text = question
    if rng.random() < 0.5:
        prefix = rng.choice(QUESTION_PREFIXES)
        text = prefix + text[0].lower() + text[1:]
    if rng.random() < 0.4:
        suffix = rng.choice(QUESTION_SUFFIXES)
        text = text.rstrip("?") + suffix
    # synonym replacements
    for key, options in SYNONYM_MAP.items():
        if key in text and rng.random() < 0.5:
            replacement = rng.choice(options)
            text = text.replace(key, replacement, 1)
    return text


def _collect_documents(
    primary_domain: str,
    supports: Sequence[DocRef],
    distractors: Sequence[DocRef],
    rng: random.Random,
    min_extra_distractors: int = 2,
) -> List[Dict[str, str]]:
    resolved: List[Dict[str, str]] = []

    def _resolve(ref: DocRef) -> Dict[str, str]:
        if isinstance(ref, tuple):
            domain, doc_id = ref
        else:
            domain, doc_id = primary_domain, ref
        episode = _resolve_episode(domain, doc_id)
        episode.setdefault("id", doc_id)
        episode.setdefault("domain", domain)
        return episode

    for ref in supports:
        episode = _resolve(ref)
        episode["is_support"] = "True"
        resolved.append(episode)

    distractor_refs = list(distractors)
    available_domains = list(DOMAINS.keys())
    rng.shuffle(available_domains)
    while len(distractor_refs) < min_extra_distractors:
        domain_choice = rng.choice(available_domains)
        doc_keys = list(DOMAINS[domain_choice]["episodes"].keys())  # type: ignore[index]
        doc_candidate = rng.choice(doc_keys)
        candidate_ref: DocRef = _pref(domain_choice, doc_candidate)
        if candidate_ref not in distractor_refs and candidate_ref not in supports:
            distractor_refs.append(candidate_ref)

    for ref in distractor_refs:
        episode = _resolve(ref)
        episode["is_support"] = "False"
        resolved.append(episode)

    rng.shuffle(resolved)
    return resolved


def build_example(index: int, rng: random.Random) -> Dict[str, object]:
    use_cross = rng.random() < 0.35
    if use_cross:
        scenario = rng.choice(CROSS_DOMAIN_SCENARIOS)
        primary_domain = scenario["supports"][0][0]  # type: ignore[index]
        supports: Sequence[DocRef] = scenario["supports"]  # type: ignore[assignment]
        distractors: Sequence[DocRef] = scenario["distractors"]  # type: ignore[assignment]
        answer = scenario["answer"]  # type: ignore[assignment]
        question = _perturb_question(scenario["question"], rng)  # type: ignore[index]
    else:
        domain_name = list(DOMAINS.keys())[index % len(DOMAINS)]
        domain = DOMAINS[domain_name]
        scenario = rng.choice(domain["scenarios"])  # type: ignore[index]
        primary_domain = domain_name
        supports = scenario["supports"]  # type: ignore[assignment]
        distractors = scenario.get("distractors", [])  # type: ignore[assignment]
        answer = scenario["answer"]  # type: ignore[assignment]
        question = _perturb_question(scenario["question"], rng)  # type: ignore[index]

    docs = _collect_documents(primary_domain, supports, distractors, rng)
    episodes_payload: List[Dict[str, str]] = []
    for j, episode in enumerate(docs):
        domain = episode.get("domain", primary_domain)
        base_id = episode.get("id", f"ep_{index}_{j}")
        doc_label = f"{domain}_{base_id}_{index}_{j}"
        text = episode.get("text")
        if not text:
            text = (
                f"Context: {episode.get('context', '')}. Operation: {episode.get('operation', '')}. "
                f"Affordance: {episode.get('affordance', '')}. Salience: {episode.get('salience', '')}. "
                f"Outcome: {episode.get('outcome', '')}. Goal: {episode.get('goal', '')}."
            )
        role = "support" if episode.get("is_support") == "True" else "distractor"
        episodes_payload.append(
            {
                "id": doc_label,
                "domain": domain,
                "context": episode.get("context", ""),
                "operation": episode.get("operation", ""),
                "affordance": episode.get("affordance", ""),
                "salience": episode.get("salience", ""),
                "outcome": episode.get("outcome", ""),
                "goal": episode.get("goal", ""),
                "type": episode.get("type", "episode"),
                "role": role,
                "text": text,
            }
        )

    return {
        "query": question,
        "ground_truth": answer,
        "episodes": episodes_payload,
    }


def generate_dataset(num_queries: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    return [build_example(i, rng) for i in range(num_queries)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic JSONL dataset for RAG v3-lite.")
    parser.add_argument("--num-queries", type=int, required=True, help="Number of query samples to generate.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSONL file.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    args = parser.parse_args()

    dataset = generate_dataset(args.num_queries, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in dataset:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()
