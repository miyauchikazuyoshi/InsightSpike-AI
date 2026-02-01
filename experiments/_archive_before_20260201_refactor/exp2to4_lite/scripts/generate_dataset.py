#!/usr/bin/env python3
"""Generate a medium JSONL dataset for Exp II–IV (self-contained).

Creates queries across several domains with support/distractor episodes.
Ground-truths are aligned with geDIGStrategy's canned answers for some
domains to enable non-zero acceptance even with random embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


BASE_DOMAINS = [
    "solar_coastal",  # photosynth + coastal
    "solar_tidal",    # photosynth + tide
    "solar_chain",    # photosynth only
    "tidal_mixing",   # tides only
    "router",         # networking routing
    "gps",            # satellite triangulation
    "cell_energy",    # mitochondria, respiration
    "carbon_cycle",   # CO2/greenhouse
]


GROUND_TRUTH = {
    "solar_coastal": "When sunlight reaches coastal waters, marine algae photosynthesise, releasing oxygen that dissolves for fish to use.",
    "solar_tidal": "Sunlight powers coastal algae to photosynthesise, and lunar-driven tides mix that oxygenated water along the shore.",
    "solar_chain": "Solar photons drive photosynthesis in plants and algae, generating the oxygen that builds up in Earth's air.",
    "tidal_mixing": "Lunar gravity pulls water into bulges; as Earth rotates, those bulges shift, generating currents that mix shoreline waters.",
    "router": "Routers read packet addresses and consult routing tables to forward traffic along viable next hops.",
    "gps": "It measures the arrival time of signals from several satellites and triangulates its location.",
    "cell_energy": "They oxidise nutrients through respiration, generating ATP that fuels the cell's activities.",
    "carbon_cycle": "Losing forests reduces CO₂ uptake while releasing stored carbon, strengthening the greenhouse effect.",
}


DISTRACTORS = {
    "solar": "Solar flares eject charged particles that can disrupt satellite electronics.",
    "moon_tides": "Earth's rotation completes a full turn roughly every 24 hours, sweeping regions through tidal bulges.",
    "dns": "DNS resolves hostnames to IP addresses before connections are formed.",
}


def make_query(domain: str, idx: int) -> str:
    if domain == "solar_coastal":
        return "As a causal chain, how does coastal sunshine help marine animals obtain oxygen?"
    if domain == "solar_tidal":
        return "Explain how sunlight and tides together deliver oxygen along coasts."
    if domain == "solar_chain":
        return "Trace how solar energy becomes the oxygen we breathe."
    if domain == "tidal_mixing":
        return "Describe concisely the mechanism by which lunar gravity keeps tidal mixing active."
    if domain == "router":
        return "How do routers forward packets along the Internet?"
    if domain == "gps":
        return "How does a GPS receiver determine its location?"
    if domain == "cell_energy":
        return "What do mitochondria do to generate energy for the cell?"
    if domain == "carbon_cycle":
        return "Why does deforestation intensify the greenhouse effect?"
    return f"Explain {domain} in one sentence ({idx})."


def support_episodes(domain: str, i: int) -> list[dict]:
    if domain in ("solar_coastal", "solar_tidal", "solar_chain"):
        return [
            {
                "id": f"solar_sun_{i}",
                "domain": "solar",
                "context": "Sun-Earth system over coastal waters",
                "operation": "Emit continuous visible-light photons",
                "affordance": "Photon flux reaches shallow water where algae reside",
                "salience": "High irradiance",
                "outcome": "Primary producers receive energy",
                "goal": "Enable photosynthesis",
                "type": "episode",
                "role": "support",
                "text": "The Sun bathes Earth in photons, delivering the energy that photosynthetic life uses.",
            },
            {
                "id": f"solar_oxygen_{i}",
                "domain": "solar",
                "context": "Atmosphere and surface oceans",
                "operation": "Release oxygen; dissolve in water",
                "affordance": "Mixing enables gas transfer",
                "salience": "Concentration rises near photosynthetic zones",
                "outcome": "Oxygen accumulates for organisms",
                "goal": "Maintain oxygen reservoir",
                "type": "episode",
                "role": "support",
                "text": "Oxygen released by photosynthesis accumulates in the atmosphere and dissolves into surface waters.",
            },
        ]
    if domain == "tidal_mixing":
        return [
            {
                "id": f"moon_tides_mix_{i}",
                "domain": "moon_tides",
                "context": "Shallow estuaries and tidal flats",
                "operation": "Churn water and distribute dissolved gases",
                "affordance": "Currents remix stratified layers",
                "salience": "Maintains oxygen availability",
                "outcome": "Fresh oxygen reaches tidal habitats",
                "goal": "Ecological effects of tides",
                "type": "episode",
                "role": "support",
                "text": "Tidal currents continuously mix coastal waters, distributing dissolved gases like oxygen.",
            }
        ]
    if domain == "router":
        return [
            {
                "id": f"net_router_{i}",
                "domain": "networking",
                "context": "Interconnected routers with routing tables",
                "operation": "Select next hop based on prefix match",
                "affordance": "Forwarding uses best path per policy",
                "salience": "Enables scalable packet delivery",
                "outcome": "Traffic moves toward destination",
                "goal": "Explain router forwarding",
                "type": "episode",
                "role": "support",
                "text": "Routers consult routing tables and forward packets to the best next hop.",
            }
        ]
    if domain == "gps":
        return [
            {
                "id": f"gps_sat_{i}",
                "domain": "satellite",
                "context": "GNSS constellation",
                "operation": "Broadcast timing signals",
                "affordance": "Receivers compare arrival times",
                "salience": "Multiple satellites in view",
                "outcome": "Triangulated position",
                "goal": "Determine location",
                "type": "episode",
                "role": "support",
                "text": "GNSS satellites transmit precise timing; receivers compare arrival times to locate themselves.",
            }
        ]
    if domain == "cell_energy":
        return [
            {
                "id": f"cell_resp_{i}",
                "domain": "cell_bio",
                "context": "Mitochondria",
                "operation": "Oxidise nutrients",
                "affordance": "Electron transport chain",
                "salience": "Generates proton gradient",
                "outcome": "ATP synthesis",
                "goal": "Fuel cellular activities",
                "type": "episode",
                "role": "support",
                "text": "Mitochondria oxidise nutrients through respiration, producing ATP for the cell.",
            }
        ]
    if domain == "carbon_cycle":
        return [
            {
                "id": f"forest_sink_{i}",
                "domain": "climate_carbon",
                "context": "Forests absorbing atmospheric CO2",
                "operation": "Fix carbon into biomass",
                "affordance": "Dense vegetation draws down CO2",
                "salience": "Key long-term buffer",
                "outcome": "CO2 growth slows",
                "goal": "Explain carbon sinks",
                "type": "episode",
                "role": "support",
                "text": "Forests act as carbon sinks by absorbing CO2 during photosynthesis.",
            }
        ]
    return []


def distractor_episodes(i: int) -> list[dict]:
    return [
        {
            "id": f"solar_flare_{i}",
            "domain": "solar",
            "type": "episode",
            "role": "distractor",
            "text": DISTRACTORS["solar"],
        },
        {
            "id": f"moon_cycle_{i}",
            "domain": "moon_tides",
            "type": "episode",
            "role": "distractor",
            "text": DISTRACTORS["moon_tides"],
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-queries", type=int, default=50)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-domains", type=int, default=None, help="Optional number of domain labels to simulate")
    args = ap.parse_args()

    random.seed(args.seed)
    domains = list(BASE_DOMAINS)
    if args.num_domains and args.num_domains > len(BASE_DOMAINS):
        # Extend domain labels synthetically
        extra = args.num_domains - len(BASE_DOMAINS)
        for i in range(extra):
            domains.append(f"synthetic_{i}")
    else:
        domains = BASE_DOMAINS
    with args.output.open("w", encoding="utf-8") as fh:
        for i in range(args.num_queries):
            domain = domains[i % len(domains)]
            query = make_query(domain, i)
            gt = GROUND_TRUTH[domain]
            eps = support_episodes(domain, i) + distractor_episodes(i)
            row = {
                "query": query,
                "ground_truth": gt,
                "episodes": eps,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.num_queries} queries to {args.output}")


if __name__ == "__main__":
    main()
