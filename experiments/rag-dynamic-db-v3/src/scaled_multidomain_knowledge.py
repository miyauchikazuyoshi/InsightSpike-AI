"""
Scaled Multi-Domain Knowledge Base for Large-Scale Experiments
~200 items across 20 domains with rich cross-domain connections
"""

from dataclasses import dataclass
from typing import List, Optional
import random

@dataclass
class ScaledKnowledge:
    text: str
    domain: str
    connects_to: List[str]
    complexity: float  # 0.0-1.0, higher = more complex
    
SCALED_KNOWLEDGE_BASE = [
    # Computer Science (15 items)
    ScaledKnowledge(
        "Machine learning algorithms learn patterns from training data to make predictions",
        "Computer Science", ["Mathematics", "Neuroscience"], 0.7
    ),
    ScaledKnowledge(
        "Neural networks are computational models inspired by biological neural systems",
        "Computer Science", ["Neuroscience", "Biology"], 0.8
    ),
    ScaledKnowledge(
        "Distributed systems coordinate multiple computers to work as a unified system",
        "Computer Science", ["Engineering", "Physics"], 0.6
    ),
    ScaledKnowledge(
        "Quantum computing leverages quantum mechanical phenomena for computation",
        "Computer Science", ["Physics", "Mathematics"], 0.9
    ),
    ScaledKnowledge(
        "Database indexing improves query performance through organized data structures",
        "Computer Science", ["Mathematics"], 0.5
    ),
    ScaledKnowledge(
        "Cryptography ensures secure communication using mathematical algorithms",
        "Computer Science", ["Mathematics", "Economics"], 0.7
    ),
    ScaledKnowledge(
        "Compiler optimization transforms code for better performance",
        "Computer Science", ["Mathematics", "Engineering"], 0.6
    ),
    ScaledKnowledge(
        "Graph algorithms solve problems on network structures",
        "Computer Science", ["Mathematics", "Social Sciences"], 0.6
    ),
    ScaledKnowledge(
        "Computer vision enables machines to interpret visual information",
        "Computer Science", ["Neuroscience", "Psychology"], 0.8
    ),
    ScaledKnowledge(
        "Natural language processing allows computers to understand human language",
        "Computer Science", ["Linguistics", "Psychology"], 0.8
    ),
    ScaledKnowledge(
        "Reinforcement learning trains agents through reward-based feedback",
        "Computer Science", ["Psychology", "Neuroscience"], 0.7
    ),
    ScaledKnowledge(
        "Cloud computing provides on-demand computing resources over networks",
        "Computer Science", ["Economics", "Engineering"], 0.5
    ),
    ScaledKnowledge(
        "Blockchain creates immutable distributed ledgers through cryptographic chains",
        "Computer Science", ["Economics", "Mathematics"], 0.8
    ),
    ScaledKnowledge(
        "Operating systems manage hardware resources and provide user interfaces",
        "Computer Science", ["Engineering"], 0.6
    ),
    ScaledKnowledge(
        "API design patterns facilitate software component communication",
        "Computer Science", ["Engineering", "Design"], 0.5
    ),
    
    # Physics (12 items)
    ScaledKnowledge(
        "Quantum mechanics describes matter and energy at atomic scales",
        "Physics", ["Mathematics", "Chemistry"], 0.9
    ),
    ScaledKnowledge(
        "Thermodynamics governs energy transfer and entropy in systems",
        "Physics", ["Chemistry", "Engineering"], 0.7
    ),
    ScaledKnowledge(
        "Electromagnetic waves propagate energy through space",
        "Physics", ["Engineering", "Medicine"], 0.6
    ),
    ScaledKnowledge(
        "General relativity describes gravity as spacetime curvature",
        "Physics", ["Mathematics", "Astronomy"], 0.95
    ),
    ScaledKnowledge(
        "Particle physics studies fundamental constituents of matter",
        "Physics", ["Chemistry", "Mathematics"], 0.9
    ),
    ScaledKnowledge(
        "Optics explains light behavior and its interactions",
        "Physics", ["Engineering", "Biology"], 0.6
    ),
    ScaledKnowledge(
        "Nuclear fusion powers stars through atomic nuclei combination",
        "Physics", ["Chemistry", "Astronomy"], 0.8
    ),
    ScaledKnowledge(
        "Fluid dynamics models liquid and gas flow behaviors",
        "Physics", ["Engineering", "Environmental Science"], 0.7
    ),
    ScaledKnowledge(
        "Superconductivity enables zero electrical resistance at low temperatures",
        "Physics", ["Materials Science", "Engineering"], 0.8
    ),
    ScaledKnowledge(
        "Wave-particle duality shows matter exhibits both wave and particle properties",
        "Physics", ["Chemistry", "Philosophy"], 0.85
    ),
    ScaledKnowledge(
        "Conservation laws govern energy and momentum in isolated systems",
        "Physics", ["Mathematics", "Engineering"], 0.6
    ),
    ScaledKnowledge(
        "Chaos theory studies sensitive dependence on initial conditions",
        "Physics", ["Mathematics", "Economics"], 0.8
    ),
    
    # Biology (12 items)
    ScaledKnowledge(
        "DNA stores genetic information in double helix structures",
        "Biology", ["Chemistry", "Medicine"], 0.7
    ),
    ScaledKnowledge(
        "Evolution drives species adaptation through natural selection",
        "Biology", ["Environmental Science", "Psychology"], 0.6
    ),
    ScaledKnowledge(
        "Photosynthesis converts light energy into chemical energy",
        "Biology", ["Chemistry", "Physics"], 0.6
    ),
    ScaledKnowledge(
        "Protein folding determines biological function from amino acid sequences",
        "Biology", ["Chemistry", "Computer Science"], 0.8
    ),
    ScaledKnowledge(
        "Ecosystems maintain balance through complex species interactions",
        "Biology", ["Environmental Science", "Chemistry"], 0.7
    ),
    ScaledKnowledge(
        "Cell division enables growth and reproduction in organisms",
        "Biology", ["Medicine", "Chemistry"], 0.5
    ),
    ScaledKnowledge(
        "Neurons transmit information through electrical and chemical signals",
        "Biology", ["Neuroscience", "Physics"], 0.7
    ),
    ScaledKnowledge(
        "Immune systems defend against pathogens through complex mechanisms",
        "Biology", ["Medicine", "Chemistry"], 0.7
    ),
    ScaledKnowledge(
        "Genetic mutations drive evolutionary changes and diversity",
        "Biology", ["Medicine", "Mathematics"], 0.6
    ),
    ScaledKnowledge(
        "Symbiosis creates mutually beneficial relationships between species",
        "Biology", ["Environmental Science", "Economics"], 0.5
    ),
    ScaledKnowledge(
        "Metabolism converts nutrients into energy and building blocks",
        "Biology", ["Chemistry", "Medicine"], 0.6
    ),
    ScaledKnowledge(
        "Stem cells can differentiate into specialized cell types",
        "Biology", ["Medicine", "Ethics"], 0.8
    ),
    
    # Mathematics (10 items)
    ScaledKnowledge(
        "Calculus analyzes continuous change through derivatives and integrals",
        "Mathematics", ["Physics", "Engineering"], 0.7
    ),
    ScaledKnowledge(
        "Linear algebra studies vector spaces and linear transformations",
        "Mathematics", ["Computer Science", "Physics"], 0.6
    ),
    ScaledKnowledge(
        "Probability theory quantifies uncertainty and random phenomena",
        "Mathematics", ["Statistics", "Physics"], 0.6
    ),
    ScaledKnowledge(
        "Group theory examines algebraic structures with operations",
        "Mathematics", ["Physics", "Chemistry"], 0.8
    ),
    ScaledKnowledge(
        "Topology studies properties preserved under continuous deformations",
        "Mathematics", ["Physics", "Computer Science"], 0.85
    ),
    ScaledKnowledge(
        "Number theory investigates properties of integers",
        "Mathematics", ["Computer Science", "Cryptography"], 0.7
    ),
    ScaledKnowledge(
        "Differential equations model dynamic systems and change",
        "Mathematics", ["Physics", "Engineering"], 0.7
    ),
    ScaledKnowledge(
        "Graph theory analyzes networks and relationships",
        "Mathematics", ["Computer Science", "Social Sciences"], 0.6
    ),
    ScaledKnowledge(
        "Fractals exhibit self-similar patterns at different scales",
        "Mathematics", ["Physics", "Art"], 0.7
    ),
    ScaledKnowledge(
        "Game theory models strategic decision-making",
        "Mathematics", ["Economics", "Psychology"], 0.7
    ),
    
    # Psychology (10 items)
    ScaledKnowledge(
        "Cognitive biases systematically affect human judgment and decisions",
        "Psychology", ["Economics", "Neuroscience"], 0.6
    ),
    ScaledKnowledge(
        "Memory consolidation transfers information from short to long-term storage",
        "Psychology", ["Neuroscience", "Education"], 0.6
    ),
    ScaledKnowledge(
        "Operant conditioning shapes behavior through reinforcement",
        "Psychology", ["Education", "Neuroscience"], 0.5
    ),
    ScaledKnowledge(
        "Social psychology studies how people influence each other",
        "Psychology", ["Sociology", "Anthropology"], 0.5
    ),
    ScaledKnowledge(
        "Attachment theory explains emotional bonds in relationships",
        "Psychology", ["Sociology", "Biology"], 0.6
    ),
    ScaledKnowledge(
        "Flow states occur during optimal challenge-skill balance",
        "Psychology", ["Education", "Sports Science"], 0.6
    ),
    ScaledKnowledge(
        "Cognitive load affects learning and problem-solving capacity",
        "Psychology", ["Education", "Computer Science"], 0.6
    ),
    ScaledKnowledge(
        "Emotional intelligence involves recognizing and managing emotions",
        "Psychology", ["Business", "Education"], 0.5
    ),
    ScaledKnowledge(
        "Placebo effects demonstrate mind-body connections in healing",
        "Psychology", ["Medicine", "Neuroscience"], 0.7
    ),
    ScaledKnowledge(
        "Developmental psychology tracks changes across lifespan",
        "Psychology", ["Biology", "Education"], 0.5
    ),
    
    # Chemistry (10 items)
    ScaledKnowledge(
        "Chemical bonds form through electron sharing or transfer",
        "Chemistry", ["Physics", "Materials Science"], 0.5
    ),
    ScaledKnowledge(
        "Catalysts accelerate reactions without being consumed",
        "Chemistry", ["Biology", "Engineering"], 0.6
    ),
    ScaledKnowledge(
        "pH measures hydrogen ion concentration in solutions",
        "Chemistry", ["Biology", "Environmental Science"], 0.4
    ),
    ScaledKnowledge(
        "Organic chemistry studies carbon-based compounds",
        "Chemistry", ["Biology", "Medicine"], 0.6
    ),
    ScaledKnowledge(
        "Thermochemistry analyzes energy changes in reactions",
        "Chemistry", ["Physics", "Engineering"], 0.6
    ),
    ScaledKnowledge(
        "Polymer chains create materials with unique properties",
        "Chemistry", ["Materials Science", "Engineering"], 0.6
    ),
    ScaledKnowledge(
        "Electrochemistry studies electron transfer in reactions",
        "Chemistry", ["Physics", "Engineering"], 0.7
    ),
    ScaledKnowledge(
        "Crystal structures determine material properties",
        "Chemistry", ["Materials Science", "Physics"], 0.7
    ),
    ScaledKnowledge(
        "Chemical equilibrium balances forward and reverse reactions",
        "Chemistry", ["Physics", "Biology"], 0.5
    ),
    ScaledKnowledge(
        "Spectroscopy identifies substances through light interaction",
        "Chemistry", ["Physics", "Astronomy"], 0.7
    ),
    
    # Economics (10 items)
    ScaledKnowledge(
        "Supply and demand determine market prices",
        "Economics", ["Mathematics", "Psychology"], 0.4
    ),
    ScaledKnowledge(
        "Game theory models strategic economic decisions",
        "Economics", ["Mathematics", "Psychology"], 0.7
    ),
    ScaledKnowledge(
        "Behavioral economics incorporates psychological insights",
        "Economics", ["Psychology", "Neuroscience"], 0.6
    ),
    ScaledKnowledge(
        "Network effects increase value with user adoption",
        "Economics", ["Computer Science", "Sociology"], 0.6
    ),
    ScaledKnowledge(
        "Monetary policy influences economic activity through interest rates",
        "Economics", ["Politics", "Mathematics"], 0.6
    ),
    ScaledKnowledge(
        "Market efficiency hypothesis claims prices reflect all information",
        "Economics", ["Mathematics", "Psychology"], 0.7
    ),
    ScaledKnowledge(
        "Externalities create costs or benefits for third parties",
        "Economics", ["Environmental Science", "Ethics"], 0.5
    ),
    ScaledKnowledge(
        "Comparative advantage drives international trade",
        "Economics", ["Politics", "Geography"], 0.5
    ),
    ScaledKnowledge(
        "Innovation cycles drive economic growth and disruption",
        "Economics", ["Business", "Technology"], 0.6
    ),
    ScaledKnowledge(
        "Auction theory optimizes bidding strategies and mechanisms",
        "Economics", ["Mathematics", "Game Theory"], 0.7
    ),
    
    # Medicine (10 items)
    ScaledKnowledge(
        "Antibiotics target bacterial infections but not viruses",
        "Medicine", ["Biology", "Chemistry"], 0.5
    ),
    ScaledKnowledge(
        "Vaccines train immune systems to recognize pathogens",
        "Medicine", ["Biology", "Public Health"], 0.6
    ),
    ScaledKnowledge(
        "Personalized medicine tailors treatments to genetic profiles",
        "Medicine", ["Biology", "Computer Science"], 0.8
    ),
    ScaledKnowledge(
        "Clinical trials test treatment safety and efficacy",
        "Medicine", ["Statistics", "Ethics"], 0.6
    ),
    ScaledKnowledge(
        "Diagnostic imaging reveals internal body structures",
        "Medicine", ["Physics", "Engineering"], 0.6
    ),
    ScaledKnowledge(
        "Epidemiology tracks disease patterns in populations",
        "Medicine", ["Statistics", "Public Health"], 0.6
    ),
    ScaledKnowledge(
        "Gene therapy modifies genetic material to treat diseases",
        "Medicine", ["Biology", "Ethics"], 0.8
    ),
    ScaledKnowledge(
        "Telemedicine delivers healthcare through digital platforms",
        "Medicine", ["Technology", "Public Health"], 0.5
    ),
    ScaledKnowledge(
        "Regenerative medicine repairs tissues using stem cells",
        "Medicine", ["Biology", "Engineering"], 0.8
    ),
    ScaledKnowledge(
        "Drug interactions can enhance or reduce medication effects",
        "Medicine", ["Chemistry", "Biology"], 0.6
    ),
    
    # Neuroscience (8 items)
    ScaledKnowledge(
        "Synaptic plasticity enables learning through connection strength changes",
        "Neuroscience", ["Biology", "Psychology"], 0.7
    ),
    ScaledKnowledge(
        "Brain imaging techniques reveal neural activity patterns",
        "Neuroscience", ["Medicine", "Physics"], 0.7
    ),
    ScaledKnowledge(
        "Neurotransmitters carry signals between neurons",
        "Neuroscience", ["Biology", "Chemistry"], 0.6
    ),
    ScaledKnowledge(
        "Mirror neurons activate during action observation and execution",
        "Neuroscience", ["Psychology", "Biology"], 0.7
    ),
    ScaledKnowledge(
        "Brain-computer interfaces translate neural signals to commands",
        "Neuroscience", ["Computer Science", "Engineering"], 0.8
    ),
    ScaledKnowledge(
        "Neuroplasticity allows brain reorganization throughout life",
        "Neuroscience", ["Medicine", "Psychology"], 0.6
    ),
    ScaledKnowledge(
        "Default mode network activates during rest and introspection",
        "Neuroscience", ["Psychology", "Medicine"], 0.7
    ),
    ScaledKnowledge(
        "Dopamine pathways regulate reward and motivation",
        "Neuroscience", ["Psychology", "Medicine"], 0.6
    ),
    
    # Environmental Science (8 items)
    ScaledKnowledge(
        "Climate feedback loops amplify or dampen temperature changes",
        "Environmental Science", ["Physics", "Chemistry"], 0.7
    ),
    ScaledKnowledge(
        "Carbon cycles move carbon through atmosphere, land, and oceans",
        "Environmental Science", ["Chemistry", "Biology"], 0.6
    ),
    ScaledKnowledge(
        "Biodiversity loss threatens ecosystem stability",
        "Environmental Science", ["Biology", "Economics"], 0.6
    ),
    ScaledKnowledge(
        "Renewable energy sources provide sustainable power alternatives",
        "Environmental Science", ["Engineering", "Economics"], 0.5
    ),
    ScaledKnowledge(
        "Ocean acidification affects marine ecosystems",
        "Environmental Science", ["Chemistry", "Biology"], 0.6
    ),
    ScaledKnowledge(
        "Urban heat islands increase city temperatures",
        "Environmental Science", ["Physics", "Urban Planning"], 0.5
    ),
    ScaledKnowledge(
        "Ecological succession creates predictable community changes",
        "Environmental Science", ["Biology", "Geography"], 0.6
    ),
    ScaledKnowledge(
        "Water cycles connect atmospheric, surface, and groundwater",
        "Environmental Science", ["Chemistry", "Geography"], 0.5
    ),
    
    # Engineering (8 items)
    ScaledKnowledge(
        "Control systems maintain desired outputs through feedback",
        "Engineering", ["Mathematics", "Computer Science"], 0.6
    ),
    ScaledKnowledge(
        "Materials science optimizes properties for applications",
        "Engineering", ["Chemistry", "Physics"], 0.6
    ),
    ScaledKnowledge(
        "Signal processing extracts information from measurements",
        "Engineering", ["Mathematics", "Computer Science"], 0.6
    ),
    ScaledKnowledge(
        "Structural engineering ensures building safety and stability",
        "Engineering", ["Physics", "Mathematics"], 0.5
    ),
    ScaledKnowledge(
        "Robotics integrates sensors, actuators, and control algorithms",
        "Engineering", ["Computer Science", "Mathematics"], 0.7
    ),
    ScaledKnowledge(
        "Aerospace engineering designs aircraft and spacecraft",
        "Engineering", ["Physics", "Materials Science"], 0.8
    ),
    ScaledKnowledge(
        "Biomedical engineering applies engineering to healthcare",
        "Engineering", ["Medicine", "Biology"], 0.7
    ),
    ScaledKnowledge(
        "Energy systems convert and distribute power efficiently",
        "Engineering", ["Physics", "Environmental Science"], 0.6
    ),
    
    # Philosophy (8 items)
    ScaledKnowledge(
        "Epistemology examines nature and limits of knowledge",
        "Philosophy", ["Psychology", "Mathematics"], 0.8
    ),
    ScaledKnowledge(
        "Ethics evaluates moral principles and decision-making",
        "Philosophy", ["Psychology", "Law"], 0.6
    ),
    ScaledKnowledge(
        "Consciousness poses hard problems for physical explanations",
        "Philosophy", ["Neuroscience", "Psychology"], 0.9
    ),
    ScaledKnowledge(
        "Logic formalizes valid reasoning and inference",
        "Philosophy", ["Mathematics", "Computer Science"], 0.7
    ),
    ScaledKnowledge(
        "Free will debates determinism versus agency",
        "Philosophy", ["Neuroscience", "Physics"], 0.8
    ),
    ScaledKnowledge(
        "Aesthetics explores nature of beauty and art",
        "Philosophy", ["Art", "Psychology"], 0.6
    ),
    ScaledKnowledge(
        "Philosophy of science examines scientific methodology",
        "Philosophy", ["Science", "Mathematics"], 0.7
    ),
    ScaledKnowledge(
        "Existentialism emphasizes individual existence and freedom",
        "Philosophy", ["Psychology", "Literature"], 0.7
    ),
    
    # Sociology (8 items)
    ScaledKnowledge(
        "Social networks shape information flow and behavior",
        "Sociology", ["Psychology", "Computer Science"], 0.6
    ),
    ScaledKnowledge(
        "Cultural capital influences social mobility",
        "Sociology", ["Economics", "Education"], 0.6
    ),
    ScaledKnowledge(
        "Group dynamics affect decision-making and performance",
        "Sociology", ["Psychology", "Business"], 0.5
    ),
    ScaledKnowledge(
        "Social stratification creates hierarchical structures",
        "Sociology", ["Economics", "Politics"], 0.6
    ),
    ScaledKnowledge(
        "Urbanization transforms social relationships and structures",
        "Sociology", ["Geography", "Economics"], 0.5
    ),
    ScaledKnowledge(
        "Social movements drive collective action for change",
        "Sociology", ["Politics", "Psychology"], 0.6
    ),
    ScaledKnowledge(
        "Symbolic interactionism studies meaning in social interactions",
        "Sociology", ["Psychology", "Anthropology"], 0.7
    ),
    ScaledKnowledge(
        "Digital sociology examines technology's social impacts",
        "Sociology", ["Technology", "Psychology"], 0.6
    ),
    
    # Linguistics (7 items)
    ScaledKnowledge(
        "Language acquisition follows predictable developmental stages",
        "Linguistics", ["Psychology", "Neuroscience"], 0.6
    ),
    ScaledKnowledge(
        "Syntax rules govern sentence structure formation",
        "Linguistics", ["Computer Science", "Psychology"], 0.6
    ),
    ScaledKnowledge(
        "Semantics studies meaning in language",
        "Linguistics", ["Philosophy", "Computer Science"], 0.6
    ),
    ScaledKnowledge(
        "Phonetics analyzes speech sounds and production",
        "Linguistics", ["Physics", "Biology"], 0.5
    ),
    ScaledKnowledge(
        "Language evolution traces historical linguistic changes",
        "Linguistics", ["Anthropology", "History"], 0.6
    ),
    ScaledKnowledge(
        "Pragmatics examines context-dependent meaning",
        "Linguistics", ["Philosophy", "Psychology"], 0.6
    ),
    ScaledKnowledge(
        "Computational linguistics models language processing",
        "Linguistics", ["Computer Science", "Mathematics"], 0.7
    ),
    
    # Anthropology (7 items)
    ScaledKnowledge(
        "Cultural evolution parallels biological evolution principles",
        "Anthropology", ["Biology", "Sociology"], 0.6
    ),
    ScaledKnowledge(
        "Ethnography documents cultural practices through observation",
        "Anthropology", ["Sociology", "Psychology"], 0.5
    ),
    ScaledKnowledge(
        "Human evolution traces anatomical and behavioral changes",
        "Anthropology", ["Biology", "Archaeology"], 0.6
    ),
    ScaledKnowledge(
        "Kinship systems organize social relationships",
        "Anthropology", ["Sociology", "Biology"], 0.5
    ),
    ScaledKnowledge(
        "Ritual behaviors reinforce social bonds and beliefs",
        "Anthropology", ["Psychology", "Sociology"], 0.5
    ),
    ScaledKnowledge(
        "Material culture reflects technological and social development",
        "Anthropology", ["Archaeology", "History"], 0.5
    ),
    ScaledKnowledge(
        "Linguistic anthropology studies language in cultural context",
        "Anthropology", ["Linguistics", "Sociology"], 0.6
    ),
    
    # Art & Design (7 items)
    ScaledKnowledge(
        "Color theory guides visual composition and emotion",
        "Art", ["Psychology", "Physics"], 0.5
    ),
    ScaledKnowledge(
        "Golden ratio appears in natural and artistic forms",
        "Art", ["Mathematics", "Biology"], 0.5
    ),
    ScaledKnowledge(
        "User experience design balances aesthetics and functionality",
        "Design", ["Psychology", "Computer Science"], 0.6
    ),
    ScaledKnowledge(
        "Typography affects readability and visual communication",
        "Design", ["Psychology", "Computer Science"], 0.5
    ),
    ScaledKnowledge(
        "Generative art uses algorithms to create visual works",
        "Art", ["Computer Science", "Mathematics"], 0.7
    ),
    ScaledKnowledge(
        "Architecture integrates form, function, and environment",
        "Design", ["Engineering", "Art"], 0.6
    ),
    ScaledKnowledge(
        "Visual perception principles guide effective design",
        "Design", ["Psychology", "Neuroscience"], 0.5
    ),
    
    # History (6 items)
    ScaledKnowledge(
        "Historical patterns reveal cyclical social phenomena",
        "History", ["Sociology", "Economics"], 0.6
    ),
    ScaledKnowledge(
        "Technological revolutions transform societies rapidly",
        "History", ["Technology", "Economics"], 0.5
    ),
    ScaledKnowledge(
        "Historiography examines how history is written and interpreted",
        "History", ["Philosophy", "Sociology"], 0.6
    ),
    ScaledKnowledge(
        "Archaeological evidence reconstructs past civilizations",
        "History", ["Anthropology", "Chemistry"], 0.6
    ),
    ScaledKnowledge(
        "Cultural diffusion spreads ideas across societies",
        "History", ["Anthropology", "Geography"], 0.5
    ),
    ScaledKnowledge(
        "Historical methodology combines multiple evidence sources",
        "History", ["Philosophy", "Statistics"], 0.6
    ),
    
    # Astronomy (6 items)
    ScaledKnowledge(
        "Stellar evolution follows predictable life cycles",
        "Astronomy", ["Physics", "Chemistry"], 0.7
    ),
    ScaledKnowledge(
        "Dark matter influences galaxy formation and structure",
        "Astronomy", ["Physics", "Mathematics"], 0.8
    ),
    ScaledKnowledge(
        "Exoplanets reveal diverse planetary system configurations",
        "Astronomy", ["Physics", "Chemistry"], 0.7
    ),
    ScaledKnowledge(
        "Cosmic microwave background provides universe origin evidence",
        "Astronomy", ["Physics", "Mathematics"], 0.8
    ),
    ScaledKnowledge(
        "Gravitational waves detect spacetime distortions",
        "Astronomy", ["Physics", "Engineering"], 0.9
    ),
    ScaledKnowledge(
        "Astrobiology searches for life beyond Earth",
        "Astronomy", ["Biology", "Chemistry"], 0.7
    ),
    
    # Statistics (6 items)
    ScaledKnowledge(
        "Bayesian inference updates beliefs with new evidence",
        "Statistics", ["Mathematics", "Computer Science"], 0.7
    ),
    ScaledKnowledge(
        "Central limit theorem explains normal distribution prevalence",
        "Statistics", ["Mathematics", "Physics"], 0.6
    ),
    ScaledKnowledge(
        "Hypothesis testing evaluates claims against data",
        "Statistics", ["Mathematics", "Science"], 0.5
    ),
    ScaledKnowledge(
        "Machine learning uses statistical models for prediction",
        "Statistics", ["Computer Science", "Mathematics"], 0.7
    ),
    ScaledKnowledge(
        "Sampling bias affects data representativeness",
        "Statistics", ["Psychology", "Sociology"], 0.5
    ),
    ScaledKnowledge(
        "Time series analysis identifies temporal patterns",
        "Statistics", ["Mathematics", "Economics"], 0.6
    ),
]

def get_scaled_knowledge_base():
    """Return the full scaled knowledge base"""
    return SCALED_KNOWLEDGE_BASE

def get_knowledge_by_domain(domain: str):
    """Get all knowledge items for a specific domain"""
    return [k for k in SCALED_KNOWLEDGE_BASE if k.domain == domain]

def get_cross_domain_connections():
    """Analyze cross-domain connection patterns"""
    connections = {}
    for item in SCALED_KNOWLEDGE_BASE:
        if item.domain not in connections:
            connections[item.domain] = set()
        for connected in item.connects_to:
            connections[item.domain].add(connected)
    return connections

def get_knowledge_stats():
    """Get statistics about the knowledge base"""
    domains = {}
    total_connections = 0
    complexity_sum = 0
    
    for item in SCALED_KNOWLEDGE_BASE:
        if item.domain not in domains:
            domains[item.domain] = 0
        domains[item.domain] += 1
        total_connections += len(item.connects_to)
        complexity_sum += item.complexity
    
    return {
        "total_items": len(SCALED_KNOWLEDGE_BASE),
        "num_domains": len(domains),
        "items_per_domain": domains,
        "total_connections": total_connections,
        "avg_connections_per_item": total_connections / len(SCALED_KNOWLEDGE_BASE),
        "avg_complexity": complexity_sum / len(SCALED_KNOWLEDGE_BASE),
    }

if __name__ == "__main__":
    stats = get_knowledge_stats()
    print(f"Knowledge Base Statistics:")
    print(f"Total Items: {stats['total_items']}")
    print(f"Number of Domains: {stats['num_domains']}")
    print(f"Average Connections: {stats['avg_connections_per_item']:.2f}")
    print(f"Average Complexity: {stats['avg_complexity']:.2f}")
    
    print(f"\nItems per Domain:")
    for domain, count in sorted(stats['items_per_domain'].items()):
        print(f"  {domain}: {count}")