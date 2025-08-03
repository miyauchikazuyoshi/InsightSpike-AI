#!/usr/bin/env python3
"""
Generate 500 knowledge entries for the experiment
"""

import json
import random
from datetime import datetime

def generate_knowledge_base():
    """Generate 500 diverse knowledge entries"""
    
    # Knowledge templates by category
    science_tech_templates = [
        # Physics
        ("Newton's {} law states that {}", ["physics", "mechanics"]),
        ("The principle of {} demonstrates that {}", ["physics", "principles"]),
        ("In thermodynamics, {} is defined as {}", ["physics", "thermodynamics"]),
        ("Electromagnetic {} occurs when {}", ["physics", "electromagnetism"]),
        ("The {} effect in physics explains {}", ["physics", "phenomena"]),
        # Chemistry
        ("Chemical {} reactions involve {}", ["chemistry", "reactions"]),
        ("The element {} is characterized by {}", ["chemistry", "elements"]),
        ("In organic chemistry, {} compounds {}", ["chemistry", "organic"]),
        ("The process of {} in chemistry {}", ["chemistry", "processes"]),
        # Biology
        ("The {} system in organisms {}", ["biology", "anatomy"]),
        ("Cellular {} is responsible for {}", ["biology", "cell biology"]),
        ("In genetics, {} determines {}", ["biology", "genetics"]),
        ("The evolution of {} shows {}", ["biology", "evolution"]),
        # Technology
        ("The algorithm for {} works by {}", ["technology", "algorithms"]),
        ("In computer networks, {} protocols {}", ["technology", "networking"]),
        ("Data {} techniques involve {}", ["technology", "data science"]),
        ("Artificial intelligence uses {} to {}", ["technology", "AI"]),
    ]
    
    math_logic_templates = [
        ("The theorem of {} proves that {}", ["mathematics", "theorems"]),
        ("In algebra, {} operations {}", ["mathematics", "algebra"]),
        ("Statistical {} measures {}", ["mathematics", "statistics"]),
        ("The {} sequence follows the pattern {}", ["mathematics", "sequences"]),
        ("Logical {} statements are {}", ["logic", "formal logic"]),
        ("In set theory, {} represents {}", ["mathematics", "set theory"]),
        ("Calculus uses {} to find {}", ["mathematics", "calculus"]),
        ("The {} algorithm solves {}", ["mathematics", "algorithms"]),
    ]
    
    history_culture_templates = [
        ("The {} civilization developed {}", ["history", "ancient history"]),
        ("During the {} period, society {}", ["history", "historical periods"]),
        ("The cultural practice of {} originated {}", ["culture", "traditions"]),
        ("The {} war resulted in {}", ["history", "conflicts"]),
        ("The invention of {} changed {}", ["history", "innovations"]),
        ("The {} movement advocated for {}", ["history", "social movements"]),
        ("Ancient {} believed that {}", ["culture", "mythology"]),
        ("The {} dynasty ruled {}", ["history", "dynasties"]),
    ]
    
    daily_life_templates = [
        ("When cooking {}, it's important to {}", ["daily_life", "cooking"]),
        ("Regular {} exercise helps {}", ["daily_life", "health", "sports"]),
        ("The technique of {} involves {}", ["daily_life", "skills"]),
        ("In nutrition, {} provides {}", ["daily_life", "health", "nutrition"]),
        ("Home {} maintenance requires {}", ["daily_life", "household"]),
        ("The sport of {} develops {}", ["daily_life", "sports"]),
        ("Personal {} habits affect {}", ["daily_life", "lifestyle"]),
        ("Effective {} communication involves {}", ["daily_life", "social skills"]),
    ]
    
    arts_literature_templates = [
        ("The literary device of {} creates {}", ["literature", "writing techniques"]),
        ("In {} art, the use of {}", ["art", "visual arts"]),
        ("The {} musical scale consists of {}", ["music", "music theory"]),
        ("The poet {} expressed {}", ["literature", "poetry"]),
        ("The artistic style of {} emphasizes {}", ["art", "art movements"]),
        ("In theater, {} technique {}", ["art", "performing arts"]),
        ("The novel {} explores themes of {}", ["literature", "fiction"]),
        ("Musical {} harmony creates {}", ["music", "composition"]),
    ]
    
    philosophy_psychology_templates = [
        ("The philosophical concept of {} suggests {}", ["philosophy", "metaphysics"]),
        ("In psychology, {} behavior indicates {}", ["psychology", "behavioral"]),
        ("The theory of {} mind proposes {}", ["psychology", "cognitive"]),
        ("Ethical {} principles state that {}", ["philosophy", "ethics"]),
        ("The {} paradox questions {}", ["philosophy", "paradoxes"]),
        ("Psychological {} development occurs {}", ["psychology", "development"]),
        ("The philosophy of {} examines {}", ["philosophy", "branches"]),
        ("Cognitive {} processes involve {}", ["psychology", "cognition"]),
    ]
    
    economics_business_templates = [
        ("Market {} analysis shows {}", ["economics", "markets"]),
        ("The business strategy of {} focuses on {}", ["business", "strategy"]),
        ("Economic {} theory explains {}", ["economics", "theory"]),
        ("In finance, {} instruments {}", ["finance", "investments"]),
        ("The principle of {} management {}", ["business", "management"]),
        ("Global {} trade affects {}", ["economics", "international"]),
        ("Investment {} strategies aim to {}", ["finance", "portfolio"]),
        ("The {} business model relies on {}", ["business", "models"]),
    ]
    
    # Generate entries
    knowledge_entries = []
    entry_id = 1
    
    # Science & Technology (100 entries)
    for i in range(100):
        template, base_tags = random.choice(science_tech_templates)
        topic = random.choice([
            "quantum mechanics", "relativity", "entropy", "catalysis", "mitosis",
            "neural networks", "encryption", "photonics", "genomics", "nanotechnology",
            "superconductivity", "fusion", "crystallography", "spectroscopy", "biotechnology"
        ])
        description = random.choice([
            "fundamental particles interact through force carriers",
            "energy states transition between discrete levels",
            "complex systems exhibit emergent properties",
            "information processing occurs through parallel pathways",
            "molecular structures determine functional properties"
        ])
        
        entry = {
            "id": f"knowledge_{entry_id:03d}",
            "content": template.format(topic, description),
            "tags": ["science"] + base_tags,
            "difficulty": random.choice(["basic", "intermediate", "advanced"]),
            "related_concepts": random.sample([
                "energy", "matter", "information", "structure", "function",
                "systems", "emergence", "complexity", "patterns", "interactions"
            ], 3)
        }
        knowledge_entries.append(entry)
        entry_id += 1
    
    # Mathematics & Logic (75 entries)
    for i in range(75):
        template, base_tags = random.choice(math_logic_templates)
        topic = random.choice([
            "Fermat", "Euler", "Gauss", "prime numbers", "infinity",
            "topology", "probability", "matrices", "vectors", "manifolds",
            "recursion", "induction", "deduction", "axioms", "proofs"
        ])
        description = random.choice([
            "every continuous function has a fixed point",
            "the sum converges to a finite value",
            "there exists a unique solution",
            "the relationship is bijective",
            "the operation preserves structure"
        ])
        
        entry = {
            "id": f"knowledge_{entry_id:03d}",
            "content": template.format(topic, description),
            "tags": ["mathematics"] + base_tags,
            "difficulty": random.choice(["basic", "intermediate", "advanced"]),
            "related_concepts": random.sample([
                "proof", "logic", "patterns", "relationships", "abstraction",
                "generalization", "optimization", "symmetry", "invariance", "transformation"
            ], 3)
        }
        knowledge_entries.append(entry)
        entry_id += 1
    
    # History & Culture (75 entries)
    for i in range(75):
        template, base_tags = random.choice(history_culture_templates)
        topic = random.choice([
            "Roman", "Egyptian", "Mayan", "Industrial Revolution", "Renaissance",
            "Enlightenment", "Cold War", "Digital Age", "Bronze Age", "Space Age",
            "tea ceremony", "democracy", "feudalism", "capitalism", "revolution"
        ])
        description = random.choice([
            "sophisticated irrigation systems for agriculture",
            "new forms of artistic expression",
            "fundamental changes in social structure",
            "technological innovations that transformed society",
            "philosophical ideas that challenged tradition"
        ])
        
        entry = {
            "id": f"knowledge_{entry_id:03d}",
            "content": template.format(topic, description),
            "tags": ["history"] + base_tags,
            "difficulty": random.choice(["basic", "intermediate"]),
            "related_concepts": random.sample([
                "civilization", "progress", "conflict", "cooperation", "innovation",
                "tradition", "change", "power", "culture", "society"
            ], 3)
        }
        knowledge_entries.append(entry)
        entry_id += 1
    
    # Daily Life (75 entries)
    for i in range(75):
        template, base_tags = random.choice(daily_life_templates)
        topic = random.choice([
            "bread", "yoga", "gardening", "meditation", "cleaning",
            "budgeting", "time management", "social skills", "parenting", "travel",
            "photography", "writing", "speaking", "listening", "organizing"
        ])
        description = random.choice([
            "maintain proper temperature and timing",
            "improve physical and mental well-being",
            "create lasting positive habits",
            "enhance personal effectiveness",
            "build stronger relationships"
        ])
        
        entry = {
            "id": f"knowledge_{entry_id:03d}",
            "content": template.format(topic, description),
            "tags": ["daily_life"] + base_tags,
            "difficulty": "basic",
            "related_concepts": random.sample([
                "practice", "improvement", "balance", "wellness", "efficiency",
                "relationships", "habits", "skills", "growth", "mindfulness"
            ], 3)
        }
        knowledge_entries.append(entry)
        entry_id += 1
    
    # Arts & Literature (75 entries)
    for i in range(75):
        template, base_tags = random.choice(arts_literature_templates)
        topic = random.choice([
            "metaphor", "perspective", "harmony", "rhythm", "contrast",
            "symbolism", "impressionism", "realism", "abstraction", "minimalism",
            "sonnet", "fugue", "fresco", "sculpture", "choreography"
        ])
        description = random.choice([
            "emotional depth through subtle variations",
            "multiple layers of meaning",
            "tension between opposing elements",
            "unity through repetition and variation",
            "the essence of human experience"
        ])
        
        entry = {
            "id": f"knowledge_{entry_id:03d}",
            "content": template.format(topic, description),
            "tags": ["art"] + base_tags,
            "difficulty": random.choice(["basic", "intermediate"]),
            "related_concepts": random.sample([
                "expression", "creativity", "beauty", "meaning", "form",
                "content", "technique", "style", "interpretation", "aesthetics"
            ], 3)
        }
        knowledge_entries.append(entry)
        entry_id += 1
    
    # Philosophy & Psychology (50 entries)
    for i in range(50):
        template, base_tags = random.choice(philosophy_psychology_templates)
        topic = random.choice([
            "consciousness", "free will", "identity", "morality", "perception",
            "memory", "emotion", "motivation", "personality", "intelligence",
            "existentialism", "utilitarianism", "determinism", "dualism", "empiricism"
        ])
        description = random.choice([
            "the nature of subjective experience",
            "how beliefs shape behavior",
            "the relationship between mind and reality",
            "individual differences in cognitive processing",
            "the foundations of human knowledge"
        ])
        
        entry = {
            "id": f"knowledge_{entry_id:03d}",
            "content": template.format(topic, description),
            "tags": ["philosophy"] + base_tags,
            "difficulty": random.choice(["intermediate", "advanced"]),
            "related_concepts": random.sample([
                "consciousness", "reality", "knowledge", "ethics", "mind",
                "behavior", "thought", "emotion", "reason", "experience"
            ], 3)
        }
        knowledge_entries.append(entry)
        entry_id += 1
    
    # Economics & Business (50 entries)
    for i in range(50):
        template, base_tags = random.choice(economics_business_templates)
        topic = random.choice([
            "supply chain", "competitive advantage", "market equilibrium", "inflation",
            "diversification", "innovation", "branding", "efficiency", "globalization",
            "sustainability", "disruption", "scalability", "liquidity", "valuation"
        ])
        description = random.choice([
            "optimize resource allocation",
            "create value for stakeholders",
            "respond to market dynamics",
            "balance risk and return",
            "adapt to changing conditions"
        ])
        
        entry = {
            "id": f"knowledge_{entry_id:03d}",
            "content": template.format(topic, description),
            "tags": ["economics"] + base_tags,
            "difficulty": random.choice(["basic", "intermediate", "advanced"]),
            "related_concepts": random.sample([
                "value", "efficiency", "growth", "competition", "innovation",
                "risk", "opportunity", "resources", "markets", "strategy"
            ], 3)
        }
        knowledge_entries.append(entry)
        entry_id += 1
    
    # Create the final structure
    knowledge_base = {
        "metadata": {
            "version": "1.0",
            "total_entries": len(knowledge_entries),
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "language": "en",
            "categories": {
                "science_technology": 100,
                "mathematics_logic": 75,
                "history_culture": 75,
                "daily_life": 75,
                "arts_literature": 75,
                "philosophy_psychology": 50,
                "economics_business": 50
            }
        },
        "knowledge_entries": knowledge_entries
    }
    
    # Save to file
    output_path = "../data/input/knowledge_base/knowledge_500.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(knowledge_entries)} knowledge entries")
    print(f"Saved to: {output_path}")
    
    # Validate distribution
    tag_counts = {}
    for entry in knowledge_entries:
        for tag in entry['tags']:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    print("\nTag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {tag}: {count}")

if __name__ == "__main__":
    generate_knowledge_base()