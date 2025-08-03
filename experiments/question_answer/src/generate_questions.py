#!/usr/bin/env python3
"""
Generate 100 questions with varying difficulty levels for the experiment
"""

import json
import random
from datetime import datetime

def generate_questions():
    """Generate 100 questions: 30 easy, 40 medium, 20 hard, 10 very hard"""
    
    # Easy questions (30) - Single knowledge reference
    easy_questions = [
        {
            "id": "question_001",
            "question": "What is the law of conservation of energy?",
            "difficulty": "easy",
            "expected_tags": ["science", "physics"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_002",
            "question": "Define photosynthesis and its main products.",
            "difficulty": "easy",
            "expected_tags": ["science", "biology"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_003",
            "question": "What is the Pythagorean theorem?",
            "difficulty": "easy",
            "expected_tags": ["mathematics", "geometry"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_004",
            "question": "Explain the basic principle of supply and demand.",
            "difficulty": "easy",
            "expected_tags": ["economics"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_005",
            "question": "What is cognitive dissonance?",
            "difficulty": "easy",
            "expected_tags": ["psychology"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_006",
            "question": "Describe the water cycle.",
            "difficulty": "easy",
            "expected_tags": ["science", "ecology"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_007",
            "question": "What is binary code used for?",
            "difficulty": "easy",
            "expected_tags": ["technology", "computer science"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_008",
            "question": "Define fermentation and give examples.",
            "difficulty": "easy",
            "expected_tags": ["science", "chemistry"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_009",
            "question": "What was the Renaissance?",
            "difficulty": "easy",
            "expected_tags": ["history", "culture"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        },
        {
            "id": "question_010",
            "question": "Explain the golden ratio.",
            "difficulty": "easy",
            "expected_tags": ["mathematics", "art"],
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        }
    ]
    
    # Generate more easy questions
    easy_templates = [
        "What is {}?",
        "Define {} in simple terms.",
        "Explain the basic concept of {}.",
        "What are the main characteristics of {}?",
        "Describe the process of {}.",
        "What is the purpose of {}?",
        "How does {} work?",
        "What are the key features of {}?",
        "Give an overview of {}.",
        "What is the definition of {}?"
    ]
    
    easy_topics = [
        ("neuroplasticity", ["science", "psychology"]),
        ("quantum entanglement", ["science", "physics"]),
        ("compound interest", ["economics", "mathematics"]),
        ("the greenhouse effect", ["science", "ecology"]),
        ("machine learning", ["technology", "AI"]),
        ("the immune system", ["science", "biology"]),
        ("game theory", ["mathematics", "economics"]),
        ("the Silk Road", ["history", "economics"]),
        ("blockchain technology", ["technology", "economics"]),
        ("ecosystems", ["science", "ecology"]),
        ("the Doppler effect", ["science", "physics"]),
        ("impressionism", ["art", "history"]),
        ("the scientific method", ["science", "methodology"]),
        ("fractals", ["mathematics", "nature"]),
        ("the placebo effect", ["psychology", "medicine"]),
        ("jazz music", ["music", "culture"]),
        ("the hero's journey", ["literature", "psychology"]),
        ("mindfulness meditation", ["psychology", "health"]),
        ("DNA replication", ["science", "biology"]),
        ("the printing press", ["history", "technology"])
    ]
    
    for i in range(20):  # Generate 20 more easy questions
        template = random.choice(easy_templates)
        topic, tags = easy_topics[i]
        easy_questions.append({
            "id": f"question_{len(easy_questions) + 1:03d}",
            "question": template.format(topic),
            "difficulty": "easy",
            "expected_tags": tags,
            "insight_potential": "low",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": False
            }
        })
    
    # Medium questions (40) - Integration of 2-3 knowledge items
    medium_questions = [
        {
            "id": "question_031",
            "question": "How does the law of conservation of energy apply to photosynthesis?",
            "difficulty": "medium",
            "expected_tags": ["science", "physics", "biology"],
            "insight_potential": "medium",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_032",
            "question": "Compare and contrast fermentation and cellular respiration.",
            "difficulty": "medium",
            "expected_tags": ["science", "chemistry", "biology"],
            "insight_potential": "medium",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": True
            }
        },
        {
            "id": "question_033",
            "question": "How might machine learning algorithms help optimize investment strategies?",
            "difficulty": "medium",
            "expected_tags": ["technology", "economics", "finance"],
            "insight_potential": "medium",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_034",
            "question": "What role did the printing press play in the Renaissance?",
            "difficulty": "medium",
            "expected_tags": ["history", "technology", "culture"],
            "insight_potential": "medium",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": True
            }
        },
        {
            "id": "question_035",
            "question": "How does neuroplasticity relate to learning and memory?",
            "difficulty": "medium",
            "expected_tags": ["science", "psychology", "biology"],
            "insight_potential": "medium",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": False,
                "depth": True
            }
        }
    ]
    
    # Generate more medium questions
    medium_templates = [
        "How does {} relate to {}?",
        "What are the connections between {} and {}?",
        "Compare {} with {} in terms of their impact on {}.",
        "How can {} principles be applied to {}?",
        "Explain the relationship between {} and {}.",
        "What happens when {} interacts with {}?",
        "How do {} and {} complement each other?",
        "What are the similarities and differences between {} and {}?",
        "How does understanding {} help us understand {}?",
        "In what ways does {} influence {}?"
    ]
    
    medium_combinations = [
        (["quantum mechanics", "consciousness"], ["science", "physics", "philosophy"]),
        (["game theory", "business strategy"], ["mathematics", "economics", "business"]),
        (["fractals", "natural patterns"], ["mathematics", "nature", "science"]),
        (["cognitive dissonance", "decision making"], ["psychology", "behavior"]),
        (["blockchain", "trust in society"], ["technology", "economics", "society"]),
        (["ecosystem balance", "economic systems"], ["ecology", "economics"]),
        (["impressionism", "perception of reality"], ["art", "psychology", "philosophy"]),
        (["DNA replication", "information theory"], ["biology", "technology", "science"]),
        (["the Silk Road", "cultural exchange"], ["history", "culture", "economics"]),
        (["mindfulness", "neuroplasticity"], ["psychology", "health", "science"]),
        (["compound interest", "exponential growth"], ["mathematics", "economics", "finance"]),
        (["the greenhouse effect", "industrial revolution"], ["science", "history", "ecology"]),
        (["jazz improvisation", "creativity"], ["music", "art", "psychology"]),
        (["the hero's journey", "personal development"], ["literature", "psychology", "culture"]),
        (["supply and demand", "ecosystem resources"], ["economics", "ecology", "science"]),
        (["binary code", "human language"], ["technology", "linguistics", "communication"]),
        (["the scientific method", "philosophical inquiry"], ["science", "philosophy", "methodology"]),
        (["placebo effect", "belief systems"], ["psychology", "medicine", "cognition"]),
        (["Renaissance art", "mathematical principles"], ["art", "mathematics", "history"]),
        (["immune system", "computer security"], ["biology", "technology", "systems"]),
        (["water cycle", "economic cycles"], ["science", "economics", "systems"]),
        (["Pythagorean theorem", "music theory"], ["mathematics", "music", "patterns"]),
        (["fermentation", "cultural traditions"], ["science", "culture", "daily_life"]),
        (["Doppler effect", "astronomical observations"], ["physics", "astronomy", "science"]),
        (["golden ratio", "architectural design"], ["mathematics", "art", "architecture"]),
        (["neuroplasticity", "education methods"], ["psychology", "education", "learning"]),
        (["quantum entanglement", "information transfer"], ["physics", "technology", "science"]),
        (["machine learning", "pattern recognition in nature"], ["technology", "nature", "patterns"]),
        (["game theory", "evolutionary biology"], ["mathematics", "biology", "strategy"]),
        (["blockchain", "historical record keeping"], ["technology", "history", "information"]),
        (["ecosystems", "business ecosystems"], ["ecology", "business", "systems"]),
        (["mindfulness", "performance optimization"], ["psychology", "performance", "health"]),
        (["fractals", "financial markets"], ["mathematics", "economics", "patterns"]),
        (["DNA", "digital storage"], ["biology", "technology", "information"]),
        (["jazz", "mathematical improvisation"], ["music", "mathematics", "creativity"])
    ]
    
    for i in range(35):  # Generate 35 more medium questions
        if i < len(medium_combinations):
            topics, tags = medium_combinations[i]
            # Select appropriate template based on number of topics
            if len(topics) == 2:
                template = random.choice([t for t in medium_templates if t.count('{}') == 2])
                question = template.format(topics[0], topics[1])
            else:
                template = random.choice([t for t in medium_templates if t.count('{}') == 3])
                question = template.format(topics[0], topics[1], topics[2])
        else:
            # Fallback for any additional questions needed
            template = random.choice([t for t in medium_templates if t.count('{}') == 2])
            topics = ["concept A", "concept B"]
            tags = ["general"]
            question = template.format(topics[0], topics[1])
            
        medium_questions.append({
            "id": f"question_{len(easy_questions) + len(medium_questions) + 1:03d}",
            "question": question,
            "difficulty": "medium",
            "expected_tags": tags,
            "insight_potential": "medium",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": True,
                "depth": True
            }
        })
    
    # Hard questions (20) - Creative integration of multiple knowledge
    hard_questions = [
        {
            "id": "question_071",
            "question": "Could cognitive dissonance be considered a form of psychological energy that drives behavioral change, similar to physical energy driving motion?",
            "difficulty": "hard",
            "expected_tags": ["psychology", "physics", "philosophy"],
            "insight_potential": "high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_072",
            "question": "How might the principles of ecosystem balance inform the design of sustainable economic systems?",
            "difficulty": "hard",
            "expected_tags": ["ecology", "economics", "systems"],
            "insight_potential": "high",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_073",
            "question": "In what ways is the spread of information through social networks similar to energy flow in biological ecosystems?",
            "difficulty": "hard",
            "expected_tags": ["technology", "ecology", "information"],
            "insight_potential": "high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_074",
            "question": "Can the concept of neuroplasticity be applied to organizational learning and adaptation?",
            "difficulty": "hard",
            "expected_tags": ["psychology", "business", "systems"],
            "insight_potential": "high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_075",
            "question": "How do fractals in nature reflect fundamental principles of information compression and efficiency?",
            "difficulty": "hard",
            "expected_tags": ["mathematics", "nature", "information"],
            "insight_potential": "high",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": True,
                "depth": True
            }
        }
    ]
    
    # Generate more hard questions
    hard_templates = [
        "If we consider {} as a form of {}, what new insights emerge about {}?",
        "How might combining {} with {} revolutionize our understanding of {}?",
        "What universal principles connect {}, {}, and {}?",
        "Could {} be the key to solving problems in {}?",
        "How do patterns in {} mirror those in {}, and what does this tell us?",
        "What would happen if we applied {} thinking to {}?",
        "In what ways might {} and {} be manifestations of the same underlying principle?",
        "How could insights from {} transform our approach to {}?",
        "What emergent properties arise when {} meets {}?",
        "Could the paradox of {} be resolved through understanding {}?"
    ]
    
    hard_combinations = [
        (["quantum entanglement", "human relationships", "connection"], ["physics", "psychology", "philosophy"]),
        (["machine learning", "evolutionary processes", "creativity"], ["technology", "biology", "innovation"]),
        (["musical harmony", "mathematical ratios", "cosmic order"], ["music", "mathematics", "philosophy"]),
        (["blockchain trust", "social contracts", "decentralization"], ["technology", "philosophy", "society"]),
        (["placebo effect", "quantum observation", "consciousness"], ["psychology", "physics", "philosophy"]),
        (["economic cycles", "natural rhythms", "time"], ["economics", "nature", "philosophy"]),
        (["artistic expression", "scientific discovery", "truth"], ["art", "science", "philosophy"]),
        (["game theory", "moral philosophy", "cooperation"], ["mathematics", "ethics", "society"]),
        (["fractals", "consciousness", "infinite complexity"], ["mathematics", "psychology", "philosophy"]),
        (["storytelling", "memory formation", "cultural evolution"], ["literature", "psychology", "culture"]),
        (["thermodynamics", "information theory", "life"], ["physics", "technology", "biology"]),
        (["Renaissance thinking", "modern innovation", "paradigm shifts"], ["history", "technology", "progress"]),
        (["ecosystem resilience", "psychological resilience", "system stability"], ["ecology", "psychology", "systems"]),
        (["jazz improvisation", "quantum uncertainty", "spontaneous order"], ["music", "physics", "emergence"]),
        (["golden ratio", "aesthetic perception", "natural selection"], ["mathematics", "art", "evolution"])
    ]
    
    for i in range(15):  # Generate 15 more hard questions
        template = random.choice(hard_templates)
        if i < len(hard_combinations):
            topics, tags = hard_combinations[i]
            question = template.format(*topics)
        else:
            topics = ["advanced concept A", "advanced concept B", "field C"]
            tags = ["interdisciplinary"]
            question = template.format(*topics)
            
        hard_questions.append({
            "id": f"question_{len(easy_questions) + len(medium_questions) + len(hard_questions) + 1:03d}",
            "question": question,
            "difficulty": "hard",
            "expected_tags": tags,
            "insight_potential": "high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        })
    
    # Very hard questions (10) - Revolutionary cross-domain connections
    very_hard_questions = [
        {
            "id": "question_091",
            "question": "If consciousness emerges from neural complexity similar to how temperature emerges from molecular motion, what does this imply about the nature of subjective experience and its measurement?",
            "difficulty": "very_hard",
            "expected_tags": ["philosophy", "physics", "psychology", "emergence"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_092",
            "question": "Could the principles of quantum superposition help us understand how humans hold contradictory beliefs simultaneously, and what would this mean for theories of decision-making?",
            "difficulty": "very_hard",
            "expected_tags": ["physics", "psychology", "philosophy", "cognition"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_093",
            "question": "If we view cultural evolution as an information processing system analogous to biological evolution, what new strategies for societal adaptation and innovation might we discover?",
            "difficulty": "very_hard",
            "expected_tags": ["culture", "biology", "information", "evolution"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_094",
            "question": "How might the mathematical concept of infinity help us understand the paradox of continuous identity despite constant physical and mental change?",
            "difficulty": "very_hard",
            "expected_tags": ["mathematics", "philosophy", "identity", "paradox"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_095",
            "question": "Could economic value creation be understood as a form of entropy reduction, and if so, what would this mean for sustainable development?",
            "difficulty": "very_hard",
            "expected_tags": ["economics", "physics", "sustainability", "systems"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_096",
            "question": "If artistic creativity and scientific discovery both involve pattern recognition and breaking, are they fundamentally the same cognitive process expressed in different domains?",
            "difficulty": "very_hard",
            "expected_tags": ["art", "science", "cognition", "creativity"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_097",
            "question": "How might understanding the collapse of ancient civilizations through the lens of complex systems theory inform our approach to modern global challenges?",
            "difficulty": "very_hard",
            "expected_tags": ["history", "systems", "complexity", "civilization"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_098",
            "question": "Could the principle of least action in physics be a universal organizing principle that explains efficiency in biological evolution, economic markets, and even artistic expression?",
            "difficulty": "very_hard",
            "expected_tags": ["physics", "biology", "economics", "art"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_099",
            "question": "If information is neither created nor destroyed but only transformed, does this suggest a new conservation law analogous to energy conservation, and what would be its implications?",
            "difficulty": "very_hard",
            "expected_tags": ["information", "physics", "philosophy", "conservation"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": False,
                "creativity": True,
                "depth": True
            }
        },
        {
            "id": "question_100",
            "question": "How might the interplay between order and chaos in natural systems provide a blueprint for balancing innovation and stability in human institutions?",
            "difficulty": "very_hard",
            "expected_tags": ["systems", "chaos", "innovation", "institutions"],
            "insight_potential": "very_high",
            "evaluation_criteria": {
                "accuracy": True,
                "creativity": True,
                "depth": True
            }
        }
    ]
    
    # Combine all questions
    all_questions = easy_questions + medium_questions + hard_questions + very_hard_questions
    
    # Create the final structure
    questions_data = {
        "metadata": {
            "version": "1.0",
            "total_questions": len(all_questions),
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "language": "en",
            "distribution": {
                "easy": 30,
                "medium": 40,
                "hard": 20,
                "very_hard": 10
            }
        },
        "questions": all_questions
    }
    
    # Save to file
    output_path = "../data/input/questions/questions_100.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(all_questions)} questions")
    print(f"Saved to: {output_path}")
    
    # Validate distribution
    difficulty_counts = {}
    for q in all_questions:
        diff = q['difficulty']
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print("\nDifficulty distribution:")
    for diff, count in difficulty_counts.items():
        print(f"  {diff}: {count}")

if __name__ == "__main__":
    generate_questions()