"""
Generate expanded test set with 100+ questions.
Balanced difficulty distribution for statistical validity.
"""

import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class Question:
    """Question with metadata."""
    id: str
    text: str
    difficulty: str  # easy, medium, hard
    category: str    # type of reasoning
    expected_associations: List[str]
    complexity_features: Dict[str, Any]


class ExpandedQuestionGenerator:
    """Generate diverse questions for comprehensive evaluation."""
    
    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducibility."""
        random.seed(seed)
        self.categories = [
            'mathematical', 'scientific', 'linguistic',
            'logical', 'conceptual', 'analogical'
        ]
        
    def generate_questions(self, 
                          n_easy: int = 25,
                          n_medium: int = 50,
                          n_hard: int = 25) -> List[Question]:
        """
        Generate balanced question set.
        
        Args:
            n_easy: Number of easy questions
            n_medium: Number of medium questions
            n_hard: Number of hard questions
            
        Returns:
            List of Question objects
        """
        questions = []
        
        # Generate easy questions
        questions.extend(self._generate_easy_questions(n_easy))
        
        # Generate medium questions
        questions.extend(self._generate_medium_questions(n_medium))
        
        # Generate hard questions
        questions.extend(self._generate_hard_questions(n_hard))
        
        # Shuffle to avoid difficulty ordering bias
        random.shuffle(questions)
        
        return questions
    
    def _generate_easy_questions(self, n: int) -> List[Question]:
        """Generate easy questions requiring single-hop reasoning."""
        templates = [
            # Mathematical
            ("What is {num1} + {num2}?", "mathematical", 
             lambda p: [f"addition", f"sum", str(p['num1'] + p['num2'])]),
            
            ("What is {num1} × {num2}?", "mathematical",
             lambda p: [f"multiplication", f"product", str(p['num1'] * p['num2'])]),
            
            # Scientific
            ("What is the chemical symbol for {element}?", "scientific",
             lambda p: [p['element'], "chemistry", p['symbol']]),
            
            ("What state of matter is {substance} at room temperature?", "scientific",
             lambda p: [p['substance'], "state", p['state']]),
            
            # Linguistic
            ("What is the plural of '{word}'?", "linguistic",
             lambda p: [p['word'], "plural", p['plural']]),
            
            ("What part of speech is '{word}'?", "linguistic",
             lambda p: [p['word'], "grammar", p['pos']]),
            
            # Logical
            ("If all {A} are {B}, and X is a {A}, what is X?", "logical",
             lambda p: [p['A'], p['B'], "syllogism"]),
            
            # Conceptual
            ("What color is a {object}?", "conceptual",
             lambda p: [p['object'], "color", p['color']])
        ]
        
        questions = []
        for i in range(n):
            template, category, assoc_fn = random.choice(templates)
            params = self._get_easy_params(template)
            
            question = Question(
                id=f"easy_{i+1:03d}",
                text=template.format(**params),
                difficulty="easy",
                category=category,
                expected_associations=assoc_fn(params),
                complexity_features={
                    'hops': 1,
                    'abstract': False,
                    'requires_combination': False
                }
            )
            questions.append(question)
        
        return questions
    
    def _generate_medium_questions(self, n: int) -> List[Question]:
        """Generate medium questions requiring multi-hop reasoning."""
        templates = [
            # Mathematical with fractions/decimals
            ("What is {num1} × {decimal}?", "mathematical",
             lambda p: ["multiplication", "decimal", str(p['num1'] * p['decimal'])]),
            
            ("What is {fraction1} + {fraction2}?", "mathematical",
             lambda p: ["fraction", "addition", "common denominator"]),
            
            # Scientific with relationships
            ("How does {property} change when {substance} is {action}?", "scientific",
             lambda p: [p['property'], p['substance'], p['action'], "relationship"]),
            
            ("What happens to {organism} in {environment}?", "scientific",
             lambda p: [p['organism'], p['environment'], "adaptation"]),
            
            # Linguistic with context
            ("What does '{idiom}' mean?", "linguistic",
             lambda p: [p['idiom'], "figurative", p['meaning']]),
            
            # Analogical
            ("{A} is to {B} as {C} is to what?", "analogical",
             lambda p: ["analogy", p['A'], p['B'], p['C'], p['D']]),
            
            # Logical with conditions
            ("If {condition1} and {condition2}, then what?", "logical",
             lambda p: ["conditional", "logic", p['conclusion']])
        ]
        
        questions = []
        for i in range(n):
            template, category, assoc_fn = random.choice(templates)
            params = self._get_medium_params(template)
            
            question = Question(
                id=f"medium_{i+1:03d}",
                text=template.format(**params),
                difficulty="medium",
                category=category,
                expected_associations=assoc_fn(params),
                complexity_features={
                    'hops': 2,
                    'abstract': True,
                    'requires_combination': True
                }
            )
            questions.append(question)
        
        return questions
    
    def _generate_hard_questions(self, n: int) -> List[Question]:
        """Generate hard questions requiring abstract/creative reasoning."""
        templates = [
            # Abstract mathematical
            ("If operation ★ is defined as a★b = {formula}, what is {expr}?", "mathematical",
             lambda p: ["custom operation", "abstract", "pattern"]),
            
            # Complex scientific
            ("Explain the relationship between {concept1}, {concept2}, and {concept3}", "scientific",
             lambda p: [p['concept1'], p['concept2'], p['concept3'], "systems thinking"]),
            
            # Meta-linguistic
            ("Why might '{word1}' and '{word2}' be considered {relationship}?", "linguistic",
             lambda p: ["etymology", "semantics", p['relationship']]),
            
            # Creative analogical
            ("Create an analogy between {abstract} and {concrete}", "analogical",
             lambda p: ["creativity", p['abstract'], p['concrete'], "mapping"]),
            
            # Paradoxical
            ("Resolve the apparent contradiction: {statement1} yet {statement2}", "logical",
             lambda p: ["paradox", "resolution", "perspective"])
        ]
        
        questions = []
        for i in range(n):
            template, category, assoc_fn = random.choice(templates)
            params = self._get_hard_params(template)
            
            question = Question(
                id=f"hard_{i+1:03d}",
                text=template.format(**params),
                difficulty="hard",
                category=category,
                expected_associations=assoc_fn(params),
                complexity_features={
                    'hops': 3,
                    'abstract': True,
                    'requires_combination': True,
                    'requires_creativity': True
                }
            )
            questions.append(question)
        
        return questions
    
    def _get_easy_params(self, template: str) -> Dict[str, Any]:
        """Get parameters for easy questions."""
        param_sets = {
            "num1": random.randint(1, 10),
            "num2": random.randint(1, 10),
            "element": random.choice([
                {"element": "gold", "symbol": "Au"},
                {"element": "silver", "symbol": "Ag"},
                {"element": "iron", "symbol": "Fe"},
                {"element": "oxygen", "symbol": "O"},
                {"element": "carbon", "symbol": "C"}
            ]),
            "substance": random.choice([
                {"substance": "water", "state": "liquid"},
                {"substance": "ice", "state": "solid"},
                {"substance": "steam", "state": "gas"},
                {"substance": "iron", "state": "solid"},
                {"substance": "oxygen", "state": "gas"}
            ]),
            "word": random.choice([
                {"word": "cat", "plural": "cats", "pos": "noun"},
                {"word": "run", "plural": "runs", "pos": "verb"},
                {"word": "happy", "plural": "happies", "pos": "adjective"},
                {"word": "child", "plural": "children", "pos": "noun"},
                {"word": "mouse", "plural": "mice", "pos": "noun"}
            ]),
            "A": random.choice(["dogs", "birds", "flowers"]),
            "B": random.choice(["animals", "creatures", "living things"]),
            "object": random.choice([
                {"object": "apple", "color": "red"},
                {"object": "banana", "color": "yellow"},
                {"object": "sky", "color": "blue"},
                {"object": "grass", "color": "green"},
                {"object": "snow", "color": "white"}
            ])
        }
        
        # Extract appropriate params based on template
        if "num1" in template:
            return {"num1": param_sets["num1"], "num2": param_sets["num2"]}
        elif "element" in template:
            return param_sets["element"]
        elif "substance" in template:
            return param_sets["substance"]
        elif "word" in template and "plural" in template:
            data = param_sets["word"]
            return {"word": data["word"], "plural": data["plural"]}
        elif "word" in template and "speech" in template:
            data = param_sets["word"]
            return {"word": data["word"], "pos": data["pos"]}
        elif "{A}" in template:
            return {"A": param_sets["A"], "B": param_sets["B"]}
        elif "object" in template:
            return param_sets["object"]
        
        return {}
    
    def _get_medium_params(self, template: str) -> Dict[str, Any]:
        """Get parameters for medium questions."""
        if "decimal" in template:
            return {
                "num1": random.randint(1, 10),
                "decimal": round(random.uniform(0.1, 0.9), 1)
            }
        elif "fraction" in template:
            return {
                "fraction1": f"{random.randint(1,5)}/{random.randint(2,8)}",
                "fraction2": f"{random.randint(1,5)}/{random.randint(2,8)}"
            }
        elif "property" in template:
            return {
                "property": random.choice(["volume", "density", "temperature"]),
                "substance": random.choice(["water", "metal", "gas"]),
                "action": random.choice(["heated", "cooled", "compressed"])
            }
        elif "organism" in template:
            return {
                "organism": random.choice(["cactus", "penguin", "fish"]),
                "environment": random.choice(["desert", "arctic", "deep ocean"])
            }
        elif "idiom" in template:
            idioms = [
                {"idiom": "break the ice", "meaning": "start a conversation"},
                {"idiom": "piece of cake", "meaning": "very easy"},
                {"idiom": "spill the beans", "meaning": "reveal a secret"}
            ]
            return random.choice(idioms)
        elif "{A}" in template and "{C}" in template:
            analogies = [
                {"A": "teacher", "B": "student", "C": "doctor", "D": "patient"},
                {"A": "painter", "B": "canvas", "C": "writer", "D": "paper"},
                {"A": "seed", "B": "tree", "C": "egg", "D": "bird"}
            ]
            return random.choice(analogies)
        elif "condition" in template:
            return {
                "condition1": "it rains",
                "condition2": "the ground is dry",
                "conclusion": "the rain will wet the ground"
            }
        
        return {}
    
    def _get_hard_params(self, template: str) -> Dict[str, Any]:
        """Get parameters for hard questions."""
        if "formula" in template:
            return {
                "formula": "2a + b - 1",
                "expr": "3★4"
            }
        elif "concept1" in template:
            return {
                "concept1": "entropy",
                "concept2": "information",
                "concept3": "complexity"
            }
        elif "word1" in template:
            return {
                "word1": "understand",
                "word2": "stand under",
                "relationship": "etymologically related"
            }
        elif "abstract" in template and "concrete" in template:
            return {
                "abstract": "democracy",
                "concrete": "a garden"
            }
        elif "statement1" in template:
            return {
                "statement1": "the more you learn, the more you realize you don't know",
                "statement2": "knowledge is power"
            }
        
        return {}
    
    def save_questions(self, questions: List[Question], filepath: str):
        """Save questions to JSON file."""
        data = {
            'metadata': {
                'total_questions': len(questions),
                'difficulty_distribution': {
                    'easy': len([q for q in questions if q.difficulty == 'easy']),
                    'medium': len([q for q in questions if q.difficulty == 'medium']),
                    'hard': len([q for q in questions if q.difficulty == 'hard'])
                },
                'categories': list(set(q.category for q in questions))
            },
            'questions': [asdict(q) for q in questions]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Generate test set
if __name__ == "__main__":
    generator = ExpandedQuestionGenerator()
    questions = generator.generate_questions(n_easy=25, n_medium=50, n_hard=25)
    
    print(f"Generated {len(questions)} questions")
    print(f"Categories: {set(q.category for q in questions)}")
    print(f"Difficulty distribution:")
    for difficulty in ['easy', 'medium', 'hard']:
        count = len([q for q in questions if q.difficulty == difficulty])
        print(f"  {difficulty}: {count}")