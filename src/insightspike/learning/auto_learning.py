"""
Layer1 Unknown Information Auto-Learning System
==============================================

This module implements automatic learning and database updating functionality
for unknown information discovered by Layer1 analysis. This mimics human-like
behavior of automatically storing new information encountered during reasoning.

Key Features:
- Automatic detection and storage of unknown concepts
- Learning context from successful query resolutions
- Building knowledge base incrementally through interaction
- Human-like memory formation and concept association
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class UnknownConcept:
    """Represents an unknown concept discovered during analysis"""

    concept: str
    context_query: str
    discovered_at: datetime
    confidence_score: float
    associated_concepts: List[str]
    resolution_attempts: int = 0
    learned_definition: Optional[str] = None
    source_context: Optional[str] = None
    learning_priority: float = 1.0


@dataclass
class LearningSession:
    """Represents a learning session where unknown concepts were resolved"""

    session_id: str
    query: str
    unknown_concepts: List[str]
    resolution_context: str
    success_score: float
    timestamp: datetime
    concepts_learned: List[str]


class Layer1AutoLearningSystem:
    """
    Automatic learning system that captures and stores unknown information
    discovered by Layer1 analysis, mimicking human-like learning behavior.
    """

    def __init__(self, storage_path: str = None):
        """Initialize the auto-learning system"""

        # Set up storage paths
        if storage_path is None:
            storage_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "learning"
            )

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Storage files
        self.unknown_concepts_file = self.storage_path / "unknown_concepts.json"
        self.learning_sessions_file = self.storage_path / "learning_sessions.json"
        self.concept_relationships_file = (
            self.storage_path / "concept_relationships.json"
        )
        self.auto_knowledge_base_file = self.storage_path / "auto_learned_knowledge.txt"

        # In-memory stores
        self.unknown_concepts: Dict[str, UnknownConcept] = {}
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.concept_relationships: Dict[str, Set[str]] = {}

        # Load existing data
        self._load_existing_data()

        logger.info(
            f"Layer1 Auto-Learning System initialized with {len(self.unknown_concepts)} unknown concepts"
        )

    def register_unknown_concepts(
        self,
        query: str,
        unknown_elements: List[str],
        associated_concepts: List[str] = None,
    ) -> List[str]:
        """
        Register newly discovered unknown concepts from Layer1 analysis.

        Args:
            query: The original query where concepts were discovered
            unknown_elements: List of unknown concepts/elements
            associated_concepts: Related known concepts for context

        Returns:
            List of concept IDs that were newly registered
        """
        newly_registered = []

        for concept in unknown_elements:
            concept_id = self._generate_concept_id(concept)

            if concept_id not in self.unknown_concepts:
                # Create new unknown concept entry
                unknown_concept = UnknownConcept(
                    concept=concept,
                    context_query=query,
                    discovered_at=datetime.now(),
                    confidence_score=0.8,  # Default confidence for discovery
                    associated_concepts=associated_concepts or [],
                    learning_priority=self._calculate_learning_priority(concept, query),
                )

                self.unknown_concepts[concept_id] = unknown_concept
                newly_registered.append(concept_id)

                # Update concept relationships
                self._update_concept_relationships(concept, associated_concepts or [])

                logger.info(
                    f"Registered new unknown concept: '{concept}' from query: '{query[:50]}...'"
                )

            else:
                # Update existing concept with new context
                existing = self.unknown_concepts[concept_id]
                existing.resolution_attempts += 1
                existing.associated_concepts.extend(associated_concepts or [])
                # Remove duplicates while preserving order
                existing.associated_concepts = list(
                    dict.fromkeys(existing.associated_concepts)
                )

                logger.debug(
                    f"Updated existing unknown concept: '{concept}' (attempts: {existing.resolution_attempts})"
                )

        # Save updates
        self._save_unknown_concepts()
        self._save_concept_relationships()

        return newly_registered

    def attempt_concept_resolution(
        self, query: str, response_context: str, quality_score: float
    ) -> Dict[str, Any]:
        """
        Attempt to resolve unknown concepts using successful query resolution context.

        Args:
            query: Original query that was resolved
            response_context: The successful response/context that might contain definitions
            quality_score: Quality score of the resolution (0.0-1.0)

        Returns:
            Dictionary with resolution results and learned concepts
        """
        resolution_results = {
            "concepts_resolved": [],
            "concepts_partially_learned": [],
            "new_knowledge_entries": [],
            "learning_session_id": None,
        }

        if quality_score < 0.5:
            logger.debug("Low quality response, skipping concept resolution")
            return resolution_results

        # Find unknown concepts mentioned in this query
        query_concepts = self._extract_concepts_from_text(query)
        unknown_in_query = [
            cid
            for cid, concept in self.unknown_concepts.items()
            if concept.concept.lower() in query.lower()
        ]

        if not unknown_in_query:
            return resolution_results

        # Create learning session
        session_id = self._generate_session_id(query)
        learning_session = LearningSession(
            session_id=session_id,
            query=query,
            unknown_concepts=[
                self.unknown_concepts[cid].concept for cid in unknown_in_query
            ],
            resolution_context=response_context,
            success_score=quality_score,
            timestamp=datetime.now(),
            concepts_learned=[],
        )

        # Attempt to extract definitions/knowledge from response
        for concept_id in unknown_in_query:
            concept = self.unknown_concepts[concept_id]

            # Try to extract relevant information about this concept
            learned_info = self._extract_concept_information(
                concept.concept, response_context
            )

            if learned_info:
                # Update concept with learned information
                concept.learned_definition = learned_info
                concept.source_context = response_context[:200] + "..."
                concept.confidence_score = min(
                    1.0, concept.confidence_score + quality_score * 0.3
                )

                resolution_results["concepts_resolved"].append(concept.concept)
                learning_session.concepts_learned.append(concept.concept)

                # Add to auto-learned knowledge base
                knowledge_entry = self._create_knowledge_entry(concept, learned_info)
                resolution_results["new_knowledge_entries"].append(knowledge_entry)
                self._append_to_knowledge_base(knowledge_entry)

                logger.info(
                    f"Learned about concept '{concept.concept}': {learned_info[:100]}..."
                )

        # Store learning session
        self.learning_sessions[session_id] = learning_session
        resolution_results["learning_session_id"] = session_id

        # Save updates
        self._save_learning_sessions()
        self._save_unknown_concepts()

        return resolution_results

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning process and discovered concepts"""

        total_concepts = len(self.unknown_concepts)
        resolved_concepts = len(
            [c for c in self.unknown_concepts.values() if c.learned_definition]
        )
        learning_rate = resolved_concepts / total_concepts if total_concepts > 0 else 0

        # Concept priority analysis
        high_priority_concepts = [
            c for c in self.unknown_concepts.values() if c.learning_priority > 0.7
        ]

        # Recent learning activity
        recent_sessions = [
            s
            for s in self.learning_sessions.values()
            if (datetime.now() - s.timestamp).days < 7
        ]

        insights = {
            "total_unknown_concepts": total_concepts,
            "resolved_concepts": resolved_concepts,
            "learning_rate": learning_rate,
            "high_priority_concepts": len(high_priority_concepts),
            "recent_learning_sessions": len(recent_sessions),
            "concept_relationships": len(self.concept_relationships),
            "top_priority_concepts": [
                c.concept
                for c in sorted(
                    high_priority_concepts,
                    key=lambda x: x.learning_priority,
                    reverse=True,
                )[:5]
            ],
            "knowledge_base_size": self._get_knowledge_base_size(),
        }

        return insights

    def export_learned_knowledge(self) -> str:
        """Export all learned knowledge as a formatted text for integration"""

        knowledge_lines = []
        knowledge_lines.append("# Auto-Learned Knowledge Base")
        knowledge_lines.append(
            f"# Generated by Layer1 Auto-Learning System on {datetime.now()}"
        )
        knowledge_lines.append(
            f"# Total concepts learned: {len([c for c in self.unknown_concepts.values() if c.learned_definition])}"
        )
        knowledge_lines.append("")

        # Group by learning priority
        resolved_concepts = [
            c for c in self.unknown_concepts.values() if c.learned_definition
        ]
        resolved_concepts.sort(key=lambda x: x.learning_priority, reverse=True)

        for concept in resolved_concepts:
            knowledge_lines.append(f"## {concept.concept}")
            knowledge_lines.append(f"**Definition**: {concept.learned_definition}")
            knowledge_lines.append(f"**Context**: {concept.source_context}")
            knowledge_lines.append(
                f"**Associated Concepts**: {', '.join(concept.associated_concepts[:5])}"
            )
            knowledge_lines.append(
                f"**Learning Priority**: {concept.learning_priority:.2f}"
            )
            knowledge_lines.append("")

        return "\n".join(knowledge_lines)

    # Private helper methods

    def _generate_concept_id(self, concept: str) -> str:
        """Generate unique ID for a concept"""
        return hashlib.md5(concept.lower().encode()).hexdigest()[:8]

    def _generate_session_id(self, query: str) -> str:
        """Generate unique ID for a learning session"""
        timestamp = str(int(time.time()))
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"session_{timestamp}_{query_hash}"

    def _calculate_learning_priority(self, concept: str, query: str) -> float:
        """Calculate learning priority for a concept"""

        # Base priority
        priority = 0.5

        # Higher priority for concepts that appear multiple times
        concept_mentions = self._count_concept_mentions(concept)
        priority += min(0.3, concept_mentions * 0.1)

        # Higher priority for technical terms
        if any(
            indicator in concept.lower()
            for indicator in ["quantum", "neural", "algorithm", "theory", "principle"]
        ):
            priority += 0.2

        # Higher priority for question-related concepts
        if any(
            q_word in query.lower()
            for q_word in ["how", "why", "what", "when", "where"]
        ):
            priority += 0.1

        return min(1.0, priority)

    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract concepts from text using simple NLP"""

        # Simple concept extraction (can be enhanced with NLP libraries)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Filter out common words
        common_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "man",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "its",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
        }

        concepts = [
            word for word in words if word not in common_words and len(word) > 3
        ]

        return list(set(concepts))  # Remove duplicates

    def _extract_concept_information(self, concept: str, context: str) -> Optional[str]:
        """Try to extract information about a concept from context"""

        concept_lower = concept.lower()
        context_lower = context.lower()

        # Look for definitions or explanations
        definition_patterns = [
            rf"{concept_lower} is (.*?)\.",
            rf"{concept_lower} refers to (.*?)\.",
            rf"{concept_lower} means (.*?)\.",
            rf"{concept_lower}: (.*?)\.",
            rf"{concept_lower} can be defined as (.*?)\.",
        ]

        for pattern in definition_patterns:
            match = re.search(pattern, context_lower)
            if match:
                definition = match.group(1).strip()
                if len(definition) > 10 and len(definition) < 200:  # Reasonable length
                    return definition

        # Look for descriptive context
        sentences = re.split(r"[.!?]", context)
        for sentence in sentences:
            if concept_lower in sentence.lower() and len(sentence.strip()) > 20:
                return sentence.strip()

        return None

    def _create_knowledge_entry(
        self, concept: UnknownConcept, learned_info: str
    ) -> str:
        """Create a formatted knowledge entry"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] {concept.concept}: {learned_info} (Priority: {concept.learning_priority:.2f})"

    def _append_to_knowledge_base(self, entry: str):
        """Append an entry to the auto-learned knowledge base file"""
        with open(self.auto_knowledge_base_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def _update_concept_relationships(
        self, concept: str, associated_concepts: List[str]
    ):
        """Update concept relationship mappings"""
        if concept not in self.concept_relationships:
            self.concept_relationships[concept] = set()

        for assoc in associated_concepts:
            self.concept_relationships[concept].add(assoc)
            # Bidirectional relationship
            if assoc not in self.concept_relationships:
                self.concept_relationships[assoc] = set()
            self.concept_relationships[assoc].add(concept)

    def _count_concept_mentions(self, concept: str) -> int:
        """Count how many times a concept has been mentioned"""
        return sum(
            1
            for c in self.unknown_concepts.values()
            if c.concept.lower() == concept.lower()
        )

    def _get_knowledge_base_size(self) -> int:
        """Get the size of the auto-learned knowledge base"""
        if self.auto_knowledge_base_file.exists():
            with open(self.auto_knowledge_base_file, "r", encoding="utf-8") as f:
                return len(f.readlines())
        return 0

    # Data persistence methods

    def _load_existing_data(self):
        """Load existing data from storage files"""

        # Load unknown concepts
        if self.unknown_concepts_file.exists():
            try:
                with open(self.unknown_concepts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for cid, concept_data in data.items():
                        # Convert datetime strings back to datetime objects
                        concept_data["discovered_at"] = datetime.fromisoformat(
                            concept_data["discovered_at"]
                        )
                        self.unknown_concepts[cid] = UnknownConcept(**concept_data)
            except Exception as e:
                logger.warning(f"Failed to load unknown concepts: {e}")

        # Load learning sessions
        if self.learning_sessions_file.exists():
            try:
                with open(self.learning_sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sid, session_data in data.items():
                        session_data["timestamp"] = datetime.fromisoformat(
                            session_data["timestamp"]
                        )
                        self.learning_sessions[sid] = LearningSession(**session_data)
            except Exception as e:
                logger.warning(f"Failed to load learning sessions: {e}")

        # Load concept relationships
        if self.concept_relationships_file.exists():
            try:
                with open(self.concept_relationships_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.concept_relationships = {k: set(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load concept relationships: {e}")

    def _save_unknown_concepts(self):
        """Save unknown concepts to storage"""
        try:
            data = {}
            for cid, concept in self.unknown_concepts.items():
                concept_dict = asdict(concept)
                concept_dict["discovered_at"] = concept.discovered_at.isoformat()
                data[cid] = concept_dict

            with open(self.unknown_concepts_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save unknown concepts: {e}")

    def _save_learning_sessions(self):
        """Save learning sessions to storage"""
        try:
            data = {}
            for sid, session in self.learning_sessions.items():
                session_dict = asdict(session)
                session_dict["timestamp"] = session.timestamp.isoformat()
                data[sid] = session_dict

            with open(self.learning_sessions_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save learning sessions: {e}")

    def _save_concept_relationships(self):
        """Save concept relationships to storage"""
        try:
            data = {k: list(v) for k, v in self.concept_relationships.items()}
            with open(self.concept_relationships_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save concept relationships: {e}")
