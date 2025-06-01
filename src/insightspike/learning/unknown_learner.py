"""
Unknown Information Learning System
==================================

Human-like learning system that automatically registers unknown information discovered
by Layer1 analysis with weak relationships, then uses sleep-mode cleanup to prune
low-confidence connections, preventing graph explosion while enabling gradual learning.

This mimics human memory formation: weak initial associations → sleep consolidation → 
long-term memory formation.
"""

import time
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class WeakRelationship:
    """Represents a weak relationship between concepts"""
    concept1: str
    concept2: str
    confidence: float
    source: str  # 'co_occurrence', 'reinforcement', 'inference'
    usage_count: int
    created_at: float
    last_accessed: float
    context: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeakRelationship':
        return cls(**data)

class UnknownLearner:
    """
    Manages learning of unknown information with weak relationships and sleep-mode cleanup.
    
    Key Features:
    - Registers weak relationships from co-occurrence in questions
    - Automatic sleep-mode cleanup of low-confidence relationships
    - Gradual reinforcement of repeatedly encountered relationships
    - Graph explosion prevention through natural pruning
    """
    
    # Learning Parameters (will be tuned through experimentation)
    INITIAL_CONFIDENCE = 0.1        # Very weak initial relationship
    CLEANUP_THRESHOLD = 0.15        # Minimum confidence to survive cleanup
    SLEEP_INTERVAL = 30             # Seconds between cleanup cycles
    REINFORCEMENT_RATE = 0.05       # Confidence increase per reinforcement
    MAX_WEAK_EDGES = 1000          # Maximum number of weak relationships
    USAGE_DECAY_RATE = 0.02        # Confidence decay for unused relationships
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the unknown learner with persistent storage"""
        if db_path is None:
            from ..core.config import get_config
            config = get_config()
            db_path = config.paths.root_dir / "data" / "unknown_learning.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for active relationships
        self.weak_relationships: Dict[Tuple[str, str], WeakRelationship] = {}
        self.concept_frequency: Dict[str, int] = defaultdict(int)
        
        # Sleep mode management
        self._sleep_thread = None
        self._sleep_active = False
        self._last_activity = time.time()
        
        # Initialize database
        self._init_database()
        self._load_from_database()
        
        logger.info(f"UnknownLearner initialized with {len(self.weak_relationships)} existing relationships")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            # Create relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weak_relationships (
                    id INTEGER PRIMARY KEY,
                    concept1 TEXT NOT NULL,
                    concept2 TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    context TEXT DEFAULT '',
                    UNIQUE(concept1, concept2)
                )
            """)
            
            # Create concept statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concept_stats (
                    concept TEXT PRIMARY KEY,
                    frequency INTEGER DEFAULT 0,
                    last_seen REAL NOT NULL
                )
            """)
            
            # Create indexes for performance
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON weak_relationships(confidence)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON weak_relationships(last_accessed)")
            except:
                pass  # Indexes may already exist
            
            conn.commit()
    
    def _load_from_database(self):
        """Load existing relationships from database"""
        with sqlite3.connect(self.db_path) as conn:
            # Load weak relationships
            cursor = conn.execute("""
                SELECT concept1, concept2, confidence, source, usage_count, 
                       created_at, last_accessed, context
                FROM weak_relationships
            """)
            
            for row in cursor.fetchall():
                rel = WeakRelationship(
                    concept1=row[0], concept2=row[1], confidence=row[2],
                    source=row[3], usage_count=row[4], created_at=row[5],
                    last_accessed=row[6], context=row[7]
                )
                key = self._make_relationship_key(row[0], row[1])
                self.weak_relationships[key] = rel
            
            # Load concept frequencies
            cursor = conn.execute("SELECT concept, frequency FROM concept_stats")
            for concept, freq in cursor.fetchall():
                self.concept_frequency[concept] = freq
    
    def _make_relationship_key(self, concept1: str, concept2: str) -> Tuple[str, str]:
        """Create normalized key for relationship (alphabetical order)"""
        return tuple(sorted([concept1.lower().strip(), concept2.lower().strip()]))
    
    def register_question_relationships(self, known_elements: List[str], 
                                      unknown_elements: List[str], 
                                      question_context: str = "") -> List[WeakRelationship]:
        """
        Register weak relationships from co-occurrence in a question.
        
        This is the main entry point from Layer1 analysis.
        """
        self._last_activity = time.time()
        new_relationships = []
        
        # Register all possible pairs between known and unknown elements
        all_elements = known_elements + unknown_elements
        
        for i, concept1 in enumerate(all_elements):
            for concept2 in all_elements[i+1:]:
                if concept1 != concept2:
                    rel = self._register_weak_relationship(
                        concept1, concept2, 
                        source="co_occurrence_in_question",
                        context=question_context[:100]  # Limit context length
                    )
                    if rel:
                        new_relationships.append(rel)
        
        # Update concept frequencies
        for concept in all_elements:
            self.concept_frequency[concept] += 1
        
        # Start sleep mode if not already running
        self._ensure_sleep_mode_active()
        
        logger.info(f"Registered {len(new_relationships)} new weak relationships from question")
        return new_relationships
    
    def _register_weak_relationship(self, concept1: str, concept2: str, 
                                   source: str, context: str = "") -> Optional[WeakRelationship]:
        """Register or reinforce a weak relationship between concepts"""
        key = self._make_relationship_key(concept1, concept2)
        current_time = time.time()
        
        if key in self.weak_relationships:
            # Reinforce existing relationship
            rel = self.weak_relationships[key]
            rel.confidence = min(0.95, rel.confidence + self.REINFORCEMENT_RATE)
            rel.usage_count += 1
            rel.last_accessed = current_time
            logger.debug(f"Reinforced relationship: {concept1} <-> {concept2} (conf: {rel.confidence:.3f})")
            
            # Update database
            self._update_relationship_in_db(rel)
            return rel
        else:
            # Check if we're at maximum capacity
            if len(self.weak_relationships) >= self.MAX_WEAK_EDGES:
                logger.warning(f"Maximum weak relationships reached ({self.MAX_WEAK_EDGES}), skipping new registration")
                return None
            
            # Create new weak relationship
            rel = WeakRelationship(
                concept1=key[0], concept2=key[1],
                confidence=self.INITIAL_CONFIDENCE,
                source=source,
                usage_count=1,
                created_at=current_time,
                last_accessed=current_time,
                context=context
            )
            
            self.weak_relationships[key] = rel
            logger.debug(f"New weak relationship: {concept1} <-> {concept2} (conf: {rel.confidence:.3f})")
            
            # Store in database
            self._store_relationship_in_db(rel)
            return rel
    
    def _store_relationship_in_db(self, rel: WeakRelationship):
        """Store new relationship in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO weak_relationships 
                (concept1, concept2, confidence, source, usage_count, created_at, last_accessed, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (rel.concept1, rel.concept2, rel.confidence, rel.source, 
                  rel.usage_count, rel.created_at, rel.last_accessed, rel.context))
            conn.commit()
    
    def _update_relationship_in_db(self, rel: WeakRelationship):
        """Update existing relationship in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE weak_relationships 
                SET confidence = ?, usage_count = ?, last_accessed = ?
                WHERE concept1 = ? AND concept2 = ?
            """, (rel.confidence, rel.usage_count, rel.last_accessed, 
                  rel.concept1, rel.concept2))
            conn.commit()
    
    def _ensure_sleep_mode_active(self):
        """Ensure sleep mode cleanup thread is running"""
        if not self._sleep_active:
            self._sleep_active = True
            self._sleep_thread = threading.Thread(target=self._sleep_mode_loop, daemon=True)
            self._sleep_thread.start()
            logger.info("Sleep mode cleanup activated")
    
    def _sleep_mode_loop(self):
        """Main sleep mode loop - runs in background thread"""
        while self._sleep_active:
            time.sleep(self.SLEEP_INTERVAL)
            
            # Check if we've been idle long enough
            idle_time = time.time() - self._last_activity
            if idle_time >= self.SLEEP_INTERVAL:
                cleanup_stats = self._run_sleep_cleanup()
                logger.info(f"Sleep cleanup: removed {cleanup_stats['removed']}, "
                           f"decayed {cleanup_stats['decayed']}, "
                           f"total relationships: {len(self.weak_relationships)}")
    
    def _run_sleep_cleanup(self) -> Dict[str, int]:
        """
        Run sleep mode cleanup to remove low-confidence relationships.
        
        This prevents graph explosion by naturally pruning unused relationships.
        """
        removed_count = 0
        decayed_count = 0
        current_time = time.time()
        
        # Find relationships to remove
        to_remove = []
        for key, rel in self.weak_relationships.items():
            # Apply usage decay
            if rel.usage_count == 0:
                time_since_creation = current_time - rel.created_at
                if time_since_creation > 3600:  # 1 hour decay grace period
                    rel.confidence = max(0.0, rel.confidence - self.USAGE_DECAY_RATE)
                    decayed_count += 1
            
            # Mark for removal if below threshold
            if rel.confidence < self.CLEANUP_THRESHOLD:
                to_remove.append(key)
        
        # Remove low-confidence relationships
        for key in to_remove:
            del self.weak_relationships[key]
            removed_count += 1
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM weak_relationships 
                    WHERE concept1 = ? AND concept2 = ?
                """, key)
                conn.commit()
        
        return {'removed': removed_count, 'decayed': decayed_count}
    
    def get_related_concepts(self, concept: str, min_confidence: float = 0.2) -> List[Tuple[str, float]]:
        """Get concepts related to the given concept above minimum confidence"""
        concept_lower = concept.lower().strip()
        related = []
        
        for (c1, c2), rel in self.weak_relationships.items():
            if rel.confidence >= min_confidence:
                if c1 == concept_lower:
                    related.append((c2, rel.confidence))
                elif c2 == concept_lower:
                    related.append((c1, rel.confidence))
        
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system (alias for get_learning_stats)"""
        return self.get_learning_stats()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        if not self.weak_relationships:
            return {'total_relationships': 0, 'avg_confidence': 0.0}
        
        confidences = [rel.confidence for rel in self.weak_relationships.values()]
        strong_relationships = sum(1 for c in confidences if c >= 0.5)
        weak_relationships = sum(1 for c in confidences if c < 0.2)
        
        return {
            'total_relationships': len(self.weak_relationships),
            'strong_relationships': strong_relationships,
            'weak_relationships': weak_relationships,
            'avg_confidence': sum(confidences) / len(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'total_concepts': len(self.concept_frequency),
            'most_frequent_concepts': sorted(self.concept_frequency.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]
        }
    
    def force_cleanup(self) -> Dict[str, int]:
        """Force immediate cleanup (for testing/maintenance)"""
        return self._run_sleep_cleanup()
    
    def shutdown(self):
        """Gracefully shutdown the learning system"""
        self._sleep_active = False
        if self._sleep_thread and self._sleep_thread.is_alive():
            self._sleep_thread.join(timeout=5)
        
        # Final save to database
        for rel in self.weak_relationships.values():
            self._update_relationship_in_db(rel)
        
        # Update concept frequencies
        with sqlite3.connect(self.db_path) as conn:
            for concept, freq in self.concept_frequency.items():
                conn.execute("""
                    INSERT OR REPLACE INTO concept_stats (concept, frequency, last_seen)
                    VALUES (?, ?, ?)
                """, (concept, freq, time.time()))
            conn.commit()
        
        logger.info("UnknownLearner shutdown complete")

# Global instance for easy access
_global_learner: Optional[UnknownLearner] = None

def get_unknown_learner() -> UnknownLearner:
    """Get or create global unknown learner instance"""
    global _global_learner
    if _global_learner is None:
        _global_learner = UnknownLearner()
    return _global_learner

def shutdown_unknown_learner():
    """Shutdown global unknown learner"""
    global _global_learner
    if _global_learner is not None:
        _global_learner.shutdown()
        _global_learner = None
