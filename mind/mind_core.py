"""Core Mind class that manages the sub-neural networks.

This module implements the central coordinator for all neural networks,
managing communication, development, and the overall state of the artificial mind.
"""

import json
import logging
import os
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field

from communication.message_bus import GlobalMessageBus, MessageFilter
from config import config
from core.neural_network import NeuralNetwork
from core.schemas import (
    Belief,
    DevelopmentalStage,
    Memory,
    Need,
    NetworkMessage,
)
from mind.schemas import (
    Emotion,
    EmotionType,
    LanguageAbility,
    MindState,
    ObservableState,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MemoryCluster(BaseModel):
    """Cluster of related memories forming a coherent concept or experience."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str = Field(..., description="Human-readable label for this cluster")
    memory_ids: List[str] = Field(default_factory=list, description="IDs of memories in this cluster")
    centroid: Optional[List[float]] = Field(default=None, description="Vector representation of cluster center")
    creation_time: datetime = Field(default_factory=datetime.now)
    last_access_time: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance of this cluster")
    developmental_stage: DevelopmentalStage = Field(default=DevelopmentalStage.INFANT)

    class Config:
        arbitrary_types_allowed = True

    def access(self) -> None:
        """Access the memory cluster, updating its timestamp."""
        self.last_access_time = datetime.now()

    def add_memory(self, memory_id: str) -> None:
        """Add a memory to this cluster."""
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)

    def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory from this cluster."""
        if memory_id in self.memory_ids:
            self.memory_ids.remove(memory_id)
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "memory_ids": self.memory_ids,
            "centroid": self.centroid,
            "creation_time": self.creation_time.isoformat(),
            "last_access_time": self.last_access_time.isoformat(),
            "importance": self.importance,
            "developmental_stage": self.developmental_stage.name,
        }

class BeliefNetwork(BaseModel):
    """Network of beliefs and their relationships."""

    beliefs: Dict[str, Belief] = Field(default_factory=dict, description="Beliefs by ID")
    belief_relationships: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Relationships between beliefs (id -> {id -> strength})",
    )
    evidence_index: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Index of memories that support beliefs (memory_id -> [belief_ids])",
    )
    contradictions: List[Tuple[str, str, float]] = Field(
        default_factory=list,
        description="Pairs of contradicting beliefs with strength (id1, id2, strength)",
    )

    class Config:
        arbitrary_types_allowed = True

    def add_belief(self, belief: Belief) -> str:
        """Add a belief to the network.
        
        Args:
            belief: Belief to add
            
        Returns:
            ID of the added belief

        """
        belief_id = belief.id

        # Add to beliefs dictionary
        self.beliefs[belief_id] = belief

        # Initialize relationships dictionary
        if belief_id not in self.belief_relationships:
            self.belief_relationships[belief_id] = {}

        # Add to evidence index
        for memory_id in belief.supporting_memories:
            if memory_id not in self.evidence_index:
                self.evidence_index[memory_id] = []
            if belief_id not in self.evidence_index[memory_id]:
                self.evidence_index[memory_id].append(belief_id)

        return belief_id

    def remove_belief(self, belief_id: str) -> bool:
        """Remove a belief from the network.
        
        Args:
            belief_id: ID of belief to remove
            
        Returns:
            True if removed, False if not found

        """
        if belief_id not in self.beliefs:
            return False

        # Remove from beliefs dictionary
        belief = self.beliefs.pop(belief_id)

        # Remove from relationships
        if belief_id in self.belief_relationships:
            del self.belief_relationships[belief_id]

        # Remove references to this belief from other relationships
        for other_id in self.belief_relationships:
            if belief_id in self.belief_relationships[other_id]:
                del self.belief_relationships[other_id][belief_id]

        # Remove from evidence index
        for memory_id in belief.supporting_memories:
            if memory_id in self.evidence_index and belief_id in self.evidence_index[memory_id]:
                self.evidence_index[memory_id].remove(belief_id)
                if not self.evidence_index[memory_id]:
                    del self.evidence_index[memory_id]

        # Remove from contradictions
        self.contradictions = [
            (id1, id2, strength) for id1, id2, strength in self.contradictions
            if id1 != belief_id and id2 != belief_id
        ]

        return True

    def add_relationship(self, belief_id1: str, belief_id2: str, strength: float) -> bool:
        """Add a relationship between two beliefs.
        
        Args:
            belief_id1: ID of first belief
            belief_id2: ID of second belief
            strength: Relationship strength (0.0 to 1.0)
            
        Returns:
            True if added, False if error

        """
        if belief_id1 not in self.beliefs or belief_id2 not in self.beliefs:
            return False

        # Ensure relationships dictionaries exist
        if belief_id1 not in self.belief_relationships:
            self.belief_relationships[belief_id1] = {}
        if belief_id2 not in self.belief_relationships:
            self.belief_relationships[belief_id2] = {}

        # Add bidirectional relationship
        self.belief_relationships[belief_id1][belief_id2] = strength
        self.belief_relationships[belief_id2][belief_id1] = strength

        return True

    def add_contradiction(self, belief_id1: str, belief_id2: str, strength: float = 1.0) -> bool:
        """Add a contradiction between two beliefs.
        
        Args:
            belief_id1: ID of first belief
            belief_id2: ID of second belief
            strength: Contradiction strength (0.0 to 1.0)
            
        Returns:
            True if added, False if error

        """
        if belief_id1 not in self.beliefs or belief_id2 not in self.beliefs:
            return False

        # Check if contradiction already exists
        for i, (id1, id2, _) in enumerate(self.contradictions):
            if (id1 == belief_id1 and id2 == belief_id2) or (id1 == belief_id2 and id2 == belief_id1):
                # Update existing contradiction
                self.contradictions[i] = (id1, id2, strength)
                return True

        # Add new contradiction
        self.contradictions.append((belief_id1, belief_id2, strength))
        return True

    def update_with_new_evidence(self, memory_id: str, memory_content: Dict[str, Any]) -> List[str]:
        """Update beliefs based on new evidence.
        
        Args:
            memory_id: ID of the new memory
            memory_content: Content of the memory
            
        Returns:
            List of affected belief IDs

        """
        affected_beliefs = []

        # Check for relevant beliefs that this memory might support or contradict
        relevant_beliefs = self._find_relevant_beliefs(memory_content)

        for belief_id in relevant_beliefs:
            belief = self.beliefs[belief_id]

            # Calculate evidence impact
            impact = self._calculate_evidence_impact(belief, memory_content)

            if impact > 0:
                # Memory supports the belief
                if memory_id not in belief.supporting_memories:
                    belief.supporting_memories.append(memory_id)

                # Update confidence based on evidence strength
                old_confidence = belief.confidence
                belief.update_confidence(min(1.0, belief.confidence + impact * 0.1))

                # Add to evidence index
                if memory_id not in self.evidence_index:
                    self.evidence_index[memory_id] = []
                if belief_id not in self.evidence_index[memory_id]:
                    self.evidence_index[memory_id].append(belief_id)

                affected_beliefs.append(belief_id)

            elif impact < 0:
                # Memory contradicts the belief
                # Reduce confidence based on evidence strength
                old_confidence = belief.confidence
                belief.update_confidence(max(0.0, belief.confidence + impact * 0.1))

                affected_beliefs.append(belief_id)

        return affected_beliefs

    def resolve_contradictions(self) -> List[str]:
        """Resolve contradictions between beliefs.
        
        Returns:
            List of affected belief IDs

        """
        affected_beliefs = []

        for belief_id1, belief_id2, strength in self.contradictions:
            if belief_id1 not in self.beliefs or belief_id2 not in self.beliefs:
                continue

            belief1 = self.beliefs[belief_id1]
            belief2 = self.beliefs[belief_id2]

            # Skip if confidences are too low to matter
            if belief1.confidence < 0.2 and belief2.confidence < 0.2:
                continue

            # Calculate confidence difference
            diff = abs(belief1.confidence - belief2.confidence)

            # If confidences are similar, slightly reduce both
            if diff < 0.2:
                old_conf1 = belief1.confidence
                old_conf2 = belief2.confidence

                belief1.update_confidence(max(0.1, belief1.confidence - 0.05 * strength))
                belief2.update_confidence(max(0.1, belief2.confidence - 0.05 * strength))

                affected_beliefs.extend([belief_id1, belief_id2])

            # If one is significantly more confident, reduce the less confident one
            else:
                if belief1.confidence > belief2.confidence:
                    weaker_belief = belief2
                    weaker_id = belief_id2
                else:
                    weaker_belief = belief1
                    weaker_id = belief_id1

                old_conf = weaker_belief.confidence
                weaker_belief.update_confidence(max(0.1, weaker_belief.confidence - 0.1 * strength))

                affected_beliefs.append(weaker_id)

        return affected_beliefs

    def _find_relevant_beliefs(self, memory_content: Dict[str, Any]) -> List[str]:
        """Find beliefs that might be relevant to a memory.
        
        Args:
            memory_content: Content of the memory
            
        Returns:
            List of relevant belief IDs

        """
        relevant_beliefs = []

        # Extract key phrases from memory
        key_phrases = []
        if "type" in memory_content:
            key_phrases.append(memory_content["type"])

        # Flatten the content to extract text from nested dictionaries
        flat_content = self._flatten_dict(memory_content)

        for key, value in flat_content.items():
            if isinstance(value, str):
                key_phrases.append(value)
            elif isinstance(value, dict) and "text" in value:
                key_phrases.append(value["text"])

        # Check each belief for relevance
        for belief_id, belief in self.beliefs.items():
            # Check if any key phrase appears in the belief parts
            if any(phrase in belief.subject for phrase in key_phrases if isinstance(phrase, str)):
                relevant_beliefs.append(belief_id)
                continue

            if any(phrase in belief.predicate for phrase in key_phrases if isinstance(phrase, str)):
                relevant_beliefs.append(belief_id)
                continue

            if any(phrase in belief.object for phrase in key_phrases if isinstance(phrase, str)):
                relevant_beliefs.append(belief_id)
                continue

        return relevant_beliefs

    def _calculate_evidence_impact(self, belief: Belief, memory_content: Dict[str, Any]) -> float:
        """Calculate the impact of evidence on a belief.
        
        Args:
            belief: Belief to evaluate
            memory_content: Content of the memory
            
        Returns:
            Impact value (-1.0 to 1.0, negative=contradicts, positive=supports)

        """
        # Default small positive impact for relevant memories
        impact = 0.1

        # Extract emotional valence if present
        if "emotional_valence" in memory_content:
            emotional_valence = memory_content["emotional_valence"]

            # Emotions tend to strengthen congruent beliefs
            if "emotional_context" in memory_content:
                emotional_context = memory_content.get("emotional_context", {})

                # Check if emotions align with belief
                positive_emotions = [
                    EmotionType.JOY, EmotionType.TRUST,
                    EmotionType.ANTICIPATION, EmotionType.INTEREST,
                ]
                negative_emotions = [
                    EmotionType.SADNESS, EmotionType.FEAR,
                    EmotionType.ANGER, EmotionType.DISGUST,
                ]

                # Convert string emotions to enum
                present_emotions = set()
                for emotion_str, intensity in emotional_context.items():
                    try:
                        emotion = EmotionType(emotion_str)
                        present_emotions.add(emotion)
                    except ValueError:
                        pass

                # Beliefs with "good", "like" tend to align with positive emotions
                has_positive_terms = any(term in belief.object for term in ["good", "like", "happy", "nice"])
                has_negative_terms = any(term in belief.object for term in ["bad", "dislike", "sad", "angry"])

                # Modify impact based on emotional congruence
                if (has_positive_terms and any(emotion in positive_emotions for emotion in present_emotions)) or (has_negative_terms and any(emotion in negative_emotions for emotion in present_emotions)):
                    impact += 0.2
                elif (has_positive_terms and any(emotion in negative_emotions for emotion in present_emotions)) or (has_negative_terms and any(emotion in positive_emotions for emotion in present_emotions)):
                    impact -= 0.2

        # More complex evidence impact calculation could be implemented here

        return impact

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """Flatten nested dictionaries for easier searching.
        
        Args:
            d: Dictionary to flatten
            parent_key: Key from parent dictionary
            sep: Separator for keys
            
        Returns:
            Flattened dictionary

        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def get_related_beliefs(self, belief_id: str, min_strength: float = 0.3) -> List[Tuple[str, float]]:
        """Get beliefs related to a given belief.
        
        Args:
            belief_id: ID of the belief to get relations for
            min_strength: Minimum relationship strength to include
            
        Returns:
            List of tuples (belief_id, relationship_strength)

        """
        if belief_id not in self.belief_relationships:
            return []

        return [
            (related_id, strength)
            for related_id, strength in self.belief_relationships[belief_id].items()
            if strength >= min_strength
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "beliefs": {k: v.to_dict() for k, v in self.beliefs.items()},
            "belief_relationships": self.belief_relationships,
            "evidence_index": self.evidence_index,
            "contradictions": self.contradictions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeliefNetwork":
        """Create a belief network from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            BeliefNetwork instance

        """
        network = cls()

        # Restore beliefs
        for belief_id, belief_data in data.get("beliefs", {}).items():
            # Convert stage name to enum
            stage_name = belief_data.pop("developmental_stage", "INFANT")
            try:
                stage = DevelopmentalStage[stage_name]
            except KeyError:
                stage = DevelopmentalStage.INFANT

            # Convert timestamp strings to datetime
            creation_time = datetime.fromisoformat(belief_data.pop("creation_time", datetime.now().isoformat()))
            last_update_time = datetime.fromisoformat(belief_data.pop("last_update_time", datetime.now().isoformat()))

            stored_id = belief_data.pop("id", None)
            if stored_id is not None and stored_id != belief_id:
                logger.warning(f"Belief keyed {belief_id} stores id {stored_id}; keeping {belief_id}")

            # Create belief
            belief = Belief(
                id=belief_id,
                creation_time=creation_time,
                last_update_time=last_update_time,
                developmental_stage=stage,
                **belief_data,
            )
            network.beliefs[belief_id] = belief

        # Restore relationships
        network.belief_relationships = data.get("belief_relationships", {})

        # Restore evidence index
        network.evidence_index = data.get("evidence_index", {})

        # Restore contradictions
        network.contradictions = data.get("contradictions", [])

        return network

class NeedMotivationSystem(BaseModel):
    """System for managing needs and motivations that drive behavior."""

    needs: Dict[str, Need] = Field(default_factory=dict, description="Current needs")
    need_history: Dict[str, List[Tuple[datetime, float]]] = Field(
        default_factory=dict,
        description="History of need intensity over time",
    )
    need_satisfaction_history: Dict[str, List[Tuple[datetime, float]]] = Field(
        default_factory=dict,
        description="History of need satisfaction over time",
    )
    need_priorities: Dict[str, float] = Field(
        default_factory=dict,
        description="Priority weights for different needs",
    )
    developmental_need_profiles: Dict[DevelopmentalStage, Dict[str, float]] = Field(
        default_factory=dict,
        description="Need profiles for different developmental stages",
    )
    last_update: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize the need motivation system."""
        super().__init__(**data)

        # Initialize default needs if empty
        if not self.needs:
            self._initialize_default_needs()

        # Initialize developmental profiles if empty
        if not self.developmental_need_profiles:
            self._initialize_developmental_profiles()

    def _initialize_default_needs(self) -> None:
        """Initialize the default set of needs."""
        default_needs = {
            "comfort": Need(name="comfort", intensity=0.7, satisfaction_level=0.3),
            "stimulation": Need(name="stimulation", intensity=0.8, satisfaction_level=0.2),
            "rest": Need(name="rest", intensity=0.3, satisfaction_level=0.7),
            "bonding": Need(name="bonding", intensity=0.9, satisfaction_level=0.1),
            "autonomy": Need(name="autonomy", intensity=0.1, satisfaction_level=0.1),
            "understanding": Need(name="understanding", intensity=0.6, satisfaction_level=0.2),
            "competence": Need(name="competence", intensity=0.4, satisfaction_level=0.3),
        }

        self.needs.update(default_needs)

        # Initialize empty history for each need
        for need_name in default_needs:
            if need_name not in self.need_history:
                self.need_history[need_name] = []
            if need_name not in self.need_satisfaction_history:
                self.need_satisfaction_history[need_name] = []

        # Set initial priorities
        self.need_priorities = {
            "comfort": 1.0,
            "stimulation": 0.8,
            "rest": 0.7,
            "bonding": 0.9,
            "autonomy": 0.4,
            "understanding": 0.6,
            "competence": 0.5,
        }

    def _initialize_developmental_profiles(self) -> None:
        """Initialize need profiles for different developmental stages."""
        self.developmental_need_profiles = {
            DevelopmentalStage.INFANT: {
                "comfort": 1.0,
                "stimulation": 0.8,
                "rest": 0.9,
                "bonding": 1.0,
                "autonomy": 0.1,
                "understanding": 0.7,
                "competence": 0.3,
            },
            DevelopmentalStage.TODDLER: {
                "comfort": 0.8,
                "stimulation": 1.0,
                "rest": 0.8,
                "bonding": 0.9,
                "autonomy": 0.5,
                "understanding": 0.9,
                "competence": 0.6,
            },
            DevelopmentalStage.CHILD: {
                "comfort": 0.6,
                "stimulation": 0.9,
                "rest": 0.7,
                "bonding": 0.8,
                "autonomy": 0.7,
                "understanding": 1.0,
                "competence": 0.8,
            },
            DevelopmentalStage.ADOLESCENT: {
                "comfort": 0.5,
                "stimulation": 0.8,
                "rest": 0.6,
                "bonding": 0.7,
                "autonomy": 0.9,
                "understanding": 0.9,
                "competence": 0.9,
            },
            DevelopmentalStage.MATURE: {
                "comfort": 0.6,
                "stimulation": 0.7,
                "rest": 0.6,
                "bonding": 0.8,
                "autonomy": 1.0,
                "understanding": 0.8,
                "competence": 1.0,
            },
        }

    def update_needs(self, elapsed_seconds: float, stage: DevelopmentalStage) -> None:
        """Update need intensities based on time elapsed and development.
        
        Args:
            elapsed_seconds: Time elapsed since last update (seconds)
            stage: Current developmental stage

        """
        # Get current time
        current_time = datetime.now()

        # Get stage profile
        stage_profile = self.developmental_need_profiles.get(stage, {})

        # Update all needs
        for need_name, need in self.needs.items():
            # Base rate of need increase (per second)
            base_rate = 0.0003  # Very slow base rate

            # Scale by stage-specific priority and elapsed time
            stage_priority = stage_profile.get(need_name, 0.5)
            rate_factor = stage_priority * base_rate * elapsed_seconds

            # Apply different dynamics based on need type
            if need_name == "stimulation":
                # Stimulation need increases faster if satisfaction is high
                # (representing habituation/boredom)
                habituation_factor = 1.0 + need.satisfaction_level
                need.update_intensity(rate_factor * habituation_factor)

            elif need_name == "rest":
                # Rest need increases proportional to elapsed time and inversely to satisfaction
                fatigue_factor = 1.0 + (1.0 - need.satisfaction_level)
                need.update_intensity(rate_factor * fatigue_factor)

            elif need_name == "autonomy":
                # Autonomy need grows more with developmental stage
                stage_factor = 0.2 * stage.value
                need.update_intensity(rate_factor * stage_factor)

            else:
                # Default linear increase
                need.update_intensity(rate_factor)

            # Natural decrease in satisfaction over time
            satisfaction_decay = 0.0002 * elapsed_seconds
            need.satisfaction_level = max(0.0, need.satisfaction_level - satisfaction_decay)

            # Record history
            self.need_history[need_name].append((current_time, need.intensity))
            self.need_satisfaction_history[need_name].append((current_time, need.satisfaction_level))

            # Limit history length
            max_history = 100
            if len(self.need_history[need_name]) > max_history:
                self.need_history[need_name] = self.need_history[need_name][-max_history:]
            if len(self.need_satisfaction_history[need_name]) > max_history:
                self.need_satisfaction_history[need_name] = self.need_satisfaction_history[need_name][-max_history:]

        self.last_update = current_time

    def satisfy_need(self, need_name: str, amount: float) -> bool:
        """Satisfy a specific need.
        
        Args:
            need_name: Name of the need to satisfy
            amount: Amount to satisfy by (0.0 to 1.0)
            
        Returns:
            True if successful, False if need not found

        """
        if need_name not in self.needs:
            return False

        need = self.needs[need_name]
        need.satisfy(amount)

        # When a need is satisfied, its intensity is reduced
        intensity_reduction = amount * 0.5
        need.update_intensity(-intensity_reduction)

        # Record in history
        current_time = datetime.now()
        self.need_history[need_name].append((current_time, need.intensity))
        self.need_satisfaction_history[need_name].append((current_time, need.satisfaction_level))

        return True

    def get_dominant_need(self) -> Optional[Tuple[str, float]]:
        """Get the currently dominant need.
        
        Returns:
            Tuple of (need_name, weighted_intensity) or None if no needs

        """
        if not self.needs:
            return None

        # Calculate weighted intensities
        weighted_intensities = []
        for need_name, need in self.needs.items():
            priority = self.need_priorities.get(need_name, 0.5)
            satisfaction_factor = 1.0 - (need.satisfaction_level * 0.7)  # Less satisfied needs get higher weight
            weighted_intensity = need.intensity * priority * satisfaction_factor
            weighted_intensities.append((need_name, weighted_intensity))

        # Sort by weighted intensity (descending)
        weighted_intensities.sort(key=lambda x: x[1], reverse=True)

        return weighted_intensities[0]

    def get_expressed_needs(self, threshold: float = 0.5) -> Dict[str, float]:
        """Get needs with intensity above a threshold.
        
        Args:
            threshold: Minimum intensity to include
            
        Returns:
            Dictionary of need_name -> intensity

        """
        return {
            name: need.intensity
            for name, need in self.needs.items()
            if need.intensity >= threshold
        }

    def update_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update need system for a new developmental stage.
        
        Args:
            stage: New developmental stage

        """
        # Update priorities based on stage profile
        if stage in self.developmental_need_profiles:
            self.need_priorities = self.developmental_need_profiles[stage].copy()

        # Add new needs that become relevant at this stage
        if stage == DevelopmentalStage.TODDLER:
            # Toddlers develop need for exploration
            if "exploration" not in self.needs:
                self.needs["exploration"] = Need(name="exploration", intensity=0.7, satisfaction_level=0.2)
                self.need_history["exploration"] = []
                self.need_satisfaction_history["exploration"] = []
                self.need_priorities["exploration"] = 0.8

        elif stage == DevelopmentalStage.CHILD:
            # Children develop need for achievement
            if "achievement" not in self.needs:
                self.needs["achievement"] = Need(name="achievement", intensity=0.6, satisfaction_level=0.3)
                self.need_history["achievement"] = []
                self.need_satisfaction_history["achievement"] = []
                self.need_priorities["achievement"] = 0.7

        elif stage == DevelopmentalStage.ADOLESCENT:
            # Adolescents develop need for identity and belonging
            if "identity" not in self.needs:
                self.needs["identity"] = Need(name="identity", intensity=0.8, satisfaction_level=0.2)
                self.need_history["identity"] = []
                self.need_satisfaction_history["identity"] = []
                self.need_priorities["identity"] = 0.9

            if "belonging" not in self.needs:
                self.needs["belonging"] = Need(name="belonging", intensity=0.7, satisfaction_level=0.3)
                self.need_history["belonging"] = []
                self.need_satisfaction_history["belonging"] = []
                self.need_priorities["belonging"] = 0.8

        elif stage == DevelopmentalStage.MATURE:
            # Mature individuals develop need for self-actualization
            if "self_actualization" not in self.needs:
                self.needs["self_actualization"] = Need(name="self_actualization", intensity=0.6, satisfaction_level=0.2)
                self.need_history["self_actualization"] = []
                self.need_satisfaction_history["self_actualization"] = []
                self.need_priorities["self_actualization"] = 0.9

    def get_need_trend(self, need_name: str, window: int = 10) -> Optional[float]:
        """Calculate trend in a need's intensity over recent history.
        
        Args:
            need_name: Name of need to calculate trend for
            window: Number of recent entries to use
            
        Returns:
            Trend value (positive=increasing, negative=decreasing) or None if insufficient data

        """
        if need_name not in self.need_history or len(self.need_history[need_name]) < window:
            return None

        # Get recent history entries
        recent = self.need_history[need_name][-window:]

        # Extract intensities (ignoring timestamps)
        intensities = [intensity for _, intensity in recent]

        if len(intensities) < 2:
            return 0.0

        # Calculate simple trend (average of sequential differences)
        diffs = [intensities[i] - intensities[i-1] for i in range(1, len(intensities))]
        return sum(diffs) / len(diffs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "needs": {name: {
                "intensity": need.intensity,
                "satisfaction_level": need.satisfaction_level,
                "last_update": need.last_update.isoformat(),
            } for name, need in self.needs.items()},
            "need_priorities": self.need_priorities,
            "last_update": self.last_update.isoformat(),
        }

class Mind:
    """Core class for the mind simulation.
    
    The Mind coordinates all neural networks, manages development,
    and maintains the overall state of the artificial mind.
    """

    def __init__(self):
        """Initialize the mind simulation."""
        self.networks: Dict[str, NeuralNetwork] = {}
        self.state = MindState(
            consciousness_level=0.2,  # Start with lower consciousness (infant-like)
            emotional_state={
                EmotionType.JOY: 0.3,
                EmotionType.TRUST: 0.3,
                EmotionType.FEAR: 0.3,
                EmotionType.SURPRISE: 0.3,
            },
            energy_level=0.7,
            developmental_stage=DevelopmentalStage.INFANT,
            language_ability=LanguageAbility(
                vocabulary_size=0,
                sentence_complexity=0.0,
                understanding_level=0.1,
                expression_level=0.0,
            ),
        )

        # Subscribe to message bus
        self.message_bus = GlobalMessageBus.get_instance()
        self.message_queue = self.message_bus.subscribe(
            "mind",
            MessageFilter(receiver="mind", min_priority=0.3),
        )

        # Initialize memory systems
        self.short_term_memory: List[Memory] = []
        self.long_term_memory: List[Memory] = []
        self.memory_clusters: Dict[str, MemoryCluster] = {}

        # Initialize belief network
        self.belief_network = BeliefNetwork()

        # Initialize need motivation system
        self.need_system = NeedMotivationSystem()

        # Development tracking
        self.developmental_milestones = {
            "emotions_experienced": set(),
            "vocabulary_learned": set(),
            "beliefs_formed": 0,
            "interactions_count": 0,
            "memories_formed": 0,
            "memories_consolidated": 0,
            "network_growth_events": 0,
        }

        # Development thresholds for advancing to next stage
        self.development_thresholds = {
            DevelopmentalStage.INFANT: {
                "emotions_experienced": 3,  # Number of distinct emotions experienced
                "interactions_count": 20,
                "memories_formed": 10,
            },
            DevelopmentalStage.TODDLER: {
                "emotions_experienced": 5,
                "vocabulary_learned": 20,
                "interactions_count": 50,
                "memories_formed": 30,
                "memories_consolidated": 10,
            },
            DevelopmentalStage.CHILD: {
                "emotions_experienced": 7,
                "vocabulary_learned": 100,
                "beliefs_formed": 10,
                "interactions_count": 100,
                "memories_formed": 100,
                "memories_consolidated": 50,
            },
            DevelopmentalStage.ADOLESCENT: {
                "emotions_experienced": 8,
                "vocabulary_learned": 500,
                "beliefs_formed": 50,
                "interactions_count": 200,
                "memories_formed": 200,
                "memories_consolidated": 100,
                "network_growth_events": 20,
            },
        }

        # Timing trackers
        self.last_developmental_check = datetime.now()
        self.last_need_update = datetime.now()
        self.last_memory_consolidation = datetime.now()
        self.last_belief_update = datetime.now()
        self.last_network_growth_check = datetime.now()
        self.simulation_time = 0.0  # Time in seconds since simulation start

        # Self-reflection and monitoring
        self.self_awareness_level = 0.1  # Starts low, increases with development
        self.performance_metrics = {
            "avg_response_time": 0.0,
            "memory_utilization": 0.0,
            "belief_consistency": 0.0,
            "learning_efficiency": 0.0,
            "need_balance": 0.0,
        }

        # Adaptive growth parameters
        self.growth_schedule = {
            DevelopmentalStage.INFANT: 0.0,  # No growth in infant stage
            DevelopmentalStage.TODDLER: 0.001,  # Slow growth
            DevelopmentalStage.CHILD: 0.002,
            DevelopmentalStage.ADOLESCENT: 0.003,
            DevelopmentalStage.MATURE: 0.001,  # Slows down again in maturity
        }

        logger.info("Mind initialized at infant developmental stage")

    def register_network(self, network: NeuralNetwork) -> None:
        """Register a neural network with the mind.
        
        Args:
            network: Neural network to register

        """
        self.networks[network.name] = network
        network.update_developmental_stage(self.state.developmental_stage)
        logger.info(f"Registered network: {network.name}")

    def process_messages(self) -> None:
        """Process incoming messages from the message queue."""
        messages = self.message_bus.get_messages(self.message_queue, block=False)

        for message in messages:
            self._process_mind_message(message)

    def process_input(self, input_data: Dict[str, Any]) -> None:
        """Process input data from the environment.
        
        Args:
            input_data: Dictionary of input data containing sensory information

        """
        # Process perception inputs (visual and auditory)
        if "perception" in self.networks:
            # Check if we have any perception-related data
            has_perception_data = "visual" in input_data or "auditory" in input_data

            if has_perception_data:
                # Prepare properly sized tensors for both modalities
                visual_data = input_data.get("visual", [0.0] * 64)  # Default to zeros if missing
                auditory_data = input_data.get("auditory", [0.0] * 64)  # Default to zeros if missing

                # Ensure proper length for visual data
                if isinstance(visual_data, list):
                    if len(visual_data) > 64:
                        visual_data = visual_data[:64]
                    elif len(visual_data) < 64:
                        visual_data = visual_data + [0.0] * (64 - len(visual_data))

                # Ensure proper length for auditory data
                if isinstance(auditory_data, list):
                    if len(auditory_data) > 64:
                        auditory_data = auditory_data[:64]
                    elif len(auditory_data) < 64:
                        auditory_data = auditory_data + [0.0] * (64 - len(auditory_data))

                # Combine into a single tensor for perception network
                combined_data = torch.tensor(visual_data + auditory_data, dtype=torch.float32)
                self.networks["perception"].experiential_learning(combined_data)

        # Process language input
        if "language" in input_data and "language" in self.networks:
            # Process language input - helps with language acquisition
            language_data = input_data["language"]

            # Convert text to appropriate tensor representation
            if isinstance(language_data, str):
                # Very simple tokenization for demonstration
                tokens = language_data.lower().split()
                # Add words to vocabulary
                self.developmental_milestones["vocabulary_learned"].update(tokens)

                # As development progresses, language processing becomes more sophisticated
                if self.state.developmental_stage.value >= DevelopmentalStage.TODDLER.value:
                    # Simple numeric representation of tokens
                    tensor_data = torch.zeros(len(tokens), 10)  # Simple embedding
                    for i, token in enumerate(tokens):
                        # Hash the token to get a consistent embedding
                        hash_val = hash(token) % 1000
                        tensor_data[i] = torch.tensor([int(d) for d in f"{hash_val:010}"])

                    if "language" in self.networks:
                        self.networks["language"].experiential_learning(tensor_data)

        # Form a memory of this input regardless of type
        self._form_memory({
            "type": input_data.get("type", "sensory_input"),
            "data": input_data,
            "time": datetime.now().isoformat(),
        })

        # Increment interaction count
        self.developmental_milestones["interactions_count"] += 1

    def step(self) -> None:
        """Advance the mind simulation by one step."""
        start_time = datetime.now()

        # Process messages from the message bus
        self.process_messages()

        # Update active networks through their autonomous steps
        for network in self.networks.values():
            # Generate text output for observability
            network.generate_text_output()

            # Each network gets a chance to do autonomous processing
            if hasattr(network, "autonomous_step"):
                network.autonomous_step()

            # Retrieve and process any pending messages from the network
            self._retrieve_network_messages(network)

        # Update needs
        self._update_needs()

        # Consolidate memories periodically
        self._consolidate_memories()

        # Update belief system periodically
        self._update_belief_system()

        # Check for neural network growth
        self._check_network_growth()

        # Update overall mind state based on network states
        self._update_mind_state()

        # Check for developmental progress
        self._check_developmental_progress()

        # Self-reflection and monitoring
        self._self_reflection()

        # Increment simulation time
        step_duration = (datetime.now() - start_time).total_seconds()
        self.simulation_time += step_duration

    def _retrieve_network_messages(self, network: NeuralNetwork) -> None:
        """Retrieve and process pending messages from a network.
        
        Args:
            network: Neural network to retrieve messages from

        """
        # Check if the network has pending messages in its state
        if "pending_messages" in network.state.parameters:
            pending_messages = network.state.parameters.get("pending_messages", [])

            for message_dict in pending_messages:
                # Convert dictionary to NetworkMessage
                try:
                    # Extract developmental stage
                    stage_name = message_dict.pop("developmental_stage", self.state.developmental_stage.name)
                    try:
                        stage = DevelopmentalStage[stage_name]
                    except KeyError:
                        stage = self.state.developmental_stage

                    # Create message
                    message = NetworkMessage(
                        developmental_stage=stage,
                        **message_dict,
                    )

                    # Process the message
                    if message.receiver == "mind":
                        self._process_mind_message(message)
                    else:
                        # Forward to the message bus
                        self.message_bus.publish(message)
                except Exception as e:
                    logger.error(f"Error processing message from {network.name}: {e!s}")

            # Clear pending messages
            network.update_state({"pending_messages": []})

    def _process_mind_message(self, message: NetworkMessage) -> None:
        """Process a message directed to the mind itself.
        
        Args:
            message: Message to process

        """
        if message.message_type == "emotion":
            # Update emotional state
            if "emotions" in message.content:
                emotions = message.content["emotions"]
                for emotion_str, intensity in emotions.items():
                    try:
                        emotion_type = EmotionType(emotion_str)
                        self.state.emotional_state[emotion_type] = intensity
                        # Add to experienced emotions for developmental tracking
                        self.developmental_milestones["emotions_experienced"].add(emotion_type)
                    except ValueError:
                        logger.warning(f"Unknown emotion type: {emotion_str}")

                # Form memory of significant emotions
                if any(intensity > 0.7 for intensity in emotions.values()):
                    self._form_memory({
                        "type": "emotional_event",
                        "emotions": emotions,
                        "stimulus": message.content.get("stimulus", "unknown"),
                        "intensity": message.content.get("intensity", 0.5),
                        "valence": message.content.get("valence", 0.0),
                    })

        elif message.message_type == "belief":
            # Add or update a belief
            if all(k in message.content for k in ["subject", "predicate", "object"]):
                subject = message.content["subject"]
                predicate = message.content["predicate"]
                obj = message.content["object"]
                confidence = float(message.content.get("confidence", 0.5))

                # Check if this is a new belief or update to existing
                existing_belief = None
                for belief_id, belief in self.belief_network.beliefs.items():
                    if belief.subject == subject and belief.predicate == predicate and belief.object == obj:
                        existing_belief = belief
                        break

                if existing_belief:
                    # Update existing belief
                    existing_belief.update_confidence(confidence)
                else:
                    # Create new belief
                    belief = Belief(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=confidence,
                        developmental_stage=self.state.developmental_stage,
                    )
                    self.belief_network.add_belief(belief)
                    self.developmental_milestones["beliefs_formed"] += 1

                    # Form memory of significant new belief
                    if confidence > 0.7:
                        self._form_memory({
                            "type": "belief_formation",
                            "belief": {
                                "subject": subject,
                                "predicate": predicate,
                                "object": obj,
                                "confidence": confidence,
                            },
                        })

        elif message.message_type == "consciousness":
            # Update consciousness level
            if "level" in message.content:
                self.state.consciousness_level = min(1.0, max(0.0, float(message.content["level"])))

            # Update current focus/attending
            if message.content.get("attending_to"):
                self.state.current_focus = message.content["attending_to"]

            # Update self-awareness
            if "self_awareness" in message.content:
                self.self_awareness_level = float(message.content["self_awareness"])

        elif message.message_type == "language_output":
            # Process language output
            if "text" in message.content:
                text = message.content["text"]

                # Check if this is a need expression
                need_expressions = {
                    "comfort": ["comfort", "hug", "hold", "safe"],
                    "stimulation": ["play", "toy", "see", "look", "explore"],
                    "rest": ["tired", "sleep", "rest", "quiet"],
                    "bonding": ["mama", "dada", "love", "smile", "together"],
                    "autonomy": ["me", "mine", "self", "do", "my"],
                }

                # Check each need for mentions
                for need_name, keywords in need_expressions.items():
                    if need_name in self.need_system.needs and any(word in text.lower() for word in keywords):
                        # This language output expresses a need
                        need = self.need_system.needs[need_name]

                        # Form memory of need expression
                        self._form_memory({
                            "type": "need_expression",
                            "need": need_name,
                            "intensity": need.intensity,
                            "expression": text,
                        })

        elif message.message_type == "need":
            # Update a need
            if "name" in message.content and "change" in message.content:
                need_name = message.content["name"]
                change = float(message.content["change"])

                if message.content.get("satisfy"):
                    self.need_system.satisfy_need(need_name, change)
                elif need_name in self.need_system.needs:
                    self.need_system.needs[need_name].update_intensity(change)

    def _update_needs(self) -> None:
        """Update the intensity of needs based on time and state."""
        current_time = datetime.now()
        elapsed = (current_time - self.last_need_update).total_seconds()

        if elapsed < config.mind.need_update_interval:
            return

        # Update needs based on elapsed time and developmental stage
        self.need_system.update_needs(elapsed, self.state.developmental_stage)

        self.last_need_update = current_time

    def _form_memory(self, content: Dict[str, Any]) -> None:
        """Form a new short-term memory.
        
        Args:
            content: Memory content

        """
        # Add emotional context
        emotional_context = {
            emotion.name: intensity
            for emotion, intensity in self.state.emotional_state.items()
            if intensity > 0.2  # Only include significant emotions
        }

        # Calculate emotional valence (-1 to 1)
        valence = sum([
            self.state.emotional_state.get(EmotionType.JOY, 0) * 1.0,
            self.state.emotional_state.get(EmotionType.TRUST, 0) * 0.8,
            self.state.emotional_state.get(EmotionType.SADNESS, 0) * -0.8,
            self.state.emotional_state.get(EmotionType.FEAR, 0) * -0.6,
            self.state.emotional_state.get(EmotionType.ANGER, 0) * -1.0,
        ])
        valence = max(-1.0, min(1.0, valence))

        # Create memory with unique ID
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"

        memory = Memory(
            id=memory_id,
            content={
                **content,
                "emotional_context": emotional_context,
                "consciousness_level": self.state.consciousness_level,
                "developmental_stage": self.state.developmental_stage.name,
            },
            emotional_valence=valence,
            developmental_stage=self.state.developmental_stage,
            tags=["recent"],
        )

        # Add to short-term memory
        self.short_term_memory.append(memory)

        # Limit short-term memory size based on developmental stage
        max_stm_size = 3 + (self.state.developmental_stage.value * 2)
        if len(self.short_term_memory) > max_stm_size:
            self.short_term_memory = self.short_term_memory[-max_stm_size:]

        self.developmental_milestones["memories_formed"] += 1

    def _consolidate_memories(self) -> None:
        """Consolidate short-term memories into long-term memory."""
        current_time = datetime.now()

        # Only consolidate periodically
        if (current_time - self.last_memory_consolidation).total_seconds() < config.mind.memory_consolidation_interval:
            return

        # Consolidate memories with enough strength or emotional significance
        memories_to_consolidate = []
        for memory in self.short_term_memory:
            # Memories with strong emotional valence or accessed multiple times are consolidated
            if (abs(memory.emotional_valence) > 0.6 or
                memory.strength > 1.5 or
                (current_time - memory.creation_time).total_seconds() > 300):  # 5 minutes old
                memories_to_consolidate.append(memory)

                # Tag important memories
                if abs(memory.emotional_valence) > 0.7:
                    memory.tags.append("emotionally_significant")
                if memory.strength > 2.0:
                    memory.tags.append("well_reinforced")

        # Move memories to long-term storage
        for memory in memories_to_consolidate:
            if memory in self.short_term_memory:
                self.short_term_memory.remove(memory)
                self.long_term_memory.append(memory)
                self.developmental_milestones["memories_consolidated"] += 1

                # Try to cluster similar memories
                self._cluster_memory(memory)

                # Update belief system with this memory
                self.belief_network.update_with_new_evidence(memory.id, memory.content)

        # Forget weak long-term memories (more aggressive at infant/toddler stages)
        forget_threshold = 0.1
        if self.state.developmental_stage == DevelopmentalStage.INFANT:
            forget_threshold = 0.3
        elif self.state.developmental_stage == DevelopmentalStage.TODDLER:
            forget_threshold = 0.2

        # Apply decay to all long-term memories
        memories_to_forget = []
        for memory in self.long_term_memory:
            # More important memories decay slower
            decay_rate = 0.01
            if "emotionally_significant" in memory.tags:
                decay_rate *= 0.5
            if "well_reinforced" in memory.tags:
                decay_rate *= 0.7

            memory.decay(decay_rate)

            if memory.strength < forget_threshold:
                memories_to_forget.append(memory)

        # Remove forgotten memories
        for memory in memories_to_forget:
            if memory in self.long_term_memory:
                self.long_term_memory.remove(memory)

                # Remove from clusters
                for cluster in self.memory_clusters.values():
                    cluster.remove_memory(memory.id)

                # Remove from belief evidence index
                for belief_ids in self.belief_network.evidence_index.values():
                    if memory.id in belief_ids:
                        belief_ids.remove(memory.id)

        self.last_memory_consolidation = current_time

    def _cluster_memory(self, memory: Memory) -> None:
        """Cluster a memory with similar memories.
        
        Args:
            memory: Memory to cluster

        """
        # Skip clustering for earlier developmental stages
        if self.state.developmental_stage.value < DevelopmentalStage.TODDLER.value:
            return

        # Extract memory type
        memory_type = memory.content.get("type", "unknown")

        # Find the best matching cluster
        best_cluster = None
        best_match_score = 0.3  # Minimum threshold for matching

        for cluster in self.memory_clusters.values():
            # Calculate match score
            match_score = self._calculate_memory_match(memory, cluster)

            if match_score > best_match_score:
                best_match_score = match_score
                best_cluster = cluster

        # If good match found, add to existing cluster
        if best_cluster is not None:
            best_cluster.add_memory(memory.id)
            best_cluster.access()  # Update access time

            # If emotionally significant, increase cluster importance
            if abs(memory.emotional_valence) > 0.7:
                best_cluster.importance = min(1.0, best_cluster.importance + 0.1)

        # Otherwise create a new cluster
        else:
            # Generate a label for the cluster
            if memory_type == "emotional_event":
                label = f"Feelings about {memory.content.get('stimulus', 'something')}"
            elif memory_type == "need_expression":
                label = f"Expressions of {memory.content.get('need', 'a need')}"
            elif memory_type == "belief_formation":
                belief = memory.content.get("belief", {})
                label = f"Beliefs about {belief.get('subject', 'something')}"
            elif memory_type == "sensory_input":
                label = f"Experiences with {memory.content.get('data', {}).get('type', 'sensory input')}"
            else:
                label = f"Memories of {memory_type}"

            # Create new cluster
            new_cluster = MemoryCluster(
                label=label,
                memory_ids=[memory.id],
                importance=0.5 + abs(memory.emotional_valence) * 0.3,
                developmental_stage=self.state.developmental_stage,
            )

            # Add to clusters
            self.memory_clusters[new_cluster.id] = new_cluster

    def _calculate_memory_match(self, memory: Memory, cluster: MemoryCluster) -> float:
        """Calculate how well a memory matches a cluster.
        
        Args:
            memory: Memory to evaluate
            cluster: Cluster to compare with
            
        Returns:
            Match score (0.0 to 1.0)

        """
        # If cluster is empty, no match
        if not cluster.memory_ids:
            return 0.0

        # Get a sample of memories from the cluster
        sample_size = min(5, len(cluster.memory_ids))
        sample_memory_ids = random.sample(cluster.memory_ids, sample_size)

        # Find these memories in long-term memory
        cluster_memories = [
            mem for mem in self.long_term_memory
            if mem.id in sample_memory_ids
        ]

        if not cluster_memories:
            return 0.0

        # Calculate match scores
        match_scores = []

        for cluster_memory in cluster_memories:
            # Type match
            if memory.content.get("type") == cluster_memory.content.get("type"):
                match_scores.append(0.5)

            # For emotional events, check stimulus match
            if memory.content.get("type") == "emotional_event" and cluster_memory.content.get("type") == "emotional_event":
                memory_stimulus = memory.content.get("stimulus", "").lower()
                cluster_stimulus = cluster_memory.content.get("stimulus", "").lower()

                if memory_stimulus and cluster_stimulus and memory_stimulus == cluster_stimulus:
                    match_scores.append(0.7)

            # For need expressions, check need match
            if memory.content.get("type") == "need_expression" and cluster_memory.content.get("type") == "need_expression":
                if memory.content.get("need") == cluster_memory.content.get("need"):
                    match_scores.append(0.8)

            # For beliefs, check subject/predicate match
            if memory.content.get("type") == "belief_formation" and cluster_memory.content.get("type") == "belief_formation":
                memory_belief = memory.content.get("belief", {})
                cluster_belief = cluster_memory.content.get("belief", {})

                if memory_belief.get("subject") == cluster_belief.get("subject"):
                    match_scores.append(0.6)

                if memory_belief.get("predicate") == cluster_belief.get("predicate"):
                    match_scores.append(0.3)

            # Emotional valence similarity
            valence_diff = abs(memory.emotional_valence - cluster_memory.emotional_valence)
            if valence_diff < 0.3:
                match_scores.append(0.4)

        # Return average match score if we have any
        if match_scores:
            return sum(match_scores) / len(match_scores)

        return 0.0

    def _update_belief_system(self) -> None:
        """Update the belief system periodically."""
        current_time = datetime.now()

        # Only update periodically
        update_interval = 60.0  # 1 minute
        if (current_time - self.last_belief_update).total_seconds() < update_interval:
            return

        # More sophisticated belief updates happen at higher developmental stages
        if self.state.developmental_stage.value >= DevelopmentalStage.CHILD.value:
            # Resolve contradictions
            affected_beliefs = self.belief_network.resolve_contradictions()

            # Form relationships between beliefs
            if self.state.developmental_stage.value >= DevelopmentalStage.ADOLESCENT.value:
                self._form_belief_relationships()

        self.last_belief_update = current_time

    def _form_belief_relationships(self) -> None:
        """Form relationships between beliefs based on shared memories and subjects."""
        # Get all belief pairs
        beliefs = list(self.belief_network.beliefs.items())
        if len(beliefs) < 2:
            return

        # Sample some pairs to avoid excessive processing
        sample_size = min(10, len(beliefs) * (len(beliefs) - 1) // 2)
        sampled_pairs = random.sample([(i, j) for i in range(len(beliefs)) for j in range(i+1, len(beliefs))], sample_size)

        for i, j in sampled_pairs:
            belief1_id, belief1 = beliefs[i]
            belief2_id, belief2 = beliefs[j]

            # Calculate relationship strength
            relationship_strength = 0.0

            # Check for shared subject
            if belief1.subject == belief2.subject:
                relationship_strength += 0.6

            # Check for shared predicate
            if belief1.predicate == belief2.predicate:
                relationship_strength += 0.3

            # Check for shared object
            if belief1.object == belief2.object:
                relationship_strength += 0.4

            # Check for shared supporting memories
            shared_memories = set(belief1.supporting_memories).intersection(set(belief2.supporting_memories))
            if shared_memories:
                relationship_strength += min(0.8, len(shared_memories) * 0.2)

            # Detect potential contradictions
            if (belief1.subject == belief2.subject and
                belief1.predicate == belief2.predicate and
                belief1.object != belief2.object):
                # Potential contradiction - subject and predicate match but objects differ
                self.belief_network.add_contradiction(belief1_id, belief2_id, 0.8)

            # Add relationship if strong enough
            if relationship_strength > 0.3:
                self.belief_network.add_relationship(belief1_id, belief2_id, relationship_strength)

    def _check_network_growth(self) -> None:
        """Check if neural networks should grow based on development."""
        current_time = datetime.now()

        # Only check periodically
        if (current_time - self.last_network_growth_check).total_seconds() < config.mind.network_growth_check_interval:
            return

        # Get growth rate for current stage
        growth_rate = self.growth_schedule.get(self.state.developmental_stage, 0.0)

        if growth_rate <= 0.0:
            self.last_network_growth_check = current_time
            return

        # Check for networks that need to grow
        for name, network in self.networks.items():
            # Check if eligible and time for growth
            if (hasattr(network, "growth_eligible") and
                network.growth_eligible and
                random.random() < growth_rate):

                # Get current dimensions
                current_dims = [network.input_dim, network.output_dim]

                # Clone with growth
                try:
                    # This now works because all subclasses implement clone_with_growth
                    grown_network = network.clone_with_growth(growth_factor=1.2)

                    # Replace the network in our registry
                    self.networks[name] = grown_network
                    self.developmental_milestones["network_growth_events"] += 1

                    logger.info(f"Network {name} grown from {current_dims} → "
                            f"[{grown_network.input_dim}, {grown_network.output_dim}]")

                except Exception as e:
                    logger.error(f"Error growing network {name}: {e!s}")

        self.last_network_growth_check = current_time

    def _update_mind_state(self) -> None:
        """Update the overall mind state based on network and need states."""
        # Update energy level based on rest need
        rest_deficit = self.need_system.needs.get("rest", Need(name="rest")).intensity
        self.state.energy_level = max(0.1, min(1.0, 1.0 - (rest_deficit * 0.5)))

        # Update current focus based on most active network
        if not self.state.current_focus:
            most_active_network = None
            highest_activation = 0.0

            for name, network in self.networks.items():
                if "average_activation" in network.state.parameters:
                    activation = network.state.parameters["average_activation"]
                    if activation > highest_activation:
                        highest_activation = activation
                        most_active_network = name

            if most_active_network:
                self.state.current_focus = most_active_network

        # Update language ability based on developmental stage and language network
        if "language" in self.networks:
            language_network = self.networks["language"]

            # Get vocabulary size from network
            vocab_size = language_network.state.parameters.get("vocabulary_size", 0)
            self.developmental_milestones["vocabulary_learned"] = set(list(self.developmental_milestones["vocabulary_learned"])[:vocab_size])

            # Get language abilities from network state
            sentence_complexity = language_network.state.parameters.get("sentence_complexity", 0.0)
            understanding_level = language_network.state.parameters.get("understanding_level", 0.1)
            expression_level = language_network.state.parameters.get("expression_level", 0.0)

            self.state.language_ability = LanguageAbility(
                vocabulary_size=vocab_size,
                sentence_complexity=sentence_complexity,
                understanding_level=understanding_level,
                expression_level=expression_level,
            )
        else:
            # Fall back to developmental stage-based estimates if no language network
            vocab_size = len(self.developmental_milestones["vocabulary_learned"])

            # Language ability scales with developmental stage and vocabulary
            sentence_complexity = min(1.0, (self.state.developmental_stage.value - 1) * 0.25 + (vocab_size / 1000))
            understanding = min(1.0, (self.state.developmental_stage.value - 1) * 0.2 + (vocab_size / 800))
            expression = min(1.0, (self.state.developmental_stage.value - 1) * 0.18 + (vocab_size / 1200))

            self.state.language_ability = LanguageAbility(
                vocabulary_size=vocab_size,
                sentence_complexity=sentence_complexity,
                understanding_level=understanding,
                expression_level=expression,
            )

        # Update timestamp
        self.state.timestamp = datetime.now()

    def _check_developmental_progress(self) -> None:
        """Check if the mind has progressed to the next developmental stage."""
        current_time = datetime.now()

        # Only check periodically
        if (current_time - self.last_developmental_check).total_seconds() < config.mind.development_check_interval:
            return

        # Can't progress beyond mature
        if self.state.developmental_stage == DevelopmentalStage.MATURE:
            return

        # Get thresholds for current stage
        next_stage_value = self.state.developmental_stage.value + 1
        next_stage = DevelopmentalStage(next_stage_value)

        current_thresholds = self.development_thresholds[self.state.developmental_stage]

        # Check if all thresholds are met
        all_met = True
        milestone_progress = {}

        for metric, threshold in current_thresholds.items():
            current_value = 0

            if metric == "emotions_experienced":
                current_value = len(self.developmental_milestones["emotions_experienced"])
            elif metric == "vocabulary_learned":
                current_value = len(self.developmental_milestones["vocabulary_learned"])
            elif metric in self.developmental_milestones:
                current_value = self.developmental_milestones[metric]

            milestone_progress[metric] = {
                "current": current_value,
                "threshold": threshold,
                "percentage": min(100, int((current_value / threshold) * 100)),
            }

            if current_value < threshold:
                all_met = False

        # Apply development acceleration if configured
        development_acceleration = config.mind.development_acceleration
        if development_acceleration > 1.0 and not all_met:
            # Calculate overall progress percentage
            total_percentage = sum(data["percentage"] for data in milestone_progress.values()) / len(milestone_progress)

            # Accelerated development has a chance to skip remaining requirements
            advancement_probability = (total_percentage / 100) * (development_acceleration - 1.0) * 0.1

            if random.random() < advancement_probability:
                logger.info(f"Development acceleration triggered advancement with {total_percentage:.1f}% milestone completion")
                all_met = True

        if all_met:
            # Progress to next stage!
            self.state.developmental_stage = next_stage
            logger.info(f"Mind has advanced to {next_stage.name} stage!")

            # Update all networks
            for network in self.networks.values():
                network.update_developmental_stage(next_stage)

            # Update need system
            self.need_system.update_developmental_stage(next_stage)

            # Announce developmental progress
            progress_message = NetworkMessage(
                sender="mind",
                receiver="mind",  # Mind gets its own message
                message_type="development",
                content={
                    "previous_stage": DevelopmentalStage(next_stage_value - 1).name,
                    "new_stage": next_stage.name,
                    "milestones": milestone_progress,
                },
                priority=1.0,  # Highest priority
                developmental_stage=next_stage,
            )

            # Process our own developmental message
            self._process_mind_message(progress_message)

            # Form a memory of this significant event
            self._form_memory({
                "type": "developmental_progress",
                "previous_stage": DevelopmentalStage(next_stage_value - 1).name,
                "new_stage": next_stage.name,
                "milestones": milestone_progress,
            })

        self.last_developmental_check = current_time

    def _self_reflection(self) -> None:
        """Perform self-reflection based on developmental capacity."""
        # Skip if self-awareness is too low
        if self.self_awareness_level < 0.3:
            return

        # Self-reflection increases with self-awareness and developmental stage
        reflection_depth = self.self_awareness_level * 0.2 * self.state.developmental_stage.value

        # Update performance metrics
        # Memory utilization - ratio of long-term to short-term memory
        if self.short_term_memory:
            ltm_ratio = len(self.long_term_memory) / (len(self.short_term_memory) + len(self.long_term_memory))
            self.performance_metrics["memory_utilization"] = ltm_ratio

        # Belief consistency - inverse of contradiction ratio
        total_beliefs = len(self.belief_network.beliefs)
        if total_beliefs > 1:
            contradiction_ratio = len(self.belief_network.contradictions) / (total_beliefs * (total_beliefs - 1) / 2)
            self.performance_metrics["belief_consistency"] = 1.0 - min(1.0, contradiction_ratio * 5)

        # Need balance - inverse of variation in need intensities
        need_intensities = [need.intensity for need in self.need_system.needs.values()]
        if need_intensities:
            need_variance = np.var(need_intensities) if len(need_intensities) > 1 else 0.0
            self.performance_metrics["need_balance"] = 1.0 - min(1.0, need_variance * 3)

        # Apply insights from self-reflection at higher development stages
        if reflection_depth > 0.5 and self.state.developmental_stage.value >= DevelopmentalStage.CHILD.value:
            # If poor memory utilization, boost memory consolidation
            if self.performance_metrics["memory_utilization"] < 0.3:
                # Trigger extra memory consolidation
                self._consolidate_memories()

            # If poor belief consistency, resolve contradictions
            if self.performance_metrics["belief_consistency"] < 0.4:
                self.belief_network.resolve_contradictions()

            # If poor need balance, adjust need priorities
            if self.performance_metrics["need_balance"] < 0.3:
                # Find most intense need
                most_intense = max(self.need_system.needs.items(), key=lambda x: x[1].intensity)

                # Satisfy it slightly to maintain balance
                self.need_system.satisfy_need(most_intense[0], 0.1)

    def get_state(self) -> MindState:
        """Get the current state of the mind.
        
        Returns:
            Current mind state

        """
        return self.state

    def get_observable_state(self) -> ObservableState:
        """Get the observable state of the mind.
        
        Creates a representation of what would be externally observable
        about the mind's state, rather than its internal state.
        
        Returns:
            Observable state

        """
        # Calculate apparent mood from emotional state
        apparent_mood = sum([
            self.state.emotional_state.get(EmotionType.JOY, 0) * 1.0,
            self.state.emotional_state.get(EmotionType.TRUST, 0) * 0.8,
            self.state.emotional_state.get(EmotionType.SADNESS, 0) * -0.8,
            self.state.emotional_state.get(EmotionType.FEAR, 0) * -0.6,
            self.state.emotional_state.get(EmotionType.ANGER, 0) * -1.0,
        ])

        # Clamp apparent mood to [-1, 1]
        apparent_mood = max(-1.0, min(1.0, apparent_mood))

        # Get recent emotions
        recent_emotions = [
            Emotion(name=name, intensity=intensity)
            for name, intensity in self.state.emotional_state.items()
            if intensity > 0.2  # Only include significant emotions
        ]

        # Calculate expressed needs based on current needs
        expressed_needs = self.need_system.get_expressed_needs(threshold=0.5)

        # Determine vocalization appropriate for developmental stage
        vocalization = self._generate_age_appropriate_vocalization()

        return ObservableState(
            apparent_mood=apparent_mood,
            energy_level=self.state.energy_level,
            current_focus=self.state.current_focus,
            recent_emotions=recent_emotions,
            expressed_needs=expressed_needs,
            developmental_stage=self.state.developmental_stage,
            vocalization=vocalization,
            age_appropriate_behaviors=self._get_age_appropriate_behaviors(),
        )

    def _generate_age_appropriate_vocalization(self) -> str:
        """Generate an age-appropriate vocalization based on developmental stage.
        
        Returns:
            Age-appropriate vocalization

        """
        # If we have a language network, use it
        if "language" in self.networks:
            # Get the most recent output from language network
            recent_utterances = self.networks["language"].state.parameters.get("recent_utterances", [])
            if recent_utterances:
                # Get most recent utterance
                latest = recent_utterances[-1]
                return f"{latest.get('type', 'says')} \"{latest.get('text', '')}\""

        # Fall back to default vocalizations if no language network or no recent utterances
        if self.state.developmental_stage == DevelopmentalStage.INFANT:
            # Infant vocalizations - cries, coos, etc.
            sounds = ["cries", "coos", "babbles", "gurgles", "fusses"]
            return random.choice(sounds)

        if self.state.developmental_stage == DevelopmentalStage.TODDLER:
            # Toddler speech - single words and simple phrases
            vocab = list(self.developmental_milestones["vocabulary_learned"])
            if not vocab:
                return "babbles simple syllables"

            if len(vocab) < 5 or random.random() < 0.5:
                # Single word
                return f'says "{random.choice(vocab)}"'
            # Simple phrase (2-3 words)
            phrase_len = min(3, len(vocab))
            phrase = " ".join(random.sample(vocab, phrase_len))
            return f'says "{phrase}"'

        if self.state.developmental_stage == DevelopmentalStage.CHILD:
            # Child speech - simple sentences
            vocab = list(self.developmental_milestones["vocabulary_learned"])
            if len(vocab) < 10:
                # Fall back to toddler speech
                return self._generate_age_appropriate_vocalization()

            # Simple sentence templates
            templates = [
                "I want {}",
                "I like {}",
                "I see {}",
                "Can I have {}?",
                "Where is {}?",
                "This is {}",
                "{} is mine",
                "I don't like {}",
            ]

            template = random.choice(templates)
            words = random.sample(vocab, min(3, len(vocab)))
            object_phrase = " ".join(words)
            return f'says "{template.format(object_phrase)}"'

        if self.state.developmental_stage == DevelopmentalStage.ADOLESCENT:
            # More complex sentences
            # (In a real implementation, this would use more sophisticated language generation)
            return "expresses thoughts in complete sentences"

        # MATURE
        return "communicates fluently"

    def _get_age_appropriate_behaviors(self) -> List[str]:
        """Get a list of age-appropriate behaviors for the current developmental stage.
        
        Returns:
            List of behavior descriptions

        """
        behaviors = []

        if self.state.developmental_stage == DevelopmentalStage.INFANT:
            infant_behaviors = [
                "makes eye contact",
                "reaches for objects",
                "responds to voices",
                "shows interest in faces",
                "startles at loud noises",
                "smiles responsively",
                "tracks moving objects",
                "recognizes familiar faces",
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(infant_behaviors, random.randint(1, 2))

        elif self.state.developmental_stage == DevelopmentalStage.TODDLER:
            toddler_behaviors = [
                "points at objects of interest",
                "imitates simple actions",
                "explores surroundings",
                "shows interest in peers",
                "expresses emotions more clearly",
                "attempts simple tasks",
                "shows preference for certain objects",
                "demonstrates simple problem-solving",
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(toddler_behaviors, random.randint(1, 2))

        elif self.state.developmental_stage == DevelopmentalStage.CHILD:
            child_behaviors = [
                "engages in symbolic play",
                "follows simple instructions",
                "shows more complex emotions",
                "attempts to assert independence",
                "shows interest in rules and order",
                "asks many questions",
                "develops friendships",
                "demonstrates logical thinking",
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(child_behaviors, random.randint(1, 2))

        elif self.state.developmental_stage == DevelopmentalStage.ADOLESCENT:
            adolescent_behaviors = [
                "shows abstract thinking",
                "contemplates hypothetical scenarios",
                "demonstrates complex emotional understanding",
                "shows more independence",
                "develops personal identity",
                "exhibits more complex social interactions",
                "shows interest in deeper concepts",
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(adolescent_behaviors, random.randint(1, 2))

        else:  # MATURE
            mature_behaviors = [
                "demonstrates full emotional regulation",
                "exhibits complex reasoning",
                "shows nuanced social awareness",
                "displays integrated sense of self",
                "demonstrates abstract problem-solving",
                "exhibits creative thinking",
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(mature_behaviors, random.randint(1, 2))

        return behaviors

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by its ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory object or None if not found

        """
        # Check short-term memory
        for memory in self.short_term_memory:
            if memory.id == memory_id:
                memory.access()  # Update access time and strengthen
                return memory

        # Check long-term memory
        for memory in self.long_term_memory:
            if memory.id == memory_id:
                memory.access()  # Update access time and strengthen
                return memory

        return None

    def get_memories_by_tag(self, tag: str) -> List[Memory]:
        """Get memories with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching memories

        """
        matching_memories = []

        # Check short-term memory
        for memory in self.short_term_memory:
            if tag in memory.tags:
                memory.access()  # Update access time and strengthen
                matching_memories.append(memory)

        # Check long-term memory
        for memory in self.long_term_memory:
            if tag in memory.tags:
                memory.access()  # Update access time and strengthen
                matching_memories.append(memory)

        return matching_memories

    def search_memories(self, query: Dict[str, Any]) -> List[Memory]:
        """Search memories based on content criteria.
        
        Args:
            query: Dictionary of search criteria
            
        Returns:
            List of matching memories

        """
        matching_memories = []

        # Simple search implementation - check each memory
        all_memories = self.short_term_memory + self.long_term_memory

        for memory in all_memories:
            matches = True

            # Check each query criterion
            for key, value in query.items():
                # Special handling for 'type' which is at the top level
                if key == "type":
                    if memory.content.get("type") != value:
                        matches = False
                        break

                # Special handling for 'tags' which is a list
                elif key == "tags":
                    if not all(tag in memory.tags for tag in value):
                        matches = False
                        break

                # Special handling for 'emotional_valence' with range
                elif key == "emotional_valence":
                    if isinstance(value, dict):
                        min_val = value.get("min", -1.0)
                        max_val = value.get("max", 1.0)

                        if not (min_val <= memory.emotional_valence <= max_val):
                            matches = False
                            break
                    elif memory.emotional_valence != value:
                        matches = False
                        break

                # Check nested content
                elif key in memory.content:
                    if memory.content[key] != value:
                        matches = False
                        break

                # If key not found, it's not a match
                else:
                    matches = False
                    break

            if matches:
                memory.access()  # Update access time and strengthen
                matching_memories.append(memory)

        return matching_memories

    def get_belief_by_statement(self, subject: str, predicate: str, object: str) -> Optional[Belief]:
        """Get a belief by its statement components.
        
        Args:
            subject: Subject of the belief
            predicate: Predicate of the belief
            object: Object of the belief
            
        Returns:
            Matching belief or None if not found

        """
        for belief_id, belief in self.belief_network.beliefs.items():
            if (belief.subject == subject and
                belief.predicate == predicate and
                belief.object == object):
                return belief

        return None

    def get_beliefs_about(self, subject: str) -> List[Belief]:
        """Get all beliefs about a specific subject.
        
        Args:
            subject: Subject to search for
            
        Returns:
            List of beliefs about the subject

        """
        matching_beliefs = []

        for belief in self.belief_network.beliefs.values():
            if belief.subject == subject:
                matching_beliefs.append(belief)

        return matching_beliefs

    def get_related_beliefs(self, belief_id: str) -> List[Tuple[Belief, float]]:
        """Get beliefs related to a specific belief.
        
        Args:
            belief_id: ID of the belief to get relations for
            
        Returns:
            List of tuples containing (related_belief, relationship_strength)

        """
        if belief_id not in self.belief_network.beliefs:
            return []

        related_beliefs = []

        # Get relationship data
        relationships = self.belief_network.get_related_beliefs(belief_id)

        # Get actual belief objects
        for related_id, strength in relationships:
            if related_id in self.belief_network.beliefs:
                related_beliefs.append((self.belief_network.beliefs[related_id], strength))

        return related_beliefs

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics of the mind.
        
        Returns:
            Dictionary of performance metrics

        """
        return self.performance_metrics

    def save_state(self, directory: str = "saved_models", format: str = "pytorch") -> bool:
        """Save the current state of the mind and all networks.
        
        Args:
            directory: Directory to save models
            format: Model save format ("pytorch", "torchscript", "onnx")
            
        Returns:
            True if saved successfully, False otherwise

        """
        try:
            os.makedirs(directory, exist_ok=True)

            # Save each network
            for name, network in self.networks.items():
                network_path = os.path.join(directory, f"{name}.{format}")
                network.save_model(network_path, format=format)

            # Save mind state
            mind_state = {
                "developmental_stage": self.state.developmental_stage.value,
                "consciousness_level": self.state.consciousness_level,
                "emotional_state": {k.value: v for k, v in self.state.emotional_state.items()},
                "energy_level": self.state.energy_level,
                "simulation_time": self.simulation_time,
                "developmental_milestones": {
                    k: (list(v) if isinstance(v, set) else v)
                    for k, v in self.developmental_milestones.items()
                },
                "language_ability": {
                    "vocabulary_size": self.state.language_ability.vocabulary_size,
                    "sentence_complexity": self.state.language_ability.sentence_complexity,
                    "understanding_level": self.state.language_ability.understanding_level,
                    "expression_level": self.state.language_ability.expression_level,
                },
                "self_awareness_level": self.self_awareness_level,
                "performance_metrics": self.performance_metrics,
            }

            with open(os.path.join(directory, "mind_state.json"), "w") as f:
                json.dump(mind_state, f, indent=2)

            # Save memories
            def memory_to_dict(memory):
                memory_dict = memory.dict()
                # Convert datetime objects to ISO format strings
                if "creation_time" in memory_dict and isinstance(memory_dict["creation_time"], datetime):
                    memory_dict["creation_time"] = memory_dict["creation_time"].isoformat()
                if "last_access_time" in memory_dict and isinstance(memory_dict["last_access_time"], datetime):
                    memory_dict["last_access_time"] = memory_dict["last_access_time"].isoformat()
                return memory_dict

            memories_data = {
                "short_term": [memory_to_dict(memory) for memory in self.short_term_memory],
                "long_term": [memory_to_dict(memory) for memory in self.long_term_memory],
                "clusters": {cluster_id: cluster.to_dict() for cluster_id, cluster in self.memory_clusters.items()},
            }

            with open(os.path.join(directory, "memories.json"), "w") as f:
                json.dump(memories_data, f, indent=2)

            # Save belief network
            belief_data = self.belief_network.to_dict()

            with open(os.path.join(directory, "beliefs.json"), "w") as f:
                json.dump(belief_data, f, indent=2)

            # Save need system
            need_data = self.need_system.to_dict()

            with open(os.path.join(directory, "needs.json"), "w") as f:
                json.dump(need_data, f, indent=2)

            logger.info(f"Mind state saved to {directory}")
            return True

        except Exception as e:
            logger.error(f"Error saving mind state: {e!s}")
            return False

    def load_state(self, directory: str = "saved_models", format: str = "pytorch") -> None:
        """Load the mind state and all networks from disk.
        
        Args:
            directory: Directory to load models from
            
        Returns:
            True if loaded successfully, False otherwise

        """
        if not os.path.exists(directory):
            logger.warning(f"Save directory not found: {directory}")
            return False

        try:
            # Load mind state
            mind_state_path = os.path.join(directory, "mind_state.json")
            if os.path.exists(mind_state_path):
                with open(mind_state_path) as f:
                    mind_state = json.load(f)

                # Restore core state attributes
                if "developmental_stage" in mind_state:
                    self.state.developmental_stage = DevelopmentalStage(mind_state["developmental_stage"])

                if "consciousness_level" in mind_state:
                    self.state.consciousness_level = mind_state["consciousness_level"]

                if "emotional_state" in mind_state:
                    self.state.emotional_state = {
                        EmotionType(k): v for k, v in mind_state["emotional_state"].items()
                    }

                if "energy_level" in mind_state:
                    self.state.energy_level = mind_state["energy_level"]

                if "simulation_time" in mind_state:
                    self.simulation_time = mind_state["simulation_time"]

                if "developmental_milestones" in mind_state:
                    self.developmental_milestones = mind_state["developmental_milestones"]

                    # Convert sets from lists
                    if "emotions_experienced" in self.developmental_milestones:
                        emotions = []
                        for emotion_str in self.developmental_milestones["emotions_experienced"]:
                            try:
                                emotions.append(EmotionType(emotion_str))
                            except ValueError:
                                pass
                        self.developmental_milestones["emotions_experienced"] = set(emotions)

                    if "vocabulary_learned" in self.developmental_milestones:
                        self.developmental_milestones["vocabulary_learned"] = set(
                            self.developmental_milestones["vocabulary_learned"],
                        )

                if "language_ability" in mind_state:
                    self.state.language_ability = LanguageAbility(
                        **mind_state["language_ability"],
                    )

                if "self_awareness_level" in mind_state:
                    self.self_awareness_level = mind_state["self_awareness_level"]

                if "performance_metrics" in mind_state:
                    self.performance_metrics = mind_state["performance_metrics"]

            # Load memories
            memories_path = os.path.join(directory, "memories.json")
            if os.path.exists(memories_path):
                with open(memories_path) as f:
                    memories_data = json.load(f)

                # Restore short-term memories
                self.short_term_memory = []
                for memory_dict in memories_data.get("short_term", []):
                    try:
                        # Convert stage name to enum
                        stage_name = memory_dict.pop("developmental_stage", "INFANT")
                        stage = DevelopmentalStage[stage_name]

                        # Convert timestamps
                        creation_time = datetime.fromisoformat(memory_dict.pop("creation_time"))
                        last_access_time = datetime.fromisoformat(memory_dict.pop("last_access_time"))

                        memory = Memory(
                            developmental_stage=stage,
                            creation_time=creation_time,
                            last_access_time=last_access_time,
                            **memory_dict,
                        )
                        self.short_term_memory.append(memory)
                    except Exception as e:
                        logger.warning(f"Error restoring memory: {e!s}")

                # Restore long-term memories
                self.long_term_memory = []
                for memory_dict in memories_data.get("long_term", []):
                    try:
                        # Convert stage name to enum
                        stage_name = memory_dict.pop("developmental_stage", "INFANT")
                        stage = DevelopmentalStage[stage_name]

                        # Convert timestamps
                        creation_time = datetime.fromisoformat(memory_dict.pop("creation_time"))
                        last_access_time = datetime.fromisoformat(memory_dict.pop("last_access_time"))

                        memory = Memory(
                            developmental_stage=stage,
                            creation_time=creation_time,
                            last_access_time=last_access_time,
                            **memory_dict,
                        )
                        self.long_term_memory.append(memory)
                    except Exception as e:
                        logger.warning(f"Error restoring memory: {e!s}")

                # Restore memory clusters
                self.memory_clusters = {}
                for cluster_id, cluster_dict in memories_data.get("clusters", {}).items():
                    try:
                        # Convert stage name to enum
                        stage_name = cluster_dict.pop("developmental_stage", "INFANT")
                        stage = DevelopmentalStage[stage_name]

                        # Convert timestamps
                        creation_time = datetime.fromisoformat(cluster_dict.pop("creation_time"))
                        last_access_time = datetime.fromisoformat(cluster_dict.pop("last_access_time"))

                        cluster = MemoryCluster(
                            id=cluster_id,
                            developmental_stage=stage,
                            creation_time=creation_time,
                            last_access_time=last_access_time,
                            **cluster_dict,
                        )
                        self.memory_clusters[cluster_id] = cluster
                    except Exception as e:
                        logger.warning(f"Error restoring memory cluster: {e!s}")

            # Load belief network
            beliefs_path = os.path.join(directory, "beliefs.json")
            if os.path.exists(beliefs_path):
                with open(beliefs_path) as f:
                    belief_data = json.load(f)

                self.belief_network = BeliefNetwork.from_dict(belief_data)

            # Load need system
            needs_path = os.path.join(directory, "needs.json")
            if os.path.exists(needs_path):
                with open(needs_path) as f:
                    need_data = json.load(f)

                # Restore needs
                if "needs" in need_data:
                    self.need_system.needs = {}
                    for name, need_dict in need_data["needs"].items():
                        try:
                            last_update = datetime.fromisoformat(need_dict.pop("last_update"))
                            self.need_system.needs[name] = Need(
                                name=name,
                                last_update=last_update,
                                **need_dict,
                            )
                        except Exception as e:
                            logger.warning(f"Error restoring need {name}: {e!s}")

                # Restore priorities
                if "need_priorities" in need_data:
                    self.need_system.need_priorities = need_data["need_priorities"]

                # Initialize histories
                for name in self.need_system.needs:
                    if name not in self.need_system.need_history:
                        self.need_system.need_history[name] = []
                    if name not in self.need_system.need_satisfaction_history:
                        self.need_system.need_satisfaction_history[name] = []

            # Load networks
            for name, network in self.networks.items():
                network_path = os.path.join(directory, f"{name}.pt")
                if os.path.exists(network_path):
                    success = network.load_model(network_path)
                    if not success:
                        logger.warning(f"Failed to load network: {name}")
                else:
                    logger.warning(f"Network model not found: {name}")

            logger.info(f"Mind state loaded from {directory}")
            return True

        except Exception as e:
            logger.error(f"Error loading mind state: {e!s}")

        return False
