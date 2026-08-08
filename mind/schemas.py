"""Schemas for mind simulation components.

This module defines the data structures specific to the mind simulation,
including emotional states, language abilities, and mind states.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from core.schemas import DevelopmentalStage


class EmotionType(str, Enum):
    """Types of emotions experienced by the mind."""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    CONFUSION = "confusion"
    INTEREST = "interest"
    BOREDOM = "boredom"

class Emotion(BaseModel):
    """Representation of an emotion in the mind."""

    name: EmotionType
    intensity: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert emotion to dictionary representation."""
        return {
            "name": self.name.value,
            "intensity": self.intensity,
            "timestamp": self.timestamp.isoformat(),
        }

class LanguageAbility(BaseModel):
    """Representation of language abilities in the mind."""

    vocabulary_size: int = 0
    sentence_complexity: float = Field(ge=0.0, le=1.0, default=0.0)
    understanding_level: float = Field(ge=0.0, le=1.0, default=0.0)
    expression_level: float = Field(ge=0.0, le=1.0, default=0.0)

    def generate_vocalization(self) -> str:
        """Generate a vocalization based on current language ability.
        
        Returns:
            Description of the vocalization

        """
        if self.vocabulary_size == 0:
            return "pre-linguistic sounds"
        if self.vocabulary_size < 10:
            return "single words"
        if self.vocabulary_size < 50:
            return "simple phrases"
        if self.vocabulary_size < 200:
            return "simple sentences"
        if self.vocabulary_size < 500:
            return "complex sentences"
        return "fluent speech"

    def to_dict(self) -> Dict:
        """Convert language ability to dictionary representation."""
        return {
            "vocabulary_size": self.vocabulary_size,
            "sentence_complexity": self.sentence_complexity,
            "understanding_level": self.understanding_level,
            "expression_level": self.expression_level,
        }

class MindState(BaseModel):
    """Overall state of the mind."""

    consciousness_level: float = Field(ge=0.0, le=1.0)
    emotional_state: Dict[EmotionType, float] = Field(default_factory=dict)
    current_focus: Optional[str] = None
    energy_level: float = Field(ge=0.0, le=1.0)
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT
    language_ability: Optional[LanguageAbility] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert mind state to dictionary representation."""
        return {
            "consciousness_level": self.consciousness_level,
            "emotional_state": {k.value: v for k, v in self.emotional_state.items()},
            "current_focus": self.current_focus,
            "energy_level": self.energy_level,
            "developmental_stage": self.developmental_stage.name,
            "language_ability": self.language_ability.to_dict() if self.language_ability else None,
            "timestamp": self.timestamp.isoformat(),
        }

class ObservableState(BaseModel):
    """Observable state of the mind by external entities."""

    apparent_mood: float = Field(ge=-1.0, le=1.0)
    energy_level: float = Field(ge=0.0, le=1.0)
    current_focus: Optional[str] = None
    recent_emotions: List[Emotion] = Field(default_factory=list)
    expressed_needs: Dict[str, float] = Field(default_factory=dict)
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT
    vocalization: Optional[str] = None
    age_appropriate_behaviors: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert observable state to dictionary representation."""
        return {
            "apparent_mood": self.apparent_mood,
            "energy_level": self.energy_level,
            "current_focus": self.current_focus,
            "recent_emotions": [e.to_dict() for e in self.recent_emotions],
            "expressed_needs": self.expressed_needs,
            "developmental_stage": self.developmental_stage.name,
            "vocalization": self.vocalization,
            "age_appropriate_behaviors": self.age_appropriate_behaviors,
        }

    def get_developmental_description(self) -> str:
        """Get a human-readable description of developmental state.
        
        Returns:
            Text description of developmental state

        """
        stage_descriptions = {
            DevelopmentalStage.INFANT: "an infant (0-12 months)",
            DevelopmentalStage.TODDLER: "a toddler (1-3 years)",
            DevelopmentalStage.CHILD: "a child (3-12 years)",
            DevelopmentalStage.ADOLESCENT: "an adolescent (12-18 years)",
            DevelopmentalStage.MATURE: "a mature individual (18+ years)",
        }

        return f"Developmentally equivalent to {stage_descriptions.get(self.developmental_stage)}"
