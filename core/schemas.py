"""Core schemas for the NeuralChild project.

This module defines the data structures and models used throughout the system,
establishing a consistent interface for component interaction.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum, auto
import uuid
import torch

class DevelopmentalStage(Enum):
    """Developmental stages for the artificial mind."""
    INFANT = 1      # 0-12 months equivalent
    TODDLER = 2     # 1-3 years equivalent
    CHILD = 3       # 3-12 years equivalent
    ADOLESCENT = 4  # 12-18 years equivalent
    MATURE = 5      # 18+ years equivalent

class NetworkMessage(BaseModel):
    """Message passed between neural networks.
    
    These messages form the basis of inter-network communication,
    allowing different parts of the mind to coordinate.
    """
    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str = "standard"  # standard, emotional, belief, etc.
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: float = 1.0  # Higher values indicate higher priority
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "developmental_stage": self.developmental_stage.name
        }
    
class NetworkState(BaseModel):
    """State of a neural network.
    
    Maintains the current state of a neural network, including
    its developmental weights and operational parameters.
    """
    name: str
    active: bool = True
    last_update: datetime = Field(default_factory=datetime.now)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    developmental_weights: Dict[DevelopmentalStage, float] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary for serialization."""
        return {
            "name": self.name,
            "active": self.active,
            "last_update": self.last_update.isoformat(),
            "parameters": self.parameters,
            "developmental_weights": {k.name: v for k, v in self.developmental_weights.items()}
        }
    
class Memory(BaseModel):
    """Memory representation for storing experiences.
    
    Memories are encoded experiences that can be stored and recalled
    by the mind, forming the basis for learning and development.
    """
    id: str
    content: Dict[str, Any]
    creation_time: datetime = Field(default_factory=datetime.now)
    last_access_time: datetime = Field(default_factory=datetime.now)
    strength: float = 1.0  # How strongly the memory is retained (decays over time)
    emotional_valence: float = 0.0  # Emotional significance (-1 to 1)
    tags: List[str] = Field(default_factory=list)
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT
    
    def access(self) -> None:
        """Access the memory, updating its strength and access time."""
        self.last_access_time = datetime.now()
        # Memory becomes stronger when accessed (hebbian learning)
        self.strength = min(5.0, self.strength + 0.1)
    
    def decay(self, amount: float = 0.01) -> None:
        """Decay the memory strength over time."""
        self.strength = max(0.0, self.strength - amount)
    
    def is_forgotten(self) -> bool:
        """Check if the memory has been forgotten (strength below threshold)."""
        return self.strength < 0.1
    
class VectorOutput(BaseModel):
    """Vector output from a neural network.
    
    Represents numerical output from a neural network for
    machine consumption and processing.
    """
    source: str
    data: List[float]
    timestamp: datetime = Field(default_factory=datetime.now)
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT
    
class TextOutput(BaseModel):
    """Text output from a neural network for human consumption.
    
    Represents natural language output that can be understood
    by humans interacting with the system.
    """
    source: str
    text: str
    confidence: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.now)
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT
    
class Belief(BaseModel):
    """Belief representation in the mind.
    
    Beliefs are mental constructs that represent the mind's
    understanding of concepts and relationships in the world.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    creation_time: datetime = Field(default_factory=datetime.now)
    last_update_time: datetime = Field(default_factory=datetime.now)
    supporting_memories: List[str] = Field(default_factory=list)  # IDs of supporting memories
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT
    
    def update_confidence(self, new_evidence: float) -> None:
        """Update the belief confidence based on new evidence."""
        # Simple Bayesian-inspired update
        self.confidence = (self.confidence + new_evidence) / 2
        self.last_update_time = datetime.now()
    
    def to_natural_language(self) -> str:
        """Convert the belief to natural language representation."""
        confidence_text = ""
        if self.confidence > 0.8:
            confidence_text = "I'm sure that "
        elif self.confidence > 0.5:
            confidence_text = "I think that "
        else:
            confidence_text = "I'm not sure, but maybe "
        
        return f"{confidence_text}{self.subject} {self.predicate} {self.object}"
    
class Need(BaseModel):
    """Representation of a need in the mind.
    
    Needs drive behavior and motivation, creating
    the impetus for action and learning.
    """
    name: str
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)
    last_update: datetime = Field(default_factory=datetime.now)
    satisfaction_level: float = Field(ge=0.0, le=1.0, default=0.5)
    
    def update_intensity(self, amount: float) -> None:
        """Update the need intensity."""
        self.intensity = max(0.0, min(1.0, self.intensity + amount))
        self.last_update = datetime.now()
    
    def satisfy(self, amount: float) -> None:
        """Satisfy the need by the specified amount."""
        self.satisfaction_level = min(1.0, self.satisfaction_level + amount)
        # Reduce intensity when satisfied
        self.intensity = max(0.0, self.intensity - (amount * 0.5))
        self.last_update = datetime.now()