"""Emotions neural network implementation.

This network processes emotional states and responses based on interactions and experiences.
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from core.neural_network import NeuralNetwork
from core.schemas import DevelopmentalStage, NetworkMessage, TextOutput, VectorOutput
from mind.schemas import EmotionType

# Configure logging
logger = logging.getLogger(__name__)

class EmotionsNetwork(NeuralNetwork):
    """Emotions network that processes and generates emotional responses.
    
    This network maintains the emotional state of the mind and responds
    to external stimuli with appropriate emotional reactions that evolve
    with developmental stage.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 32):
        """Initialize the emotions network.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output vectors

        """
        super().__init__(name="emotions", input_dim=input_dim, output_dim=output_dim)

        # Simple feed-forward network for processing emotional inputs
        self.emotion_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        # Emotional state - starts with basic emotions at infant stage
        self.emotional_state = {
            EmotionType.JOY: 0.3,
            EmotionType.FEAR: 0.1,
            EmotionType.SURPRISE: 0.2,
            EmotionType.TRUST: 0.3,
        }

        # Emotional memory - stores recent emotional events
        self.emotional_memory: List[Dict[str, Any]] = []

        # Emotional reactivity (decreases with development/maturity)
        self.reactivity = 0.8

        # Emotional regulation (increases with development/maturity)
        self.regulation = 0.2

        # Initialize state parameters
        self.update_state({
            "emotional_state": {k.value: v for k, v in self.emotional_state.items()},
            "reactivity": self.reactivity,
            "regulation": self.regulation,
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.
        
        Args:
            x: Input tensor representing emotional stimulus
            
        Returns:
            Output tensor representing emotional response

        """
        # Process the emotional input
        output = self.emotion_processor(x)

        # Scale output by reactivity and regulation
        effective_output = output * self.reactivity
        if self.regulation > 0.5:
            # Apply dampening to extreme values when regulation is higher
            dampening = torch.abs(effective_output - 0.5) * (self.regulation - 0.5) * 2
            effective_output = effective_output - (dampening * torch.sign(effective_output - 0.5))

        return effective_output

    def process_message(self, message: NetworkMessage) -> Optional[VectorOutput]:
        """Process a message from another neural network.
        
        Args:
            message: Message from another network
            
        Returns:
            Optional vector output as response

        """
        # Process different message types
        if message.message_type == "perception":
            # Process perception input as emotional stimulus
            if "stimulus" in message.content and "valence" in message.content:
                stimulus = message.content["stimulus"]
                valence = float(message.content["valence"])  # -1 to 1
                intensity = float(message.content.get("intensity", 0.5))

                # Create a simple vector representation of the emotional stimulus
                stimulus_vector = torch.zeros(self.input_dim)

                # Encode valence in the first part of the vector
                valence_idx = int((valence + 1) / 2 * (self.input_dim // 3))
                stimulus_vector[valence_idx] = intensity

                # Process the stimulus
                with torch.no_grad():
                    response = self.forward(stimulus_vector.unsqueeze(0))

                # Update emotional state based on response
                self._update_emotional_state(stimulus, valence, intensity, response[0])

                # Remember this emotional event
                self._remember_emotional_event(stimulus, valence, intensity)

                return VectorOutput(
                    source=self.name,
                    data=response[0].tolist(),
                )

        elif message.message_type == "query":
            # Respond with current emotional state
            if "emotion" in message.content:
                emotion_name = message.content["emotion"]
                try:
                    emotion = EmotionType(emotion_name)
                    intensity = self.emotional_state.get(emotion, 0.0)

                    return VectorOutput(
                        source=self.name,
                        data=[intensity] * self.output_dim,  # Simple representation
                    )
                except ValueError:
                    pass

        return None

    def _update_emotional_state(self, stimulus: str, valence: float, intensity: float, response: torch.Tensor) -> None:
        """Update the emotional state based on a processed stimulus.
        
        Args:
            stimulus: Description of the emotional stimulus
            valence: Emotional valence (-1 to 1)
            intensity: Intensity of the stimulus (0 to 1)
            response: Network response to the stimulus

        """
        # Map valence to primary emotions
        if valence > 0.3:
            # Positive emotions
            self._update_emotion(EmotionType.JOY, response.mean().item() * intensity)
            self._update_emotion(EmotionType.TRUST, response.mean().item() * intensity * 0.7)

            # Decrease negative emotions
            self._update_emotion(EmotionType.SADNESS, -0.1 * intensity)
            self._update_emotion(EmotionType.FEAR, -0.1 * intensity)
            self._update_emotion(EmotionType.ANGER, -0.1 * intensity)

        elif valence < -0.3:
            # Negative emotions
            if valence < -0.7:
                # Very negative - anger or fear
                if random.random() < 0.5:
                    self._update_emotion(EmotionType.ANGER, response.mean().item() * intensity)
                else:
                    self._update_emotion(EmotionType.FEAR, response.mean().item() * intensity)
            else:
                # Moderately negative - sadness
                self._update_emotion(EmotionType.SADNESS, response.mean().item() * intensity)

            # Decrease positive emotions
            self._update_emotion(EmotionType.JOY, -0.1 * intensity)

        else:
            # Neutral valence - surprise or interest
            self._update_emotion(EmotionType.SURPRISE, response.mean().item() * intensity * 0.5)

            # More complex emotions become available at higher developmental stages
            if self.developmental_stage.value >= DevelopmentalStage.TODDLER.value:
                self._update_emotion(EmotionType.INTEREST, response.mean().item() * intensity * 0.3)

            if self.developmental_stage.value >= DevelopmentalStage.CHILD.value:
                self._update_emotion(EmotionType.ANTICIPATION, response.mean().item() * intensity * 0.2)

        # Update state parameters
        self.update_state({
            "emotional_state": {k.value: v for k, v in self.emotional_state.items()},
        })

        # Send an emotion update message to the mind
        emotion_update = NetworkMessage(
            sender=self.name,
            receiver="mind",
            message_type="emotion",
            content={
                "emotions": {k.value: v for k, v in self.emotional_state.items() if v > 0.2},
                "stimulus": stimulus,
                "valence": valence,
                "intensity": intensity,
            },
            priority=0.8,
        )

        # This would be sent to the mind, but since we don't have direct access,
        # we'll add it to the state for the mind to retrieve
        self.update_state({
            "pending_messages": self.state.parameters.get("pending_messages", []) + [emotion_update.to_dict()],
        })

    def _update_emotion(self, emotion: EmotionType, change: float) -> None:
        """Update a specific emotion intensity.
        
        Args:
            emotion: The emotion to update
            change: Amount to change the emotion intensity

        """
        current = self.emotional_state.get(emotion, 0.0)
        # Apply change, scaled by reactivity and regulated by regulation capability
        effective_change = change * self.reactivity
        if change > 0 and self.regulation > 0.5:
            # Higher regulation dampens increases in emotional intensity
            effective_change *= 2 - self.regulation

        new_value = max(0.0, min(1.0, current + effective_change))
        self.emotional_state[emotion] = new_value

    def _remember_emotional_event(self, stimulus: str, valence: float, intensity: float) -> None:
        """Remember an emotional event.
        
        Args:
            stimulus: Description of the emotional stimulus
            valence: Emotional valence (-1 to 1)
            intensity: Intensity of the stimulus (0 to 1)

        """
        # Create a memory entry
        memory = {
            "timestamp": datetime.now().isoformat(),
            "stimulus": stimulus,
            "valence": valence,
            "intensity": intensity,
            "emotional_state": {k.value: v for k, v in self.emotional_state.items() if v > 0.1},
        }

        # Add to memory
        self.emotional_memory.append(memory)

        # Limit memory size based on developmental stage
        max_memory = 5 + (self.developmental_stage.value * 5)
        if len(self.emotional_memory) > max_memory:
            self.emotional_memory = self.emotional_memory[-max_memory:]

        # Update state
        self.update_state({
            "emotional_memory": self.emotional_memory,
        })

    def autonomous_step(self) -> None:
        """Autonomous processing step.
        
        This function is called periodically by the mind to allow
        the network to perform autonomous processing.
        """
        # Natural emotional decay over time
        for emotion in list(self.emotional_state.keys()):
            # Different emotions decay at different rates
            decay_rate = 0.01
            if emotion == EmotionType.FEAR:
                decay_rate = 0.015
            elif emotion == EmotionType.SURPRISE:
                decay_rate = 0.03
            elif emotion == EmotionType.JOY:
                decay_rate = 0.005

            # Apply decay
            current = self.emotional_state.get(emotion, 0.0)
            if current > 0:
                self.emotional_state[emotion] = max(0.0, current - decay_rate)

                # Remove emotions that have decayed to 0
                if self.emotional_state[emotion] == 0:
                    del self.emotional_state[emotion]

        # Update state parameters
        self.update_state({
            "emotional_state": {k.value: v for k, v in self.emotional_state.items()},
        })

    def update_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update the developmental stage of the network.
        
        As the network develops, emotional regulation increases and
        reactivity decreases, simulating emotional maturation.
        
        Args:
            stage: New developmental stage

        """
        super().update_developmental_stage(stage)

        # Update reactivity and regulation based on developmental stage
        stage_values = {
            DevelopmentalStage.INFANT: {"reactivity": 0.8, "regulation": 0.2},
            DevelopmentalStage.TODDLER: {"reactivity": 0.7, "regulation": 0.4},
            DevelopmentalStage.CHILD: {"reactivity": 0.6, "regulation": 0.6},
            DevelopmentalStage.ADOLESCENT: {"reactivity": 0.5, "regulation": 0.7},
            DevelopmentalStage.MATURE: {"reactivity": 0.4, "regulation": 0.9},
        }

        if stage in stage_values:
            self.reactivity = stage_values[stage]["reactivity"]
            self.regulation = stage_values[stage]["regulation"]

            self.update_state({
                "reactivity": self.reactivity,
                "regulation": self.regulation,
            })

        # Unlock more complex emotions at higher developmental stages
        if stage == DevelopmentalStage.TODDLER:
            # Toddlers gain disgust and anticipation
            if EmotionType.DISGUST not in self.emotional_state:
                self.emotional_state[EmotionType.DISGUST] = 0.0
            if EmotionType.ANTICIPATION not in self.emotional_state:
                self.emotional_state[EmotionType.ANTICIPATION] = 0.0

        elif stage == DevelopmentalStage.CHILD:
            # Children gain more complex emotions
            if EmotionType.CONFUSION not in self.emotional_state:
                self.emotional_state[EmotionType.CONFUSION] = 0.0
            if EmotionType.INTEREST not in self.emotional_state:
                self.emotional_state[EmotionType.INTEREST] = 0.0

        elif stage == DevelopmentalStage.ADOLESCENT:
            # Adolescents gain boredom
            if EmotionType.BOREDOM not in self.emotional_state:
                self.emotional_state[EmotionType.BOREDOM] = 0.0

    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network.
        
        Returns:
            Text representation of the network's current state

        """
        # Get significant emotions (intensity > 0.3)
        significant_emotions = {k: v for k, v in self.emotional_state.items() if v > 0.3}

        if not significant_emotions:
            return TextOutput(
                source=self.name,
                text="Emotional state is neutral.",
                confidence=0.7,
            )

        # Sort emotions by intensity
        sorted_emotions = sorted(
            significant_emotions.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Generate text based on developmental stage
        text = ""
        if self.developmental_stage == DevelopmentalStage.INFANT:
            # Infants have simple emotional states
            if sorted_emotions:
                primary_emotion, intensity = sorted_emotions[0]
                text = f"Primarily feeling {primary_emotion.value} ({intensity:.1f})"

        elif self.developmental_stage == DevelopmentalStage.TODDLER:
            # Toddlers can express a primary and secondary emotion
            if len(sorted_emotions) >= 2:
                primary, p_intensity = sorted_emotions[0]
                secondary, s_intensity = sorted_emotions[1]
                text = f"Feeling {primary.value} ({p_intensity:.1f}) and {secondary.value} ({s_intensity:.1f})"
            elif sorted_emotions:
                primary, intensity = sorted_emotions[0]
                text = f"Feeling {primary.value} ({intensity:.1f})"

        else:
            # More mature stages can express complex emotional states
            emotion_texts = [f"{e.value} ({i:.1f})" for e, i in sorted_emotions[:3]]
            text = f"Emotional state: {', '.join(emotion_texts)}"

            # Add regulation information for more mature stages
            if self.developmental_stage.value >= DevelopmentalStage.CHILD.value:
                text += f" with {self.regulation:.1f} regulation capability"

        return TextOutput(
            source=self.name,
            text=text,
            confidence=max(0.5, min(sorted_emotions, key=lambda x: x[1])[1]),
        )

    def clone_with_growth(self, growth_factor: float = 1.2, min_dim: int = 8) -> "EmotionsNetwork":
        """Create a larger clone of this network with scaled dimensions.
        
        Args:
            growth_factor: Factor to scale dimensions by
            min_dim: Minimum dimension size to ensure
            
        Returns:
            Larger clone of this network with scaled dimensions

        """
        # Calculate new dimensions
        new_input_dim = max(min_dim, int(self.input_dim * growth_factor))
        new_hidden_dim = max(min_dim * 2, int(self.emotion_processor[1].out_features * growth_factor))
        new_output_dim = max(min_dim, int(self.output_dim * growth_factor))

        # Create new network with expanded dimensions
        new_network = EmotionsNetwork(
            input_dim=new_input_dim,
            hidden_dim=new_hidden_dim,
            output_dim=new_output_dim,
        )

        # Transfer emotional state
        new_network.emotional_state = copy.deepcopy(self.emotional_state)
        new_network.emotional_memory = copy.deepcopy(self.emotional_memory)
        new_network.reactivity = self.reactivity
        new_network.regulation = self.regulation

        # Transfer growth metrics
        new_network.growth_metrics = copy.deepcopy(self.growth_metrics)
        new_network.experience_count = self.experience_count

        # Record growth event
        new_network.growth_history = copy.deepcopy(self.growth_history)
        new_network.growth_history.append(NeuralGrowthRecord(
            event_type="network_expansion",
            layer_affected="emotion_processor",
            old_shape=[self.input_dim, self.emotion_processor[1].out_features, self.output_dim],
            new_shape=[new_input_dim, new_hidden_dim, new_output_dim],
            growth_factor=growth_factor,
            trigger="clone_with_growth",
            developmental_stage=self.developmental_stage,
        ))

        logger.info(
            f"EmotionsNetwork cloned with growth factor {growth_factor}: "
            f"({self.input_dim}, {self.emotion_processor[1].out_features}, {self.output_dim}) → "
            f"({new_input_dim}, {new_hidden_dim}, {new_output_dim})",
        )

        return new_network
