"""Perception neural network implementation.

This network processes sensory inputs and builds increasingly complex
perceptual representations as development progresses.
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from torch import nn

from core.neural_network import NeuralNetwork
from core.schemas import DevelopmentalStage, NetworkMessage, TextOutput, VectorOutput

# Configure logging
logger = logging.getLogger(__name__)

class PerceptionNetwork(NeuralNetwork):
    """Perception network that processes sensory inputs.
    
    This network handles visual, auditory, and other sensory inputs,
    building increasingly complex perceptual representations as the
    mind develops through different stages.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64):
        """Initialize the perception network.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output vectors

        """
        super().__init__(name="perception", input_dim=input_dim, output_dim=output_dim)

        # Visual processing network
        self.visual_processor = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )

        # Auditory processing network
        self.auditory_processor = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )

        # Integration network
        self.integration_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

        # Attention mechanism (develops with maturity)
        self.attention = nn.Linear(hidden_dim // 2, 2)  # Attention weights for visual and auditory

        # Object recognition capacity (develops with stage)
        self.object_recognition = 0.2  # Starts basic, improves with development

        # Pattern recognition capacity (develops with stage)
        self.pattern_recognition = 0.1  # Starts minimal, improves with development

        # Recent perceptions
        self.recent_perceptions = []

        # Attentional focus (what aspect is being attended to)
        self.attentional_focus = "visual"  # Default focus

        # Initialize state parameters
        self.update_state({
            "object_recognition": self.object_recognition,
            "pattern_recognition": self.pattern_recognition,
            "attentional_focus": self.attentional_focus,
            "recent_perceptions": [],
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.
        
        Args:
            x: Input tensor representing sensory input
            
        Returns:
            Output tensor representing processed perception

        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        # Split input into visual and auditory components
        visual_input = x[:, :self.input_dim // 2]
        auditory_input = x[:, self.input_dim // 2:]

        # Process each sensory modality
        visual_features = self.visual_processor(visual_input)
        auditory_features = self.auditory_processor(auditory_input)

        # Combine features
        combined_features = torch.cat([visual_features, auditory_features], dim=1)

        # Apply attention if beyond infant stage
        if self.developmental_stage.value > DevelopmentalStage.INFANT.value:
            attention_weights = torch.softmax(self.attention(combined_features), dim=1)

            # Apply attention weights
            visual_features = visual_features * attention_weights[:, 0].unsqueeze(1)
            auditory_features = auditory_features * attention_weights[:, 1].unsqueeze(1)
            combined_features = torch.cat([visual_features, auditory_features], dim=1)

            # Update attentional focus based on weights
            with torch.no_grad():
                avg_weights = attention_weights.mean(dim=0)
                self.attentional_focus = "visual" if avg_weights[0] > avg_weights[1] else "auditory"

        # Integrate features
        output = self.integration_network(combined_features)

        # Apply developmental stage effects
        if self.object_recognition < 0.5:
            # Limited object recognition makes perception more fuzzy
            noise = torch.randn_like(output) * (0.5 - self.object_recognition)
            output = torch.clamp(output + noise, 0, 1)

        return output

    def process_message(self, message: NetworkMessage) -> Optional[VectorOutput]:
        """Process a message from another neural network.
        
        Args:
            message: Message from another network
            
        Returns:
            Optional vector output as response

        """
        # Process different message types
        if message.message_type == "sensory_input":
            # Process raw sensory input
            sensory_data = None

            if "visual" in message.content and "auditory" in message.content:
                # Get visual and auditory data
                visual_data = message.content["visual"]
                auditory_data = message.content["auditory"]

                # Ensure they're the right size
                if len(visual_data) > self.input_dim // 2:
                    visual_data = visual_data[:self.input_dim // 2]
                elif len(visual_data) < self.input_dim // 2:
                    visual_data = visual_data + [0.0] * (self.input_dim // 2 - len(visual_data))

                if len(auditory_data) > self.input_dim // 2:
                    auditory_data = auditory_data[:self.input_dim // 2]
                elif len(auditory_data) < self.input_dim // 2:
                    auditory_data = auditory_data + [0.0] * (self.input_dim // 2 - len(auditory_data))

                # Combine data
                sensory_data = visual_data + auditory_data

            elif "visual" in message.content:
                # Only visual data
                visual_data = message.content["visual"]

                # Ensure right size
                if len(visual_data) > self.input_dim // 2:
                    visual_data = visual_data[:self.input_dim // 2]
                elif len(visual_data) < self.input_dim // 2:
                    visual_data = visual_data + [0.0] * (self.input_dim // 2 - len(visual_data))

                # Create empty auditory data
                auditory_data = [0.0] * (self.input_dim // 2)

                # Combine data
                sensory_data = visual_data + auditory_data

            elif "auditory" in message.content:
                # Only auditory data
                auditory_data = message.content["auditory"]

                # Ensure right size
                if len(auditory_data) > self.input_dim // 2:
                    auditory_data = auditory_data[:self.input_dim // 2]
                elif len(auditory_data) < self.input_dim // 2:
                    auditory_data = auditory_data + [0.0] * (self.input_dim // 2 - len(auditory_data))

                # Create empty visual data
                visual_data = [0.0] * (self.input_dim // 2)

                # Combine data
                sensory_data = visual_data + auditory_data

            # Process sensory data if available
            if sensory_data:
                # Convert to tensor and process
                input_tensor = torch.tensor(sensory_data, dtype=torch.float32)
                with torch.no_grad():
                    output_tensor = self.forward(input_tensor.unsqueeze(0))

                # Extract perceptual information
                perception_info = self._extract_perception(output_tensor[0], message.content)

                # Remember this perception
                self._remember_perception(perception_info)

                # Send perception to emotions network if available
                if "emotional_valence" in perception_info:
                    emotion_message = NetworkMessage(
                        sender=self.name,
                        receiver="emotions",
                        message_type="perception",
                        content={
                            "stimulus": perception_info.get("description", "sensory input"),
                            "valence": perception_info["emotional_valence"],
                            "intensity": perception_info.get("salience", 0.5),
                        },
                        priority=0.7,
                    )

                    # Add to state for the mind to retrieve
                    self.update_state({
                        "pending_messages": self.state.parameters.get("pending_messages", []) + [emotion_message.to_dict()],
                    })

                # Return processed perception
                return VectorOutput(
                    source=self.name,
                    data=output_tensor[0].tolist(),
                )

        elif message.message_type == "attention_request":
            # Change attentional focus
            if "focus" in message.content:
                focus = message.content["focus"]
                if focus in ["visual", "auditory"]:
                    self.attentional_focus = focus
                    self.update_state({"attentional_focus": focus})

                    return VectorOutput(
                        source=self.name,
                        data=[float(focus == "visual")] * (self.output_dim // 2) +
                             [float(focus == "auditory")] * (self.output_dim // 2),
                    )

        return None

    def _extract_perception(self, output: torch.Tensor, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meaningful perception from network output.
        
        Args:
            output: Raw network output tensor
            content: Original content that generated the perception
            
        Returns:
            Dictionary of perception information

        """
        # Basic perception info
        perception = {
            "timestamp": datetime.now().isoformat(),
            "output_summary": output.mean().item(),
            "output_variance": output.var().item(),
        }

        # Calculate emotional valence (-1 to 1)
        # Simple heuristic: higher activations in first half = positive, second half = negative
        half = self.output_dim // 2
        positive_activation = output[:half].mean().item()
        negative_activation = output[half:].mean().item()
        emotional_valence = positive_activation - negative_activation
        perception["emotional_valence"] = max(-1.0, min(1.0, emotional_valence * 2))

        # Calculate salience (how attention-grabbing)
        salience = output.abs().mean().item()
        perception["salience"] = salience

        # Add raw sensory content summaries
        if "visual" in content:
            perception["visual_content"] = content.get("visual_description", "visual input")

        if "auditory" in content:
            perception["auditory_content"] = content.get("auditory_description", "auditory input")

        # Generate description based on developmental stage
        description = None

        if self.developmental_stage == DevelopmentalStage.INFANT:
            # Infants perceive simple, high-contrast patterns
            if positive_activation > 0.7:
                description = "bright pleasant stimulus"
            elif negative_activation > 0.7:
                description = "intense uncomfortable stimulus"
            else:
                description = "mild sensory input"

        elif self.developmental_stage == DevelopmentalStage.TODDLER:
            # Toddlers recognize basic objects and simple patterns
            if "object" in content:
                description = f"recognized {content['object']}"
            elif positive_activation > negative_activation:
                description = "interesting sensory pattern"
            else:
                description = "confusing sensory input"

        elif self.developmental_stage.value >= DevelopmentalStage.CHILD.value:
            # Children and beyond have more sophisticated perception
            # This would typically use more advanced processing
            if "object" in content and "context" in content:
                description = f"{content['object']} in {content['context']}"
            elif "object" in content:
                description = f"clearly perceived {content['object']}"
            elif "context" in content:
                description = f"scene in {content['context']}"
            else:
                description = "complex sensory pattern"

        perception["description"] = description or "sensory input"

        return perception

    def _remember_perception(self, perception: Dict[str, Any]) -> None:
        """Remember a perception.
        
        Args:
            perception: Perception information to remember

        """
        # Add to recent perceptions
        self.recent_perceptions.append(perception)

        # Limit memory size based on developmental stage
        max_memory = 3 + (self.developmental_stage.value * 3)
        if len(self.recent_perceptions) > max_memory:
            self.recent_perceptions = self.recent_perceptions[-max_memory:]

        # Update state
        self.update_state({
            "recent_perceptions": self.recent_perceptions,
        })

    def autonomous_step(self) -> None:
        """Autonomous processing step.
        
        This function is called periodically by the mind to allow
        the network to perform autonomous processing.
        """
        # Look for patterns in recent perceptions
        if len(self.recent_perceptions) > 2 and self.pattern_recognition > 0.3:
            # Simple pattern detection simulation
            has_pattern = random.random() < self.pattern_recognition

            if has_pattern:
                pattern_message = NetworkMessage(
                    sender=self.name,
                    receiver="thoughts",
                    message_type="pattern",
                    content={
                        "pattern_strength": self.pattern_recognition,
                        "pattern_source": "recent_perceptions",
                        "perception_count": len(self.recent_perceptions),
                    },
                    priority=0.6,
                )

                # Add to state for the mind to retrieve
                self.update_state({
                    "pending_messages": self.state.parameters.get("pending_messages", []) + [pattern_message.to_dict()],
                })

    def update_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update the developmental stage of the network.
        
        As the network develops, object and pattern recognition improve,
        simulating the development of perceptual capabilities.
        
        Args:
            stage: New developmental stage

        """
        super().update_developmental_stage(stage)

        # Update perception parameters based on developmental stage
        stage_values = {
            DevelopmentalStage.INFANT: {
                "object_recognition": 0.2,
                "pattern_recognition": 0.1,
            },
            DevelopmentalStage.TODDLER: {
                "object_recognition": 0.4,
                "pattern_recognition": 0.3,
            },
            DevelopmentalStage.CHILD: {
                "object_recognition": 0.6,
                "pattern_recognition": 0.5,
            },
            DevelopmentalStage.ADOLESCENT: {
                "object_recognition": 0.8,
                "pattern_recognition": 0.7,
            },
            DevelopmentalStage.MATURE: {
                "object_recognition": 0.9,
                "pattern_recognition": 0.9,
            },
        }

        if stage in stage_values:
            self.object_recognition = stage_values[stage]["object_recognition"]
            self.pattern_recognition = stage_values[stage]["pattern_recognition"]

            self.update_state({
                "object_recognition": self.object_recognition,
                "pattern_recognition": self.pattern_recognition,
            })

    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network.
        
        Returns:
            Text representation of the network's current state

        """
        # Generate text based on current state and developmental stage
        if not self.recent_perceptions:
            return TextOutput(
                source=self.name,
                text="No recent perceptions to report.",
                confidence=0.5,
            )

        # Get most recent perception
        latest = self.recent_perceptions[-1]

        # Generate text based on developmental stage
        if self.developmental_stage == DevelopmentalStage.INFANT:
            text = f"Basic sensory processing: {latest.get('description', 'sensory input')}"
            confidence = 0.4

        elif self.developmental_stage == DevelopmentalStage.TODDLER:
            text = f"Perceiving: {latest.get('description', 'sensory input')}"
            if self.attentional_focus:
                text += f" (focusing on {self.attentional_focus})"
            confidence = 0.6

        else:
            # More mature stages include object and pattern recognition info
            text = f"Perceiving: {latest.get('description', 'sensory input')}"
            if self.attentional_focus:
                text += f" (attending to {self.attentional_focus})"

            # Add recognition capabilities
            text += f" Object recognition: {int(self.object_recognition * 100)}%, "
            text += f"Pattern recognition: {int(self.pattern_recognition * 100)}%"

            confidence = 0.7

        return TextOutput(
            source=self.name,
            text=text,
            confidence=confidence,
        )

    def clone_with_growth(self, growth_factor: float = 1.2, min_dim: int = 8) -> "PerceptionNetwork":
        """Create a larger clone of this network with scaled dimensions.
        
        Args:
            growth_factor: Factor to scale dimensions by
            min_dim: Minimum dimension size to ensure
            
        Returns:
            Larger clone of this network with scaled dimensions

        """
        # Calculate new dimensions
        new_input_dim = max(min_dim, int(self.input_dim * growth_factor))
        new_hidden_dim = max(min_dim * 2, int(self.visual_processor[1].out_features * 2 * growth_factor))
        new_output_dim = max(min_dim, int(self.output_dim * growth_factor))

        # Create new network with expanded dimensions
        new_network = PerceptionNetwork(
            input_dim=new_input_dim,
            hidden_dim=new_hidden_dim,
            output_dim=new_output_dim,
        )

        # Transfer perception properties
        new_network.object_recognition = self.object_recognition
        new_network.pattern_recognition = self.pattern_recognition
        new_network.attentional_focus = self.attentional_focus
        new_network.recent_perceptions = copy.deepcopy(self.recent_perceptions)

        # Transfer growth metrics
        new_network.growth_metrics = copy.deepcopy(self.growth_metrics)
        new_network.experience_count = self.experience_count

        # Record growth event
        new_network.growth_history = copy.deepcopy(self.growth_history)
        new_network.growth_history.append(NeuralGrowthRecord(
            event_type="network_expansion",
            layer_affected="all_processors",
            old_shape=[self.input_dim, self.visual_processor[1].out_features * 2, self.output_dim],
            new_shape=[new_input_dim, new_hidden_dim, new_output_dim],
            growth_factor=growth_factor,
            trigger="clone_with_growth",
            developmental_stage=self.developmental_stage,
        ))

        logger.info(
            f"PerceptionNetwork cloned with growth factor {growth_factor}: "
            f"({self.input_dim}, {self.visual_processor[1].out_features * 2}, {self.output_dim}) → "
            f"({new_input_dim}, {new_hidden_dim}, {new_output_dim})",
        )

        return new_network
