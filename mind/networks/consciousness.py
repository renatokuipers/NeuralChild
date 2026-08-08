"""Consciousness neural network implementation.

This network integrates information from other networks to create a unified
awareness and self-model that develops over time.
"""

import logging
import random

import torch
from torch import nn

from core.neural_network import NeuralNetwork
from core.schemas import DevelopmentalStage, NetworkMessage, TextOutput, VectorOutput

# Configure logging
logger = logging.getLogger(__name__)


class ConsciousnessNetwork(NeuralNetwork):
    """Consciousness network that integrates awareness from other networks.

    Uses a recurrent neural network (RNN) architecture to maintain
    a sense of continuity and awareness over time, integrating inputs
    from all other networks into a unified conscious experience.
    """

    def __init__(
        self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64
    ):
        """Initialize the consciousness network.

        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output vectors

        """
        super().__init__(
            name="consciousness", input_dim=input_dim, output_dim=output_dim
        )

        # RNN for processing sequential inputs
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Self-model network - representation of self that grows more complex with development
        self.self_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Hidden state maintains continuity of consciousness
        self.hidden = None

        # Current awareness level (0-1)
        self.awareness_level = 0.2  # Start low for infant stage

        # Attention focus - which network is currently attended to
        self.attending_to = None

        # Self-awareness level (develops with stage)
        self.self_awareness = 0.1  # Start minimal for infant stage

        # Integration capacity - how well the consciousness integrates information
        self.integration_capacity = 0.2  # Starts low, increases with development

        # Recent network activations
        self.network_activations = {}

        # Initialize state parameters
        self.update_state(
            {
                "awareness_level": self.awareness_level,
                "attending_to": self.attending_to,
                "self_awareness": self.self_awareness,
                "integration_capacity": self.integration_capacity,
                "recent_inputs": [],
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: Input tensor

        Returns:
            Output tensor representing conscious state

        """
        # Ensure input is 3D for RNN [batch, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Initialize hidden state if None
        if self.hidden is None:
            self.hidden = torch.zeros(2, x.size(0), 128, device=x.device)

        # Process through RNN
        output, self.hidden = self.rnn(x, self.hidden)

        # Get last output
        last_output = output[:, -1, :]

        # Project to output dimension
        result = self.output_layer(last_output)

        # Update awareness level based on output activation
        self.awareness_level = torch.sigmoid(result.mean()).item()

        # Apply self-model to create self-awareness component
        if self.self_awareness > 0.1:
            self_component = self.self_model(last_output) * self.self_awareness
            # Blend self-model with external inputs
            result = result * (1 - self.self_awareness) + self_component

        # Update state
        self.update_state(
            {
                "awareness_level": self.awareness_level,
                "recent_inputs": self.state.parameters.get("recent_inputs", [])[-4:]
                + [x.detach().mean().item()],
            }
        )

        return result

    def process_message(self, message: NetworkMessage) -> VectorOutput | None:
        """Process a message from another neural network.

        Args:
            message: Message from another network

        Returns:
            Optional vector output as response

        """
        # Extract vector data from message content
        if "vector_data" in message.content and len(message.content["vector_data"]) > 0:
            # Adjust vector to expected size
            vector_data = message.content["vector_data"]
            if len(vector_data) > self.input_dim:
                vector_data = vector_data[: self.input_dim]
            elif len(vector_data) < self.input_dim:
                vector_data = vector_data + [0.0] * (self.input_dim - len(vector_data))

            # Convert to tensor and process
            input_tensor = torch.tensor(vector_data, dtype=torch.float32)
            output_tensor = self.forward(input_tensor.unsqueeze(0))

            # Update activations from this network
            self.network_activations[message.sender] = input_tensor.mean().item()

            # Update state to reflect attention to the sender
            self.attending_to = message.sender
            self.update_state({"attending_to": message.sender})

            # Return vector output
            return VectorOutput(
                source=self.name,
                data=output_tensor[0].tolist(),
            )

        if message.message_type == "activation_update":
            # Update network activation without processing
            if "activation" in message.content:
                self.network_activations[message.sender] = float(
                    message.content["activation"]
                )
                self._integrate_activations()

                # Return simple acknowledgment
                return VectorOutput(
                    source=self.name,
                    data=[self.awareness_level] * self.output_dim,
                )

        elif message.message_type == "query":
            # Handle queries about consciousness state
            if "query_type" in message.content:
                query_type = message.content["query_type"]

                if query_type == "awareness_level":
                    return VectorOutput(
                        source=self.name,
                        data=[self.awareness_level] * self.output_dim,
                    )

                if query_type == "self_model" and self.self_awareness > 0.3:
                    # Only respond to self-model queries if sufficient self-awareness
                    dummy_input = torch.zeros(1, self.input_dim)
                    with torch.no_grad():
                        _, self.hidden = self.rnn(dummy_input.unsqueeze(1), self.hidden)
                        self_rep = self.self_model(self.hidden[-1])

                    return VectorOutput(
                        source=self.name,
                        data=self_rep[0].tolist(),
                    )

        return None

    def autonomous_step(self) -> None:
        """Autonomous processing step.

        This function is called periodically by the mind to allow
        the network to perform autonomous processing.
        """
        # Integrate active network states
        self._integrate_activations()

        # Decay hidden state slightly to prevent stagnation
        if self.hidden is not None:
            self.hidden = self.hidden * 0.95

        # Consciousness naturally fluctuates slightly
        fluctuation = (random.random() - 0.5) * 0.05
        self.awareness_level = max(0.1, min(1.0, self.awareness_level + fluctuation))

        # Update state
        self.update_state(
            {
                "awareness_level": self.awareness_level,
                "network_activations": self.network_activations,
            }
        )

        # Send consciousness update to mind
        consciousness_update = NetworkMessage(
            sender=self.name,
            receiver="mind",
            message_type="consciousness",
            content={
                "level": self.awareness_level,
                "attending_to": self.attending_to,
                "self_awareness": self.self_awareness,
            },
            priority=0.7,
        )

        # Add to state for the mind to retrieve
        self.update_state(
            {
                "pending_messages": self.state.parameters.get("pending_messages", [])
                + [consciousness_update.to_dict()],
            }
        )

    def _integrate_activations(self) -> None:
        """Integrate activations from all networks into consciousness."""
        if not self.network_activations:
            return

        # Compute weighted average activation
        total_activation = sum(self.network_activations.values())
        num_networks = len(self.network_activations)

        if num_networks > 0:
            # Base activation level
            base_activation = total_activation / num_networks

            # Scale by integration capacity
            integrated_activation = base_activation * self.integration_capacity

            # Apply developmental limits
            stage_limits = {
                DevelopmentalStage.INFANT: 0.3,
                DevelopmentalStage.TODDLER: 0.5,
                DevelopmentalStage.CHILD: 0.7,
                DevelopmentalStage.ADOLESCENT: 0.9,
                DevelopmentalStage.MATURE: 1.0,
            }

            # Ensure awareness doesn't exceed stage limit
            max_awareness = stage_limits.get(self.developmental_stage, 0.3)
            self.awareness_level = min(max_awareness, integrated_activation)

    def update_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update the developmental stage of the network.

        As the network develops, self-awareness and integration capacity
        increase, simulating the development of consciousness.

        Args:
            stage: New developmental stage

        """
        super().update_developmental_stage(stage)

        # Update consciousness parameters based on developmental stage
        stage_values = {
            DevelopmentalStage.INFANT: {
                "self_awareness": 0.1,
                "integration_capacity": 0.2,
            },
            DevelopmentalStage.TODDLER: {
                "self_awareness": 0.3,
                "integration_capacity": 0.4,
            },
            DevelopmentalStage.CHILD: {
                "self_awareness": 0.5,
                "integration_capacity": 0.6,
            },
            DevelopmentalStage.ADOLESCENT: {
                "self_awareness": 0.7,
                "integration_capacity": 0.8,
            },
            DevelopmentalStage.MATURE: {
                "self_awareness": 0.9,
                "integration_capacity": 1.0,
            },
        }

        if stage in stage_values:
            self.self_awareness = stage_values[stage]["self_awareness"]
            self.integration_capacity = stage_values[stage]["integration_capacity"]

            # Increase hidden state dimensionality for more complex stages
            if (
                stage.value >= DevelopmentalStage.CHILD.value
                and self.hidden is not None
            ):
                # Reset hidden state to allow for growth
                self.hidden = None

            self.update_state(
                {
                    "self_awareness": self.self_awareness,
                    "integration_capacity": self.integration_capacity,
                }
            )

        logger.info(f"Consciousness network updated to {stage.name} stage")

    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network.

        Returns:
            Text representation of the network's current state

        """
        # Generate text based on current state
        awareness_text = (
            "fully aware"
            if self.awareness_level > 0.8
            else "aware"
            if self.awareness_level > 0.5
            else "partially aware"
            if self.awareness_level > 0.2
            else "barely aware"
        )

        attending_to = self.state.parameters.get("attending_to", None)
        attending_text = f" and focusing on {attending_to}" if attending_to else ""

        # Base text
        text = f"Consciousness is {awareness_text}{attending_text}."

        # Add developmental stage-appropriate extensions
        if self.developmental_stage == DevelopmentalStage.INFANT:
            text = f"Basic awareness is emerging. {text}"

        elif self.developmental_stage == DevelopmentalStage.TODDLER:
            text = f"Beginning to form a sense of self. {text}"

        elif self.developmental_stage == DevelopmentalStage.CHILD:
            text = f"Developing a coherent self-model. {text} Self-awareness at {int(self.self_awareness * 100)}%."

        elif self.developmental_stage == DevelopmentalStage.ADOLESCENT:
            text = f"Complex self-awareness established. {text} Integration capacity at {int(self.integration_capacity * 100)}%."

        else:  # MATURE
            text = f"Fully integrated consciousness. {text} Self-model is stable and resilient."

        return TextOutput(
            source=self.name,
            text=text,
            confidence=self.awareness_level,
        )

    def clone_with_growth(
        self, growth_factor: float = 1.2, min_dim: int = 8
    ) -> "ConsciousnessNetwork":
        """Create a larger clone of this network with scaled dimensions.

        Args:
            growth_factor: Factor to scale dimensions by
            min_dim: Minimum dimension size to ensure

        Returns:
            Larger clone of this network with scaled dimensions

        """
        # Calculate new dimensions
        new_input_dim = max(min_dim, int(self.input_dim * growth_factor))
        new_hidden_dim = max(min_dim * 2, int(self.rnn.hidden_size * growth_factor))
        new_output_dim = max(min_dim, int(self.output_dim * growth_factor))

        # Create new network with expanded dimensions
        new_network = ConsciousnessNetwork(
            input_dim=new_input_dim,
            hidden_dim=new_hidden_dim,
            output_dim=new_output_dim,
        )

        # Transfer developmental state
        new_network.developmental_stage = self.developmental_stage
        new_network.awareness_level = self.awareness_level
        new_network.attending_to = self.attending_to
        new_network.self_awareness = self.self_awareness
        new_network.integration_capacity = self.integration_capacity

        # Transfer growth metrics
        new_network.growth_metrics = copy.deepcopy(self.growth_metrics)
        new_network.experience_count = self.experience_count

        # Record growth event
        new_network.growth_history = copy.deepcopy(self.growth_history)
        new_network.growth_history.append(
            NeuralGrowthRecord(
                event_type="network_expansion",
                layer_affected="all",
                old_shape=[self.input_dim, self.rnn.hidden_size, self.output_dim],
                new_shape=[new_input_dim, new_hidden_dim, new_output_dim],
                growth_factor=growth_factor,
                trigger="clone_with_growth",
                developmental_stage=self.developmental_stage,
            )
        )

        # Reset hidden state to accommodate new dimensions
        new_network.hidden = None

        # Log the growth
        logger.info(
            f"ConsciousnessNetwork cloned with growth factor {growth_factor}: "
            f"({self.input_dim}, {self.rnn.hidden_size}, {self.output_dim}) → "
            f"({new_input_dim}, {new_hidden_dim}, {new_output_dim})",
        )

        return new_network
