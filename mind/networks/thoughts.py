"""Thoughts neural network implementation.

This network generates internal thought patterns, beliefs, and conceptual understanding
that evolve in complexity through developmental stages.
"""

import logging
import random
from datetime import datetime
from typing import Any

import torch
from torch import nn

from core.neural_network import NeuralNetwork
from core.schemas import (
    Belief,
    DevelopmentalStage,
    NetworkMessage,
    TextOutput,
    VectorOutput,
)

# Configure logging
logger = logging.getLogger(__name__)


class ThoughtsNetwork(NeuralNetwork):
    """Thoughts network that generates internal mental processes.

    This network handles the generation of thoughts, forming of beliefs,
    and the development of conceptual understanding that increases in
    complexity as the mind develops through different stages.
    """

    def __init__(
        self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64
    ):
        """Initialize the thoughts network.

        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output vectors

        """
        super().__init__(name="thoughts", input_dim=input_dim, output_dim=output_dim)

        # Thought generator network
        self.thought_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Changed from input_dim to hidden_dim
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        # Association network for connecting related concepts
        self.association_network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Belief formation network
        self.belief_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3 outputs: subject, predicate, object
            nn.Softmax(dim=1),
        )

        # RNN for maintaining thought continuity
        self.thought_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Hidden state for thought continuity
        self.hidden_state = None

        # Abstract thinking capacity (develops with stage)
        self.abstract_thinking = 0.1  # Starts minimal, improves with development

        # Logical reasoning capacity (develops with stage)
        self.logical_reasoning = 0.1  # Starts minimal, improves with development

        # Creativity capacity (has unique developmental trajectory)
        self.creativity = 0.3  # Starts moderately high in early stages

        # Current thought stream
        self.current_thoughts = []

        # Beliefs formed by the network
        self.beliefs = []

        # Simple concept network - stores associations between concepts
        self.concept_network = {}

        # Basic vocabulary for forming thoughts
        self.vocabulary = {
            "objects": ["mother", "self", "toy", "food", "light", "sound"],
            "actions": ["want", "see", "hear", "feel", "like", "need"],
            "properties": ["good", "bad", "big", "small", "happy", "sad"],
        }

        # Initialize state parameters
        self.update_state(
            {
                "abstract_thinking": self.abstract_thinking,
                "logical_reasoning": self.logical_reasoning,
                "creativity": self.creativity,
                "thought_count": 0,
                "belief_count": 0,
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: Input tensor representing thought stimulus

        Returns:
            Output tensor representing generated thought

        """
        batch_size = x.size(0)

        # Initialize hidden state if None
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(1, batch_size, 128, device=x.device)

        # Process through RNN for thought continuity
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)  # Add sequence dimension
        else:
            x_seq = x

        rnn_out, self.hidden_state = self.thought_rnn(x_seq, self.hidden_state)

        # Get last output
        last_output = rnn_out[:, -1, :]

        # Generate thought using the thought generator
        thought = self.thought_generator(last_output)

        # Apply developmental effects

        # Abstract thinking affects thought complexity
        if self.abstract_thinking < 0.5:
            # Simplified thoughts at lower developmental stages
            # Add noise to make thoughts more concrete/simple
            noise = torch.randn_like(thought) * (0.5 - self.abstract_thinking)
            thought = torch.clamp(thought + noise, 0, 1)

        # Creativity adds novel patterns to thoughts
        if random.random() < self.creativity:
            # Create a creative "insight" by emphasizing certain dimensions
            creative_mask = torch.zeros_like(thought)
            creative_idx = random.sample(
                range(self.output_dim), int(self.output_dim * 0.2)
            )
            creative_mask[:, creative_idx] = (
                torch.rand(batch_size, len(creative_idx)) * self.creativity
            )
            thought = torch.clamp(thought + creative_mask, 0, 1)

        return thought

    def process_message(self, message: NetworkMessage) -> VectorOutput | None:
        """Process a message from another neural network.

        Args:
            message: Message from another network

        Returns:
            Optional vector output as response

        """
        # Process different message types
        if message.message_type == "perception":
            # Process perception and generate a thought about it
            if "vector_data" in message.content:
                vector_data = message.content["vector_data"]

                # Ensure vector is the right size
                if len(vector_data) > self.input_dim:
                    vector_data = vector_data[: self.input_dim]
                elif len(vector_data) < self.input_dim:
                    vector_data = vector_data + [0.0] * (
                        self.input_dim - len(vector_data)
                    )

                # Convert to tensor and process
                input_tensor = torch.tensor(vector_data, dtype=torch.float32)

                with torch.no_grad():
                    thought_output = self.forward(input_tensor.unsqueeze(0))

                # Generate a thought about the perception
                thought_info = self._generate_thought(
                    thought_output[0],
                    source="perception",
                    source_info=message.content,
                )

                # Remember this thought
                self._remember_thought(thought_info)

                # Return thought vector
                return VectorOutput(
                    source=self.name,
                    data=thought_output[0].tolist(),
                )

        elif message.message_type == "emotion":
            # Generate a thought about an emotion
            if "emotion" in message.content and "intensity" in message.content:
                # Create a simple vector representation of the emotion
                emotion_vector = torch.zeros(self.input_dim)

                # We'll encode the emotion intensity in the first part of the vector
                emotion_intensity = float(message.content["intensity"])
                emotion_vector[0] = emotion_intensity

                # Process the emotion
                with torch.no_grad():
                    thought_output = self.forward(emotion_vector.unsqueeze(0))

                # Generate a thought about the emotion
                thought_info = self._generate_thought(
                    thought_output[0],
                    source="emotion",
                    source_info=message.content,
                )

                # Remember this thought
                self._remember_thought(thought_info)

                # Return thought vector
                return VectorOutput(
                    source=self.name,
                    data=thought_output[0].tolist(),
                )

        elif message.message_type == "pattern":
            # Form a belief based on an observed pattern
            if self.developmental_stage.value >= DevelopmentalStage.TODDLER.value:
                pattern_strength = message.content.get("pattern_strength", 0.5)

                # Only form beliefs from strong patterns
                if pattern_strength > 0.4 and random.random() < self.logical_reasoning:
                    # Generate a random belief for demonstration
                    # In a real implementation, this would be based on pattern content
                    belief = self._form_belief(pattern_strength, message.content)

                    if belief:
                        self.beliefs.append(belief)
                        self.update_state({"belief_count": len(self.beliefs)})

                        # Send belief to mind
                        belief_message = NetworkMessage(
                            sender=self.name,
                            receiver="mind",
                            message_type="belief",
                            content={
                                "subject": belief.subject,
                                "predicate": belief.predicate,
                                "object": belief.object,
                                "confidence": belief.confidence,
                            },
                            priority=0.7,
                        )

                        # Add to state for the mind to retrieve
                        self.update_state(
                            {
                                "pending_messages": self.state.parameters.get(
                                    "pending_messages", []
                                )
                                + [belief_message.to_dict()],
                            }
                        )

                # Generate a thought about the pattern
                dummy_vector = torch.zeros(self.input_dim)
                dummy_vector[0] = pattern_strength

                with torch.no_grad():
                    thought_output = self.forward(dummy_vector.unsqueeze(0))

                # Generate thought info
                thought_info = self._generate_thought(
                    thought_output[0],
                    source="pattern",
                    source_info=message.content,
                )

                # Remember this thought
                self._remember_thought(thought_info)

                return VectorOutput(
                    source=self.name,
                    data=thought_output[0].tolist(),
                )

        elif message.message_type == "query":
            # Respond to queries about thoughts or beliefs
            if "query_type" in message.content:
                query_type = message.content["query_type"]

                if query_type == "current_thought" and self.current_thoughts:
                    # Return vector representation of current thought
                    thought_vector = torch.zeros(self.output_dim)
                    # In a real implementation, this would encode the thought
                    thought_vector[0] = 1.0  # Simple placeholder

                    return VectorOutput(
                        source=self.name,
                        data=thought_vector.tolist(),
                    )

                if query_type == "belief" and "subject" in message.content:
                    # Find beliefs about the subject
                    subject = message.content["subject"]
                    relevant_beliefs = [b for b in self.beliefs if b.subject == subject]

                    if relevant_beliefs:
                        # Return a vector based on most confident belief
                        most_confident = max(
                            relevant_beliefs, key=lambda b: b.confidence
                        )

                        belief_vector = torch.zeros(self.output_dim)
                        # In a real implementation, this would encode the belief
                        belief_vector[0] = most_confident.confidence

                        return VectorOutput(
                            source=self.name,
                            data=belief_vector.tolist(),
                        )

        return None

    def _generate_thought(
        self, output: torch.Tensor, source: str, source_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a thought based on network output.

        Args:
            output: Network output tensor
            source: Source of the thought (perception, emotion, etc.)
            source_info: Information about the source

        Returns:
            Dictionary containing thought information

        """
        # Basic thought info
        thought = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "source_info": source_info,
            "output_summary": output.mean().item(),
        }

        # Generate thought content based on developmental stage
        text = self._generate_thought_text(source, source_info)
        thought["text"] = text

        # Add complexity metrics
        thought["complexity"] = min(1.0, self.abstract_thinking + random.random() * 0.2)
        thought["creativity"] = min(1.0, self.creativity + random.random() * 0.2)

        return thought

    def _generate_thought_text(self, source: str, source_info: dict[str, Any]) -> str:
        """Generate text representation of a thought.

        Args:
            source: Source of the thought
            source_info: Information about the source

        Returns:
            Text representation of the thought

        """
        # Generate text based on developmental stage
        if self.developmental_stage == DevelopmentalStage.INFANT:
            # Infants have simple, object-oriented thoughts
            if source == "perception":
                return random.choice(["see", "want", "like"])
            if source == "emotion":
                emotion = source_info.get("emotion", "feeling")
                if "intensity" in source_info and float(source_info["intensity"]) > 0.7:
                    return f"strong {emotion}"
                return emotion
            return random.choice(self.vocabulary["objects"])

        if self.developmental_stage == DevelopmentalStage.TODDLER:
            # Toddlers have simple subject-verb or subject-verb-object thoughts
            subject = random.choice(self.vocabulary["objects"])
            verb = random.choice(self.vocabulary["actions"])

            if random.random() < 0.5:
                # Subject-verb
                return f"{subject} {verb}"
            # Subject-verb-object
            obj = random.choice(self.vocabulary["objects"])
            while obj == subject:  # Avoid same subject and object
                obj = random.choice(self.vocabulary["objects"])
            return f"{subject} {verb} {obj}"

        if self.developmental_stage == DevelopmentalStage.CHILD:
            # Children have more complex thoughts with adjectives
            subject = random.choice(self.vocabulary["objects"])
            verb = random.choice(self.vocabulary["actions"])
            obj = random.choice(self.vocabulary["objects"])
            prop = random.choice(self.vocabulary["properties"])

            # 50% chance to add property to subject or object
            if random.random() < 0.5:
                return f"{prop} {subject} {verb} {obj}"
            return f"{subject} {verb} {prop} {obj}"

        # More complex thoughts with multiple components
        # In a real implementation, this would use more sophisticated NLG
        if source == "perception":
            desc = source_info.get("description", "something")
            return f"I perceive {desc} and am thinking about what it means"
        if source == "emotion":
            emotion = source_info.get("emotion", "feeling")
            return f"I am experiencing {emotion} and reflecting on this feeling"
        if source == "pattern":
            return "I notice a pattern that seems significant"
        return "I am contemplating various abstract concepts"

    def _remember_thought(self, thought: dict[str, Any]) -> None:
        """Remember a thought.

        Args:
            thought: Thought information to remember

        """
        # Add to current thoughts
        self.current_thoughts.append(thought)

        # Limit memory size based on developmental stage
        max_thoughts = 2 + (self.developmental_stage.value * 2)
        if len(self.current_thoughts) > max_thoughts:
            self.current_thoughts = self.current_thoughts[-max_thoughts:]

        # Update state
        self.update_state(
            {
                "thought_count": self.state.parameters.get("thought_count", 0) + 1,
                "current_thoughts": [t["text"] for t in self.current_thoughts],
            }
        )

    def _form_belief(
        self, confidence: float, source_info: dict[str, Any]
    ) -> Belief | None:
        """Form a belief based on observed patterns or experiences.

        Args:
            confidence: Confidence level in the belief
            source_info: Information about the source

        Returns:
            Formed belief or None

        """
        # Don't form beliefs at infant stage
        if self.developmental_stage == DevelopmentalStage.INFANT:
            return None

        # For demonstration, we'll generate simple beliefs
        # In a real implementation, this would be based on actual patterns and experiences

        # Select random elements from vocabulary
        subject = random.choice(self.vocabulary["objects"])
        predicate = random.choice(self.vocabulary["actions"])
        obj = random.choice(self.vocabulary["objects"])

        # Make sure object is different from subject
        while obj == subject:
            obj = random.choice(self.vocabulary["objects"])

        # Add some randomness to confidence based on logical reasoning ability
        adjusted_confidence = confidence * (0.7 + 0.3 * self.logical_reasoning)

        # Create belief with appropriate developmental stage
        belief = Belief(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=adjusted_confidence,
            developmental_stage=self.developmental_stage,
        )

        return belief

    def autonomous_step(self) -> None:
        """Autonomous processing step.

        This function is called periodically by the mind to allow
        the network to perform autonomous processing.
        """
        # Generate spontaneous thoughts with probability based on developmental stage
        spontaneous_thought_probability = 0.1 + (self.developmental_stage.value * 0.05)

        if random.random() < spontaneous_thought_probability:
            # Generate a random thought vector
            random_vector = torch.rand(self.input_dim)

            with torch.no_grad():
                thought_output = self.forward(random_vector.unsqueeze(0))

            # Generate a spontaneous thought
            thought_info = self._generate_thought(
                thought_output[0],
                source="spontaneous",
                source_info={"trigger": "autonomous_process"},
            )

            # Remember this thought
            self._remember_thought(thought_info)

        # Update beliefs based on logical reasoning
        if (
            self.developmental_stage.value >= DevelopmentalStage.CHILD.value
            and random.random() < self.logical_reasoning
        ):
            # In a real implementation, this would use actual logical inference
            # For now, we'll occasionally update belief confidence
            if self.beliefs:
                belief = random.choice(self.beliefs)
                adjustment = (random.random() - 0.5) * 0.1
                belief.update_confidence(belief.confidence + adjustment)

    def update_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update the developmental stage of the network.

        As the network develops, abstract thinking and logical reasoning improve,
        while creativity follows a unique trajectory (high in early childhood,
        then dips in adolescence before rising again in maturity).

        Args:
            stage: New developmental stage

        """
        super().update_developmental_stage(stage)

        # Update thought parameters based on developmental stage
        stage_values = {
            DevelopmentalStage.INFANT: {
                "abstract_thinking": 0.1,
                "logical_reasoning": 0.1,
                "creativity": 0.4,  # High in infancy
            },
            DevelopmentalStage.TODDLER: {
                "abstract_thinking": 0.3,
                "logical_reasoning": 0.2,
                "creativity": 0.6,  # Peaks in early childhood
            },
            DevelopmentalStage.CHILD: {
                "abstract_thinking": 0.5,
                "logical_reasoning": 0.5,
                "creativity": 0.7,  # Remains high in childhood
            },
            DevelopmentalStage.ADOLESCENT: {
                "abstract_thinking": 0.7,
                "logical_reasoning": 0.7,
                "creativity": 0.5,  # Dips in adolescence
            },
            DevelopmentalStage.MATURE: {
                "abstract_thinking": 0.9,
                "logical_reasoning": 0.9,
                "creativity": 0.8,  # Rises again in maturity
            },
        }

        if stage in stage_values:
            self.abstract_thinking = stage_values[stage]["abstract_thinking"]
            self.logical_reasoning = stage_values[stage]["logical_reasoning"]
            self.creativity = stage_values[stage]["creativity"]

            self.update_state(
                {
                    "abstract_thinking": self.abstract_thinking,
                    "logical_reasoning": self.logical_reasoning,
                    "creativity": self.creativity,
                }
            )

        # Update vocabulary with development
        if stage == DevelopmentalStage.TODDLER:
            # Toddler vocabulary expands
            self.vocabulary["objects"].extend(["water", "bed", "chair", "outside"])
            self.vocabulary["actions"].extend(["go", "sleep", "play", "eat"])
            self.vocabulary["properties"].extend(["hot", "cold", "nice", "scary"])

        elif stage == DevelopmentalStage.CHILD:
            # Child vocabulary expands further
            self.vocabulary["objects"].extend(["friend", "game", "book", "school"])
            self.vocabulary["actions"].extend(["read", "write", "think", "learn"])
            self.vocabulary["properties"].extend(
                ["interesting", "boring", "fun", "difficult"]
            )

        elif stage == DevelopmentalStage.ADOLESCENT:
            # Adolescent vocabulary becomes more abstract
            self.vocabulary["objects"].extend(["future", "past", "idea", "concept"])
            self.vocabulary["actions"].extend(
                ["understand", "question", "analyze", "believe"]
            )
            self.vocabulary["properties"].extend(
                ["complex", "simple", "meaningful", "confusing"]
            )

        # Reset hidden state to accommodate new stage
        self.hidden_state = None

        logger.info(f"Thoughts network updated to {stage.name} stage")

    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network.

        Returns:
            Text representation of the network's current state

        """
        # Generate text based on current thoughts and developmental stage
        if not self.current_thoughts:
            base_text = "No current thoughts."

            if self.developmental_stage == DevelopmentalStage.INFANT:
                text = f"Brain is processing basic sensory input. {base_text}"
            else:
                text = f"Mind is quiet at the moment. {base_text}"

            return TextOutput(
                source=self.name,
                text=text,
                confidence=0.5,
            )

        # Get most recent thought
        latest_thought = self.current_thoughts[-1]
        thought_text = latest_thought.get("text", "thought")

        # Generate appropriate text based on developmental stage
        if self.developmental_stage == DevelopmentalStage.INFANT:
            text = f"Simple thought emerging: {thought_text}"
            confidence = 0.4

        elif self.developmental_stage == DevelopmentalStage.TODDLER:
            text = f"Basic thought: {thought_text}"
            confidence = 0.5

        elif self.developmental_stage == DevelopmentalStage.CHILD:
            text = f"Thinking: {thought_text}"
            if len(self.beliefs) > 0:
                # Add a belief
                belief = random.choice(self.beliefs)
                text += f" | Belief: {belief.to_natural_language()}"
            confidence = 0.6

        elif self.developmental_stage == DevelopmentalStage.ADOLESCENT:
            text = f"Current thought: {thought_text}"
            text += f" | Abstract thinking: {int(self.abstract_thinking * 100)}%, "
            text += f"Logical reasoning: {int(self.logical_reasoning * 100)}%"
            confidence = 0.7

        else:  # MATURE
            text = f"Current thought: {thought_text}"
            if len(self.beliefs) > 0:
                # Add a mature belief
                belief = random.choice(self.beliefs)
                text += f" | Developed belief system includes: {belief.to_natural_language()}"
            confidence = 0.8

        return TextOutput(
            source=self.name,
            text=text,
            confidence=confidence,
        )

    def clone_with_growth(
        self, growth_factor: float = 1.2, min_dim: int = 8
    ) -> "ThoughtsNetwork":
        """Create a larger clone of this network with scaled dimensions.

        Args:
            growth_factor: Factor to scale dimensions by
            min_dim: Minimum dimension size to ensure

        Returns:
            Larger clone of this network with scaled dimensions

        """
        # Calculate new dimensions
        new_input_dim = max(min_dim, int(self.input_dim * growth_factor))
        new_hidden_dim = max(
            min_dim * 2, int(self.thought_rnn.hidden_size * growth_factor)
        )
        new_output_dim = max(min_dim, int(self.output_dim * growth_factor))

        # Create new network with expanded dimensions
        new_network = ThoughtsNetwork(
            input_dim=new_input_dim,
            hidden_dim=new_hidden_dim,
            output_dim=new_output_dim,
        )

        # Transfer thought properties
        new_network.abstract_thinking = self.abstract_thinking
        new_network.logical_reasoning = self.logical_reasoning
        new_network.creativity = self.creativity
        new_network.current_thoughts = copy.deepcopy(self.current_thoughts)
        new_network.beliefs = copy.deepcopy(self.beliefs)
        new_network.concept_network = copy.deepcopy(self.concept_network)
        new_network.vocabulary = copy.deepcopy(self.vocabulary)

        # Transfer growth metrics
        new_network.growth_metrics = copy.deepcopy(self.growth_metrics)
        new_network.experience_count = self.experience_count

        # Record growth event
        new_network.growth_history = copy.deepcopy(self.growth_history)
        new_network.growth_history.append(
            NeuralGrowthRecord(
                event_type="network_expansion",
                layer_affected="all",
                old_shape=[
                    self.input_dim,
                    self.thought_rnn.hidden_size,
                    self.output_dim,
                ],
                new_shape=[new_input_dim, new_hidden_dim, new_output_dim],
                growth_factor=growth_factor,
                trigger="clone_with_growth",
                developmental_stage=self.developmental_stage,
            )
        )

        # Reset hidden state for new dimensions
        new_network.hidden_state = None

        logger.info(
            f"ThoughtsNetwork cloned with growth factor {growth_factor}: "
            f"({self.input_dim}, {self.thought_rnn.hidden_size}, {self.output_dim}) → "
            f"({new_input_dim}, {new_hidden_dim}, {new_output_dim})",
        )

        return new_network
