"""Base neural network class for the mind simulation.

This module provides the foundation for all neural networks in the Neural Child project,
implementing core functionality for network development, communication, and dynamic
growth through experience.
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field, validator
from torch import nn

from core.schemas import (
    DevelopmentalStage,
    NetworkMessage,
    NetworkState,
    TextOutput,
    VectorOutput,
)

# Configure logging
logger = logging.getLogger(__name__)

class GrowthMetrics(BaseModel):
    """Metrics tracking the growth and development of a neural network."""

    connection_density: float = Field(default=0.1, ge=0.0, le=1.0, description="Density of connections in the network")
    plasticity: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to form new connections")
    pruning_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Rate at which unused connections are removed")
    specialization: float = Field(default=0.1, ge=0.0, le=1.0, description="Degree of functional specialization")
    integration: float = Field(default=0.1, ge=0.0, le=1.0, description="Degree of integration with other networks")
    adaptability: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to adapt to new inputs")

    @validator("connection_density", "plasticity", "pruning_rate", "specialization", "integration", "adaptability")
    def check_range(cls, v, values, **kwargs):
        """Ensure values are within range."""
        field_name = kwargs["field"].name
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        return v

    def update_for_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update growth metrics based on developmental stage.
        
        Args:
            stage: Current developmental stage

        """
        # Adjust metrics based on developmental stage
        stage_metrics = {
            DevelopmentalStage.INFANT: {
                "connection_density": 0.1,
                "plasticity": 0.9,
                "pruning_rate": 0.1,
                "specialization": 0.1,
                "integration": 0.1,
                "adaptability": 0.8,
            },
            DevelopmentalStage.TODDLER: {
                "connection_density": 0.3,
                "plasticity": 0.8,
                "pruning_rate": 0.3,
                "specialization": 0.3,
                "integration": 0.2,
                "adaptability": 0.7,
            },
            DevelopmentalStage.CHILD: {
                "connection_density": 0.5,
                "plasticity": 0.6,
                "pruning_rate": 0.4,
                "specialization": 0.5,
                "integration": 0.4,
                "adaptability": 0.6,
            },
            DevelopmentalStage.ADOLESCENT: {
                "connection_density": 0.7,
                "plasticity": 0.4,
                "pruning_rate": 0.3,
                "specialization": 0.7,
                "integration": 0.6,
                "adaptability": 0.5,
            },
            DevelopmentalStage.MATURE: {
                "connection_density": 0.8,
                "plasticity": 0.3,
                "pruning_rate": 0.2,
                "specialization": 0.9,
                "integration": 0.8,
                "adaptability": 0.4,
            },
        }

        # Update metrics
        metrics = stage_metrics.get(stage, {})
        for key, value in metrics.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "connection_density": self.connection_density,
            "plasticity": self.plasticity,
            "pruning_rate": self.pruning_rate,
            "specialization": self.specialization,
            "integration": self.integration,
            "adaptability": self.adaptability,
        }

class NeuralGrowthRecord(BaseModel):
    """Record of neural network growth events."""

    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str = Field(..., description="Type of growth event (expand, prune, etc.)")
    layer_affected: str = Field(..., description="Name or identifier of affected layer")
    old_shape: Optional[List[int]] = Field(default=None, description="Previous shape of layer")
    new_shape: Optional[List[int]] = Field(default=None, description="New shape of layer")
    growth_factor: float = Field(default=1.0, description="Factor by which the layer grew or shrank")
    trigger: str = Field(default="unknown", description="What triggered this growth event")
    developmental_stage: DevelopmentalStage = Field(..., description="Developmental stage when event occurred")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "layer_affected": self.layer_affected,
            "old_shape": self.old_shape,
            "new_shape": self.new_shape,
            "growth_factor": self.growth_factor,
            "trigger": self.trigger,
            "developmental_stage": self.developmental_stage.name,
        }

class NeuralNetwork(nn.Module, ABC):
    """Base class for all neural networks in the mind simulation.
    
    This abstract base class provides core functionality for development-aware neural networks
    that can adapt and grow based on the developmental stage of the mind and experiences.
    """

    def __init__(self, name: str, input_dim: int, output_dim: int):
        """Initialize the neural network.
        
        Args:
            name: Unique identifier for the network
            input_dim: Dimension of input vectors
            output_dim: Dimension of output vectors

        """
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state = NetworkState(
            name=name,
            developmental_weights=dict.fromkeys(DevelopmentalStage, 0.0),
        )
        self.developmental_stage = DevelopmentalStage.INFANT
        self.last_activations = []
        self.learning_rate = 0.01
        self.experience_count = 0

        # Growth and development metrics
        self.growth_metrics = GrowthMetrics()
        self.growth_history: List[NeuralGrowthRecord] = []

        # Activity tracker for hebbian learning
        self.activity_tracker: Dict[str, List[float]] = {}

        # Parameters for dynamic growth
        self.growth_eligible = True
        self.min_experiences_before_growth = 100
        self.experiences_since_last_growth = 0
        self.max_growth_factor = 1.5  # Maximum factor to grow by in one step
        self.min_layer_utilization = 0.3  # Utilization threshold for growth
        self.growth_threshold = 0.7  # Activity threshold to trigger growth
        self.pruning_threshold = 0.1  # Activity threshold for pruning

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor

        """

    @abstractmethod
    def process_message(self, message: NetworkMessage) -> Optional[VectorOutput]:
        """Process a message from another neural network.
        
        Args:
            message: Message from another network
            
        Returns:
            Optional vector output as response

        """

    @abstractmethod
    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network.
        
        Returns:
            Text representation of the network's current state

        """

    def update_state(self, parameters: Dict[str, Any]) -> None:
        """Update the state of the neural network.
        
        Args:
            parameters: Dictionary of parameters to update

        """
        self.state.parameters.update(parameters)
        self.state.last_update = datetime.now()

    def update_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update the developmental stage of the network.
        
        This method adjusts internal weights, parameters, and growth metrics
        based on the new developmental stage.
        
        Args:
            stage: New developmental stage

        """
        self.developmental_stage = stage

        # Update developmental weights
        new_weights = {
            DevelopmentalStage.INFANT: 0.2,
            DevelopmentalStage.TODDLER: 0.4,
            DevelopmentalStage.CHILD: 0.6,
            DevelopmentalStage.ADOLESCENT: 0.8,
            DevelopmentalStage.MATURE: 1.0,
        }

        # Set weight for current stage and above to the corresponding value
        for s in DevelopmentalStage:
            if s.value <= stage.value:
                self.state.developmental_weights[s] = new_weights[s]
            else:
                self.state.developmental_weights[s] = 0.1

        # Update growth metrics for the new stage
        self.growth_metrics.update_for_developmental_stage(stage)

        # Update learning rate based on plasticity
        self.learning_rate = 0.01 * (0.5 + self.growth_metrics.plasticity)

        # Grow network if advancing to a new developmental stage
        if self.growth_eligible and stage.value > 1:  # Beyond INFANT
            self._grow_network_for_new_stage(stage)

        self.state.parameters["developmental_stage"] = stage.value
        self.state.parameters["growth_metrics"] = self.growth_metrics.to_dict()

    def experiential_learning(self, input_data: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """Learn from experience using the input data.
        
        This method implements a form of experiential learning that becomes more
        effective as the network develops, tracking activations for layer growth.
        
        Args:
            input_data: Input tensor to learn from
            target: Optional target tensor for supervised learning
            
        Returns:
            Tuple of (output_tensor, loss_value)

        """
        # Track the experience for growth consideration
        self.experiences_since_last_growth += 1

        # Forward pass
        output = self.forward(input_data)

        # If no target is provided, use a simple self-reinforcement approach
        if target is None:
            # Generate a pseudo-target by slightly enhancing strongest activations
            values, _ = torch.max(output, dim=1, keepdim=True)
            target = torch.where(output > 0.8 * values, output * 1.1, output * 0.9)

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion.forward(output, target)

        # Scale learning by developmental stage and plasticity
        effective_lr = self.learning_rate * self.state.developmental_weights[self.developmental_stage]

        # Backward pass and update weights - only if network is actively learning
        if self.training and effective_lr > 0:
            self.zero_grad()
            loss.backward()

            # Apply modified Hebbian learning - neurons that fire together, wire together
            with torch.no_grad():
                # Track layer activations for growth metrics
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        # Standard gradient update
                        param -= effective_lr * param.grad

                        # Track activations for this layer
                        if name not in self.activity_tracker:
                            self.activity_tracker[name] = []

                        # Use mean absolute value of gradients as a measure of activity
                        activity = param.grad.abs().mean().item()
                        self.activity_tracker[name].append(activity)

                        # Keep only recent history
                        max_history = 50
                        if len(self.activity_tracker[name]) > max_history:
                            self.activity_tracker[name] = self.activity_tracker[name][-max_history:]

        # Track this experience
        self.experience_count += 1
        self.last_activations.append(output.detach().mean().item())
        if len(self.last_activations) > 100:
            self.last_activations = self.last_activations[-100:]

        # Check if ready for potential growth
        if (self.growth_eligible and
            self.experiences_since_last_growth >= self.min_experiences_before_growth):
            self._check_for_network_growth("experiential_learning")

        # Update state with training information
        self.update_state({
            "experience_count": self.experience_count,
            "last_loss": loss.item(),
            "average_activation": np.mean(self.last_activations),
            "growth_metrics": self.growth_metrics.to_dict(),
        })

        return output, loss.item()

    def _check_for_network_growth(self, trigger: str) -> bool:
        """Check if the network should grow based on recent activity.
        
        Args:
            trigger: What triggered this growth check
            
        Returns:
            True if growth occurred, False otherwise

        """
        # Skip if not eligible for growth
        if not self.growth_eligible:
            return False

        # Don't check too frequently
        if self.experiences_since_last_growth < self.min_experiences_before_growth:
            return False

        # Calculate layer utilizations based on activity history
        layer_utilizations = {}
        growth_candidates = []
        prune_candidates = []

        for name, activities in self.activity_tracker.items():
            if not activities:
                continue

            # Calculate recent utilization
            recent_avg = sum(activities) / len(activities)
            layer_utilizations[name] = recent_avg

            # Check if this layer is a candidate for growth or pruning
            if recent_avg > self.growth_threshold:
                growth_candidates.append(name)
            elif recent_avg < self.pruning_threshold:
                prune_candidates.append(name)

        # No candidates for adjustment
        if not growth_candidates and not prune_candidates:
            return False

        growth_occurred = False

        # Apply the growth metrics to decide on actual growth
        if growth_candidates and random.random() < self.growth_metrics.plasticity:
            # Select a random candidate weighted by utilization
            candidates = [(name, layer_utilizations[name]) for name in growth_candidates]
            total_util = sum(util for _, util in candidates)
            r = random.random() * total_util

            cumulative = 0
            selected_layer = candidates[0][0]  # Default to first
            for name, util in candidates:
                cumulative += util
                if r <= cumulative:
                    selected_layer = name
                    break

            # Grow the selected layer
            growth_occurred = self._grow_layer(selected_layer, trigger)

        # Apply pruning if appropriate (but not in the same step as growth)
        if not growth_occurred and prune_candidates and random.random() < self.growth_metrics.pruning_rate:
            # Select a layer for pruning
            selected_layer = random.choice(prune_candidates)

            # Prune the selected layer
            growth_occurred = self._prune_layer(selected_layer, trigger)

        return growth_occurred

    def _grow_layer(self, layer_name: str, trigger: str) -> bool:
        """Grow a specific layer based on activity patterns.
        
        Args:
            layer_name: Name of the layer to grow
            trigger: What triggered this growth
            
        Returns:
            True if growth occurred, False otherwise

        """
        # This is a more complex operation that requires knowing the specific
        # layer types and structures. Default implementation will just log.
        logger.info(f"Growth triggered for layer {layer_name} by {trigger} - Override in subclass")

        # Record the growth event
        self.growth_history.append(NeuralGrowthRecord(
            event_type="grow",
            layer_affected=layer_name,
            growth_factor=1.0,  # No actual growth in base implementation
            trigger=trigger,
            developmental_stage=self.developmental_stage,
        ))

        # Reset growth counter
        self.experiences_since_last_growth = 0

        return False

    def _prune_layer(self, layer_name: str, trigger: str) -> bool:
        """Prune a specific layer based on activity patterns.
        
        Args:
            layer_name: Name of the layer to prune
            trigger: What triggered this pruning
            
        Returns:
            True if pruning occurred, False otherwise

        """
        # Default implementation just logs
        logger.info(f"Pruning triggered for layer {layer_name} by {trigger} - Override in subclass")

        # Record the pruning event
        self.growth_history.append(NeuralGrowthRecord(
            event_type="prune",
            layer_affected=layer_name,
            growth_factor=1.0,  # No actual pruning in base implementation
            trigger=trigger,
            developmental_stage=self.developmental_stage,
        ))

        # Reset growth counter
        self.experiences_since_last_growth = 0

        return False

    def _grow_network_for_new_stage(self, stage: DevelopmentalStage) -> None:
        """Grow the network when advancing to a new developmental stage.
        
        Args:
            stage: New developmental stage

        """
        # Default implementation is a placeholder for subclasses
        logger.info(f"Network {self.name} advancing to stage {stage.name} - Structure changes should be implemented in subclass")

    def get_growth_history(self) -> List[Dict[str, Any]]:
        """Get the history of network growth events.
        
        Returns:
            List of growth events as dictionaries

        """
        return [event.to_dict() for event in self.growth_history]

    def get_developmental_capacity(self) -> float:
        """Get the current developmental capacity of the network.
        
        Returns:
            Float representing developmental capacity (0-1)

        """
        return self.state.developmental_weights[self.developmental_stage]

    def save_model(self, path: str, format: Literal["pytorch", "torchscript", "onnx"] = "pytorch") -> bool:
        """Save the neural network model to disk.
        
        Args:
            path: Path to save the model
            format: Format to save the model in ("pytorch", "torchscript", "onnx")
            
        Returns:
            True if saved successfully, False otherwise

        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            if format == "pytorch":
                return self._save_pytorch(path)
            if format == "torchscript":
                return self._save_torchscript(path)
            if format == "onnx":
                return self._save_onnx(path)
            logger.error(f"Unsupported save format: {format}")
            return False
        except Exception as e:
            logger.error(f"Error saving model: {e!s}")
            return False

    def _save_pytorch(self, path: str) -> bool:
        """Save model in PyTorch format.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if saved successfully

        """
        # Create complete metadata dict
        metadata = {
            "model_state_dict": self.state_dict(),
            "developmental_stage": self.developmental_stage.value,
            "state_parameters": self.state.parameters,
            "learning_rate": self.learning_rate,
            "experience_count": self.experience_count,
            "last_activations": self.last_activations,
            "growth_metrics": self.growth_metrics.to_dict(),
            "growth_history": [event.to_dict() for event in self.growth_history],
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "name": self.name,
            "type": self.__class__.__name__,
            "save_time": datetime.now().isoformat(),
        }

        # Save with optimized settings
        torch.save(metadata, path, _use_new_zipfile_serialization=True)
        logger.info(f"Model saved in PyTorch format to {path}")
        return True

    def _save_torchscript(self, path: str) -> bool:
        """Save model in TorchScript format.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if saved successfully

        """
        try:
            # Create a sample input tensor for tracing
            sample_input = torch.zeros(1, self.input_dim)

            # Create TorchScript model through tracing
            scripted_model = torch.jit.trace(self, sample_input)

            # Save the TorchScript model
            scripted_model.save(path)

            # Save metadata separately
            metadata_path = path + ".metadata.pt"
            torch.save({
                "developmental_stage": self.developmental_stage.value,
                "state_parameters": self.state.parameters,
                "growth_metrics": self.growth_metrics.to_dict(),
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "name": self.name,
                "type": self.__class__.__name__,
                "save_time": datetime.now().isoformat(),
            }, metadata_path)

            logger.info(f"Model saved in TorchScript format to {path}")
            return True
        except Exception as e:
            logger.error(f"Error creating TorchScript model: {e!s}")
            return False

    def _save_onnx(self, path: str) -> bool:
        """Save model in ONNX format.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if saved successfully

        """
        try:
            # Create a sample input tensor
            dummy_input = torch.zeros(1, self.input_dim)

            # Export to ONNX
            torch.onnx.export(
                self,                  # model being run
                dummy_input,           # model input
                path,                  # where to save the model
                export_params=True,    # store the trained parameter weights
                opset_version=11,      # the ONNX version
                do_constant_folding=True,  # optimization
                input_names=["input"],     # the model's input names
                output_names=["output"],   # the model's output names
                dynamic_axes={"input": {0: "batch_size"},    # variable length axes
                              "output": {0: "batch_size"}},
            )

            # Save metadata separately
            metadata_path = path + ".metadata.pt"
            torch.save({
                "developmental_stage": self.developmental_stage.value,
                "state_parameters": self.state.parameters,
                "growth_metrics": self.growth_metrics.to_dict(),
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "name": self.name,
                "type": self.__class__.__name__,
                "save_time": datetime.now().isoformat(),
            }, metadata_path)

            logger.info(f"Model saved in ONNX format to {path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e!s}")
            return False

    def load_model(self, path: str, format: Literal["pytorch", "torchscript", "onnx"] = "pytorch") -> bool:
        """Load the neural network model from disk.
        
        Args:
            path: Path to load the model from
            format: Format the model was saved in
            
        Returns:
            True if loaded successfully, False otherwise

        """
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False

        try:
            if format == "pytorch":
                return self._load_pytorch(path)
            if format == "torchscript":
                return self._load_torchscript(path)
            if format == "onnx":
                return self._load_onnx(path)
            logger.error(f"Unsupported load format: {format}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e!s}")
            return False

    def _load_pytorch(self, path: str) -> bool:
        """Load model from PyTorch format.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if loaded successfully

        """
        try:
            checkpoint = torch.load(path)

            # Check if dimensions match
            if "input_dim" in checkpoint and "output_dim" in checkpoint:
                if checkpoint["input_dim"] != self.input_dim or checkpoint["output_dim"] != self.output_dim:
                    logger.warning(
                        f"Dimension mismatch: saved model has dimensions ({checkpoint['input_dim']}, {checkpoint['output_dim']}), "
                        f"but current model has ({self.input_dim}, {self.output_dim}). Attempting to adapt.",
                    )
                    # Network dimensions changed, might need special handling in subclasses

            # Load model weights
            if "model_state_dict" in checkpoint:
                self.load_state_dict(checkpoint["model_state_dict"])

            # Restore state
            if "developmental_stage" in checkpoint:
                self.developmental_stage = DevelopmentalStage(checkpoint["developmental_stage"])

            if "state_parameters" in checkpoint:
                self.state.parameters = checkpoint["state_parameters"]

            if "learning_rate" in checkpoint:
                self.learning_rate = checkpoint["learning_rate"]

            if "experience_count" in checkpoint:
                self.experience_count = checkpoint["experience_count"]

            if "last_activations" in checkpoint:
                self.last_activations = checkpoint["last_activations"]

            if "growth_metrics" in checkpoint:
                for key, value in checkpoint["growth_metrics"].items():
                    if hasattr(self.growth_metrics, key):
                        setattr(self.growth_metrics, key, value)

            if "growth_history" in checkpoint:
                self._restore_growth_history(checkpoint["growth_history"])

            logger.info(f"Model loaded from PyTorch format: {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model from {path}: {e!s}")
            return False

    def _load_torchscript(self, path: str) -> bool:
        """Load model from TorchScript format.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if loaded successfully

        """
        try:
            # Load TorchScript model
            scripted_model = torch.jit.load(path)

            # This is tricky since we can't directly extract parameters
            # from a TorchScript model into our current instance.
            # We'll need to copy parameters manually or handle this specially

            # For now, load metadata from companion file
            metadata_path = path + ".metadata.pt"
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path)

                # Restore basic attributes from metadata
                if "developmental_stage" in metadata:
                    self.developmental_stage = DevelopmentalStage(metadata["developmental_stage"])

                if "state_parameters" in metadata:
                    self.state.parameters = metadata["state_parameters"]

                if "growth_metrics" in metadata:
                    for key, value in metadata["growth_metrics"].items():
                        if hasattr(self.growth_metrics, key):
                            setattr(self.growth_metrics, key, value)

            logger.info(f"Model metadata loaded from TorchScript format: {path}")
            logger.warning("Note: TorchScript loading only restores metadata, not model parameters.")
            return True

        except Exception as e:
            logger.error(f"Error loading TorchScript model from {path}: {e!s}")
            return False

    def _load_onnx(self, path: str) -> bool:
        """Load model from ONNX format.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if loaded successfully

        """
        # ONNX models typically require ONNX Runtime to execute
        # For PyTorch, we usually load metadata but execute via ONNX Runtime
        try:
            # Load metadata from companion file
            metadata_path = path + ".metadata.pt"
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path)

                # Restore basic attributes from metadata
                if "developmental_stage" in metadata:
                    self.developmental_stage = DevelopmentalStage(metadata["developmental_stage"])

                if "state_parameters" in metadata:
                    self.state.parameters = metadata["state_parameters"]

                if "growth_metrics" in metadata:
                    for key, value in metadata["growth_metrics"].items():
                        if hasattr(self.growth_metrics, key):
                            setattr(self.growth_metrics, key, value)

            logger.info(f"Model metadata loaded from ONNX format: {path}")
            logger.warning("Note: ONNX loading only restores metadata. Use ONNX Runtime for execution.")
            return True

        except Exception as e:
            logger.error(f"Error loading ONNX model from {path}: {e!s}")
            return False

    def _restore_growth_history(self, growth_history_data: List[Dict[str, Any]]) -> None:
        """Restore growth history from serialized data.
        
        Args:
            growth_history_data: List of serialized growth events

        """
        self.growth_history = []
        for event_dict in growth_history_data:
            try:
                # Convert stage name to enum
                stage_name = event_dict.pop("developmental_stage")
                stage = DevelopmentalStage[stage_name]

                # Convert timestamp string to datetime
                timestamp_str = event_dict.pop("timestamp")
                timestamp = datetime.fromisoformat(timestamp_str)

                # Create and add the event
                event = NeuralGrowthRecord(
                    timestamp=timestamp,
                    developmental_stage=stage,
                    **event_dict,
                )
                self.growth_history.append(event)
            except Exception as e:
                logger.warning(f"Error restoring growth event: {e!s}")

    def batch_learning(self, inputs: List[torch.Tensor], targets: Optional[List[torch.Tensor]] = None) -> float:
        """Learn from a batch of experiences.
        
        Args:
            inputs: List of input tensors
            targets: Optional list of target tensors
            
        Returns:
            Average loss

        """
        batch_size = len(inputs)
        if batch_size == 0:
            return 0.0

        # Stack inputs
        input_batch = torch.stack(inputs)

        # Forward pass
        outputs = self.forward(input_batch)

        # If no targets provided, create pseudo-targets
        if targets is None:
            values, _ = torch.max(outputs, dim=1, keepdim=True)
            target_batch = torch.where(outputs > 0.8 * values, outputs * 1.1, outputs * 0.9)
        else:
            target_batch = torch.stack(targets)

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion.forward(outputs, target_batch)

        # Scale learning by developmental stage and plasticity
        effective_lr = self.learning_rate * self.state.developmental_weights[self.developmental_stage]

        # Backward pass and update weights
        if self.training and effective_lr > 0:
            self.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        # Apply gradient update
                        param -= effective_lr * param.grad

                        # Track activations for this layer
                        if name not in self.activity_tracker:
                            self.activity_tracker[name] = []

                        # Use mean absolute value of gradients as activity measure
                        activity = param.grad.abs().mean().item()
                        self.activity_tracker[name].append(activity)

                        # Keep only recent history
                        max_history = 50
                        if len(self.activity_tracker[name]) > max_history:
                            self.activity_tracker[name] = self.activity_tracker[name][-max_history:]

        # Track this batch as an experience for growth
        self.experience_count += 1
        self.experiences_since_last_growth += 1

        # Check for potential growth
        if (self.growth_eligible and
            self.experiences_since_last_growth >= self.min_experiences_before_growth):
            self._check_for_network_growth("batch_learning")

        return loss.item()

    def evaluate(self, test_inputs: List[torch.Tensor], test_targets: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Args:
            test_inputs: List of test input tensors
            test_targets: Optional list of test target tensors
            
        Returns:
            Dictionary of evaluation metrics

        """
        # Store original training state
        was_training = self.training
        self.eval()  # Set to evaluation mode

        results = {}

        try:
            with torch.no_grad():
                # Process in batches if large
                batch_size = 32
                num_batches = (len(test_inputs) + batch_size - 1) // batch_size

                total_loss = 0.0
                all_outputs = []

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(test_inputs))

                    batch_inputs = test_inputs[start_idx:end_idx]
                    input_batch = torch.stack(batch_inputs)

                    # Forward pass
                    outputs = self.forward(input_batch)
                    all_outputs.append(outputs)

                    # Compute loss if targets provided
                    if test_targets is not None:
                        batch_targets = test_targets[start_idx:end_idx]
                        target_batch = torch.stack(batch_targets)

                        criterion = nn.MSELoss()
                        loss = criterion.forward(outputs, target_batch)
                        total_loss += loss.item() * len(batch_inputs)

                # Concatenate all batch outputs
                all_outputs = torch.cat(all_outputs, dim=0)

                if test_targets is not None:
                    results["average_loss"] = total_loss / len(test_inputs)

                # Calculate more sophisticated metrics
                results["average_activation"] = all_outputs.mean().item()
                results["activation_variance"] = all_outputs.var().item()
                results["max_activation"] = all_outputs.max().item()
                results["min_activation"] = all_outputs.min().item()

                # Calculate confidence (using entropy as inverse measure)
                # High entropy = low confidence, low entropy = high confidence
                epsilon = 1e-7  # Small constant to prevent log(0)
                normalized_outputs = all_outputs / (all_outputs.sum(dim=1, keepdim=True) + epsilon)
                entropy = -torch.sum(normalized_outputs * torch.log(normalized_outputs + epsilon), dim=1).mean().item()
                results["confidence"] = 1.0 - min(1.0, entropy)

        finally:
            # Restore original training state
            if was_training:
                self.train()

        return results

    def clone_with_growth(self, growth_factor: float = 1.2, min_dim: int = 8) -> "NeuralNetwork":
        """Create a larger clone of this network with scaled dimensions.
        
        Args:
            growth_factor: Factor to scale dimensions by
            min_dim: Minimum dimension size to ensure
            
        Returns:
            Larger clone of this network

        """
        # Default implementation raises NotImplementedError
        # Subclasses should override this to provide proper cloning with growth
        raise NotImplementedError(
            f"Network {self.name} does not implement clone_with_growth. "
            "This functionality must be implemented in the specific network subclass.",
        )

    def merge_with(self, other_network: "NeuralNetwork", alpha: float = 0.5) -> None:
        """Merge parameters with another network of the same type.
        
        Args:
            other_network: Network to merge with
            alpha: Weighting factor (0 = keep self, 1 = use other)

        """
        if not isinstance(other_network, self.__class__):
            raise TypeError(f"Cannot merge with network of different type: {type(other_network)}")

        # Verify parameter compatibility
        self_params = dict(self.named_parameters())
        other_params = dict(other_network.named_parameters())

        # Check for parameter name matches
        common_params = set(self_params.keys()).intersection(set(other_params.keys()))

        if not common_params:
            raise ValueError("No matching parameters found for merging")

        # Merge parameters
        with torch.no_grad():
            for name in common_params:
                self_param = self_params[name]
                other_param = other_params[name]

                # Check if shapes match
                if self_param.shape != other_param.shape:
                    logger.warning(f"Parameter {name} has mismatched shape: {self_param.shape} vs {other_param.shape}")
                    continue

                # Apply weighted average
                self_param.data.copy_(alpha * other_param.data + (1 - alpha) * self_param.data)

        # Record the merge event
        self.growth_history.append(NeuralGrowthRecord(
            event_type="merge",
            layer_affected="multiple",
            growth_factor=1.0,
            trigger=f"merge_with_{other_network.name}",
            developmental_stage=self.developmental_stage,
        ))

        # Update growth metrics
        self.growth_metrics.integration = min(1.0, self.growth_metrics.integration + 0.1)

        logger.info(f"Merged network {self.name} with {other_network.name} using alpha={alpha}")

    def apply_gaussian_noise(self, std_dev: float = 0.01) -> None:
        """Apply Gaussian noise to parameters to promote exploration.
        
        Args:
            std_dev: Standard deviation of the noise

        """
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * std_dev
                param.add_(noise)

        # Record the noise application
        self.growth_history.append(NeuralGrowthRecord(
            event_type="noise",
            layer_affected="all",
            growth_factor=1.0,
            trigger="exploration",
            developmental_stage=self.developmental_stage,
        ))

    def set_plasticity(self, value: float) -> None:
        """Set the plasticity level of the network.
        
        Args:
            value: New plasticity value (0.0 to 1.0)

        """
        value = max(0.0, min(1.0, value))
        self.growth_metrics.plasticity = value

        # Update learning rate based on new plasticity
        self.learning_rate = 0.01 * (0.5 + value)

        # Update state
        self.update_state({
            "growth_metrics": self.growth_metrics.to_dict(),
            "learning_rate": self.learning_rate,
        })

        logger.info(f"Set plasticity of network {self.name} to {value}")

    def get_complexity_metrics(self) -> Dict[str, Any]:
        """Calculate complexity metrics for the network.
        
        Returns:
            Dictionary of complexity metrics

        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        metrics = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layers": len(list(self.named_modules())),
            "growth_metrics": self.growth_metrics.to_dict(),
            "growth_events": len(self.growth_history),
            "experiences": self.experience_count,
            "developmental_stage": self.developmental_stage.name,
        }

        return metrics
