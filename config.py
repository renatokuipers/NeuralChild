"""Configuration management using Pydantic models.

This module defines configuration structures and provides functionality
for loading/saving configuration from YAML files.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

# Configure logging
logger = logging.getLogger(__name__)

class ServerConfig(BaseModel):
    """Configuration for external servers."""

    llm_server_url: str = "http://localhost:1234/v1/chat/completions"
    embedding_server_url: str = "http://localhost:1234/v1/embeddings"
    obsidian_api_url: Optional[str] = None

class ModelConfig(BaseModel):
    """Configuration for models."""

    llm_model: str = "mistral-7b-instruct-v0.2"
    embedding_model: str = "all-MiniLM-L6-v2"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = -1  # -1 for unlimited

class VisualizationConfig(BaseModel):
    """Configuration for visualization."""

    enabled: bool = True
    update_interval: float = 1.0  # seconds
    obsidian_integration: bool = False
    graph_enabled: bool = True

    # UI color scheme
    colors: Dict[str, str] = Field(default_factory=lambda: {
        "background": "#f0f0f0",
        "text": "#333333",
        "highlight": "#3498db",
        "secondary": "#e74c3c",
        "tertiary": "#2ecc71",
    })

    # Network visualization settings
    network_display: Dict[str, bool] = Field(default_factory=lambda: {
        "consciousness": True,
        "emotions": True,
        "perception": True,
        "thoughts": True,
        "language": True,
    })

class MindConfig(BaseModel):
    """Configuration for the mind simulation."""

    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    step_interval: float = Field(default=0.1, ge=0.01, le=10.0)  # seconds
    need_update_interval: float = Field(default=5.0, ge=0.1)  # seconds
    memory_consolidation_interval: float = Field(default=30.0, ge=1.0)  # seconds
    development_check_interval: float = Field(default=60.0, ge=5.0)  # seconds
    network_growth_check_interval: float = Field(default=120.0, ge=10.0)  # seconds - add this line!

    # Network configuration
    networks: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "consciousness": {"hidden_dim": 128, "input_dim": 64, "output_dim": 64},
        "emotions": {"hidden_dim": 64, "input_dim": 32, "output_dim": 32},
        "perception": {"hidden_dim": 256, "input_dim": 128, "output_dim": 64},
        "thoughts": {"hidden_dim": 128, "input_dim": 64, "output_dim": 64},
        "language": {"hidden_dim": 192, "input_dim": 96, "output_dim": 48},
    })

    # Developmental acceleration factor (1.0 = normal speed, >1.0 = faster)
    development_acceleration: float = Field(default=1.0, ge=0.1, le=10.0)

    # Starting developmental stage (normally INFANT)
    starting_stage: str = "INFANT"

    # Enable/disable specific simulation features
    features_enabled: Dict[str, bool] = Field(default_factory=lambda: {
        "memory_consolidation": True,
        "emotional_development": True,
        "belief_formation": True,
        "language_acquisition": True,
        "need_simulation": True,
    })

class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    file_logging: bool = True
    log_file: str = "neuralchild.log"
    console_logging: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @model_validator(mode="after")
    def validate_log_level(self) -> "LoggingConfig":
        """Validate the log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            logger.warning(f"Invalid log level: {self.level}. Defaulting to INFO.")
            self.level = "INFO"
        return self

class DevelopmentConfig(BaseModel):
    """Configuration for development features."""

    debug_mode: bool = False
    simulate_llm: bool = False  # Use simulated LLM responses instead of real API
    profile_performance: bool = False
    crash_on_error: bool = False
    record_metrics: bool = True

    # Developer settings for experimentation
    experimental_features: Dict[str, bool] = Field(default_factory=dict)

class Config(BaseModel):
    """Main configuration for the NeuralChild project."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    mind: MindConfig = Field(default_factory=MindConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML file
            
        Returns:
            Loaded configuration

        """
        if not os.path.exists(path):
            logger.warning(f"Configuration file not found: {path}, using defaults")
            return cls()

        try:
            with open(path) as f:
                config_dict = yaml.safe_load(f)

            # Handle None case
            if config_dict is None:
                logger.warning(f"Empty configuration file: {path}, using defaults")
                return cls()

            return cls.model_validate(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {e!s}")
            logger.info("Using default configuration")
            return cls()

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.
        
        Args:
            path: Path to save the YAML file

        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            with open(path, "w") as f:
                yaml.dump(self.model_dump(), f, default_flow_style=False)

            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {e!s}")

    def to_json(self, path: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            path: Path to save the JSON file

        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            with open(path, "w") as f:
                json.dump(self.model_dump(), f, indent=2)

            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {e!s}")

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        numeric_level = getattr(logging, self.logging.level, logging.INFO)

        handlers = []

        if self.logging.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(console_handler)

        if self.logging.file_logging:
            try:
                file_handler = logging.FileHandler(self.logging.log_file)
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(logging.Formatter(self.logging.format))
                handlers.append(file_handler)
            except Exception as e:
                print(f"Failed to set up file logging: {e!s}")

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Remove existing handlers
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        # Add new handlers
        for handler in handlers:
            root_logger.addHandler(handler)

        logger.info(f"Logging configured at {self.logging.level} level")

# Global configuration instance
config = Config()

def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Loaded configuration

    """
    global config
    config = Config.from_yaml(path)

    # Setup logging based on configuration
    config.setup_logging()

    return config

def get_config() -> Config:
    """Get the current configuration.
    
    Returns:
        Current configuration

    """
    global config
    return config
