import json
import os
import random
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from pydantic import BaseModel, Field, field_validator, model_validator

# Add project root to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Neural Child project components
# Configure logging
import logging

from config import get_config
from core.schemas import DevelopmentalStage
from mind.mind_core import Mind
from mind.networks.consciousness import ConsciousnessNetwork
from mind.networks.emotions import EmotionsNetwork
from mind.networks.perception import PerceptionNetwork
from mind.networks.thoughts import ThoughtsNetwork
from mother.mother_llm import MotherLLM

logger = logging.getLogger(__name__)

# Define models for input processing with Pydantic
class SimulatedInput(BaseModel):
    """Structured input model for mind simulation."""

    visual: Optional[List[float]] = None
    auditory: Optional[List[float]] = None
    language: Optional[str] = None
    source: str = "environment"
    type: Optional[str] = None

    @field_validator("visual", "auditory", mode="before")
    def ensure_proper_length(cls, v, values, **kwargs):
        """Ensure sensory inputs have proper length."""
        if v is not None:
            if len(v) > 64:
                return v[:64]
            if len(v) < 64:
                return v + [0.0] * (64 - len(v))
        return v

    @model_validator(mode="after")
    def ensure_has_content(cls, values):
        """Ensure at least one input type is provided."""
        # In Pydantic v2, values is the model instance itself in 'after' mode
        if not any([getattr(values, key, None) for key in ["visual", "auditory", "language"]]):
            raise ValueError("At least one input type (visual, auditory, language) must be provided")
        return values

    class Config:
        validate_assignment = True
        extra = "allow"

# Define models for the dashboard configuration
class TrainingConfig(BaseModel):
    """Configuration for training and saving models."""

    save_interval_steps: int = Field(default=100, ge=1, description="Steps between model saves")
    save_directory: str = Field(default="saved_models", description="Directory for saved models")
    checkpoint_count: int = Field(default=5, ge=1, description="Number of checkpoints to keep")
    step_interval: float = Field(default=0.1, ge=0.01, le=10.0, description="Time between simulation steps (seconds)")
    auto_backup: bool = Field(default=True, description="Automatically backup models")

    class Config:
        validate_assignment = True

class DashboardData(BaseModel):
    """Data store for the dashboard."""

    mind_state: Dict[str, Any] = Field(default_factory=dict)
    network_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    development_history: List[Dict[str, Any]] = Field(default_factory=list)
    emotion_history: List[Dict[str, Any]] = Field(default_factory=list)
    memory_history: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = Field(default=0)
    is_running: bool = Field(default=False)
    last_mother_response: Optional[str] = Field(default=None)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    last_saved_step: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)

    class Config:
        validate_assignment = True

# Initialize global state
dashboard_data = DashboardData()
mind = Mind()
mother = MotherLLM()

# Initialize neural networks
networks = {
    "consciousness": ConsciousnessNetwork(),
    "emotions": EmotionsNetwork(),
    "perception": PerceptionNetwork(),
    "thoughts": ThoughtsNetwork(),
}

for name, network in networks.items():
    mind.register_network(network)

# Create a background thread for the simulation
simulation_thread = None
simulation_active = False

def bootstrap_mind(mind: Mind) -> None:
    """Bootstrap the mind with initial experiences to jump-start development.
    
    This function provides foundational experiences to the mind to
    initialize development processes.
    
    Args:
        mind: The Mind instance to bootstrap

    """
    # Create initial experiences for bootstrapping
    experiences = [
        SimulatedInput(
            type="maternal_face",
            visual=[random.random() for _ in range(64)],
            source="mother",
        ),
        SimulatedInput(
            type="maternal_voice",
            auditory=[random.random() for _ in range(64)],
            language="hello my sweet baby",
            source="mother",
        ),
        SimulatedInput(
            type="comfort",
            language="there there, mommy's here",
            source="mother",
        ),
        SimulatedInput(
            type="toy_perception",
            visual=[random.random() for _ in range(64)],
            source="environment",
        ),
        SimulatedInput(
            type="environmental_sound",
            auditory=[random.random() for _ in range(64)],
            source="environment",
        ),
    ]

    # Process each bootstrap experience
    for exp in experiences:
        mind.process_input(exp.model_dump(exclude_none=True))

    logger.info("Mind bootstrapped with initial experiences")

def generate_environmental_input() -> Dict[str, Any]:
    """Generate environmental sensory inputs for the mind.
    
    Returns:
        Dictionary of sensory input data

    """
    input_type = random.choice(["visual", "auditory", "language", "combined"])

    if input_type == "visual":
        simulated_input = SimulatedInput(
            visual=[random.random() for _ in range(64)],
            type="visual_stimulus",
            source="visual_environment",
        )
    elif input_type == "auditory":
        simulated_input = SimulatedInput(
            auditory=[random.random() for _ in range(64)],
            type="auditory_stimulus",
            source="auditory_environment",
        )
    elif input_type == "language":
        words = ["mama", "dada", "baby", "milk", "toy", "hug", "play", "sleep"]
        language_input = " ".join(random.sample(words, k=random.randint(1, 2)))
        simulated_input = SimulatedInput(
            language=language_input,
            type="language_stimulus",
            source="environment_sounds",
        )
    else:  # combined
        simulated_input = SimulatedInput(
            visual=[random.random() for _ in range(64)],
            auditory=[random.random() for _ in range(64)],
            type="multimodal_stimulus",
            source="rich_environment",
        )

    return simulated_input.model_dump(exclude_none=True)

def process_mother_response(response_text: str) -> Dict[str, Any]:
    """Process mother's response as sensory input.
    
    Args:
        response_text: Text of mother's response
        
    Returns:
        Dictionary of sensory input data

    """
    # Create auditory component based on text length
    auditory_intensity = min(1.0, len(response_text) / 100)
    auditory_vector = [random.random() * auditory_intensity for _ in range(64)]

    mother_input = SimulatedInput(
        language=response_text,
        auditory=auditory_vector,
        type="maternal_communication",
        source="mother",
    )

    return mother_input.model_dump(exclude_none=True)

def run_simulation():
    """Run the simulation in the background."""
    global simulation_active, dashboard_data

    simulation_active = True
    dashboard_data.is_running = True
    step_count = dashboard_data.step_count

    # Bootstrap the mind at the start
    if step_count == 0:
        bootstrap_mind(mind)

    while simulation_active:
        try:
            # Run one simulation step
            mind.step()
            dashboard_data.step_count += 1
            step_count += 1

            # Generate simulated input periodically (every 3 steps)
            if step_count % 3 == 0:
                env_input = generate_environmental_input()
                mind.process_input(env_input)

            # Generate mother response periodically
            if step_count % 10 == 0:
                response = mother.observe_and_respond(mind)
                if response:
                    dashboard_data.last_mother_response = response.response

                    # Process mother's response as sensory input
                    mother_input = process_mother_response(response.response)
                    mind.process_input(mother_input)

            # Get observable state and update dashboard data
            observable_state = mind.get_observable_state()
            mind_state = mind.get_state()

            # Update dashboard data with current state
            dashboard_data.mind_state = {
                "developmental_stage": observable_state.developmental_stage.name,
                "energy_level": observable_state.energy_level,
                "apparent_mood": observable_state.apparent_mood,
                "vocalization": observable_state.vocalization or "None",
                "current_focus": observable_state.current_focus or "None",
                "consciousness_level": mind_state.consciousness_level,
                "timestamp": datetime.now().isoformat(),
            }

            # Update network states
            for name, network in networks.items():
                text_output = network.generate_text_output()
                dashboard_data.network_states[name] = {
                    "text_output": text_output.text,
                    "confidence": text_output.confidence,
                    "parameters": network.state.parameters,
                }

            # Track development history
            dashboard_data.development_history.append({
                "step": dashboard_data.step_count,
                "timestamp": datetime.now().isoformat(),
                "developmental_stage": observable_state.developmental_stage.name,
                "energy_level": observable_state.energy_level,
                "consciousness_level": mind_state.consciousness_level,
            })

            # Limit history size to prevent memory issues
            if len(dashboard_data.development_history) > 1000:
                dashboard_data.development_history = dashboard_data.development_history[-1000:]

            # Track emotion history
            if observable_state.recent_emotions:
                for emotion in observable_state.recent_emotions:
                    dashboard_data.emotion_history.append({
                        "step": dashboard_data.step_count,
                        "timestamp": datetime.now().isoformat(),
                        "emotion": emotion.name.value,
                        "intensity": emotion.intensity,
                    })

                # Limit history size
                if len(dashboard_data.emotion_history) > 1000:
                    dashboard_data.emotion_history = dashboard_data.emotion_history[-1000:]

            # Track memory counts
            dashboard_data.memory_history.append({
                "step": dashboard_data.step_count,
                "timestamp": datetime.now().isoformat(),
                "short_term": len(mind.short_term_memory),
                "long_term": len(mind.long_term_memory),
            })

            # Limit history size
            if len(dashboard_data.memory_history) > 1000:
                dashboard_data.memory_history = dashboard_data.memory_history[-1000:]

            # Save models at specified intervals
            if (dashboard_data.training_config.auto_backup and
                dashboard_data.step_count % dashboard_data.training_config.save_interval_steps == 0 and
                dashboard_data.step_count > dashboard_data.last_saved_step):

                save_models()
                dashboard_data.last_saved_step = dashboard_data.step_count

            # Sleep for the configured step interval
            time.sleep(dashboard_data.training_config.step_interval)

        except Exception as e:
            # Log error
            error_msg = f"Error in simulation step {dashboard_data.step_count}: {e!s}"
            dashboard_data.errors.append(error_msg)
            if len(dashboard_data.errors) > 50:
                dashboard_data.errors = dashboard_data.errors[-50:]
            time.sleep(1)  # Wait before retrying

def save_models(checkpoint_name: Optional[str] = None):
    """Save all models to disk."""
    try:
        # Create the save directory if it doesn't exist
        save_dir = dashboard_data.training_config.save_directory
        os.makedirs(save_dir, exist_ok=True)

        # Create checkpoint subdirectory
        if checkpoint_name:
            checkpoint_dir = os.path.join(save_dir, checkpoint_name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(save_dir, f"checkpoint_{timestamp}_step_{dashboard_data.step_count}")

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save mind state
        mind.save_state(checkpoint_dir)

        # Save dashboard data
        dashboard_file = os.path.join(checkpoint_dir, "dashboard_data.json")
        with open(dashboard_file, "w") as f:
            # Convert to dict and filter out some fields
            data_dict = dashboard_data.model_dump()
            # Remove large history data to keep save file manageable
            data_dict["development_history"] = data_dict["development_history"][-100:]
            data_dict["emotion_history"] = data_dict["emotion_history"][-100:]
            data_dict["memory_history"] = data_dict["memory_history"][-100:]
            json.dump(data_dict, f, indent=2, cls=DateTimeEncoder)

        # Clean up old checkpoints if needed
        checkpoints = [d for d in os.listdir(save_dir)
                      if os.path.isdir(os.path.join(save_dir, d)) and d.startswith("checkpoint_")]

        if len(checkpoints) > dashboard_data.training_config.checkpoint_count:
            # Sort by creation time and remove oldest
            checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(save_dir, x)))
            for old_checkpoint in checkpoints[:-dashboard_data.training_config.checkpoint_count]:
                old_dir = os.path.join(save_dir, old_checkpoint)
                try:
                    import shutil
                    shutil.rmtree(old_dir)
                except Exception as e:
                    dashboard_data.errors.append(f"Failed to remove old checkpoint {old_dir}: {e!s}")

        return True, checkpoint_dir
    except Exception as e:
        error_msg = f"Error saving models: {e!s}"
        dashboard_data.errors.append(error_msg)
        return False, error_msg

def load_models(checkpoint_dir: str):
    """Load models from disk."""
    try:
        # Load mind state
        success = mind.load_state(checkpoint_dir)
        if not success:
            return False, f"Failed to load mind state from {checkpoint_dir}"

        # Load dashboard data if available
        dashboard_file = os.path.join(checkpoint_dir, "dashboard_data.json")
        if os.path.exists(dashboard_file):
            with open(dashboard_file) as f:
                data_dict = json.load(f)

                # Process any datetime fields in history data
                for history_name in ["development_history", "emotion_history", "memory_history"]:
                    if history_name in data_dict:
                        for entry in data_dict[history_name]:
                            # Convert timestamp strings back to datetime objects if needed
                            if "timestamp" in entry and isinstance(entry["timestamp"], str):
                                try:
                                    entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                                except (ValueError, TypeError):
                                    # If conversion fails, leave as string
                                    pass

                # Update dashboard data, preserving training config
                training_config = dashboard_data.training_config
                dashboard_data.__dict__.update(data_dict)
                dashboard_data.training_config = training_config

        return True, f"Successfully loaded models from {checkpoint_dir}"
    except Exception as e:
        error_msg = f"Error loading models: {e!s}"
        dashboard_data.errors.append(error_msg)
        return False, error_msg

# Add a custom JSONEncoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Dash App Setup
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

# Custom CSS for gradient backgrounds and rounded corners
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Neural Child Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #6e45e2 0%, #89d4cf 100%);
                --secondary-gradient: linear-gradient(135deg, #7303c0 0%, #3b8dff 100%);
                --dark-bg: #121212;
                --card-bg: #1E1E1E;
                --text-color: #E0E0E0;
                --highlight-color: #9d68f2;
                --border-radius: 10px;
            }
            
            body {
                background-color: var(--dark-bg);
                color: var(--text-color);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .card {
                background-color: var(--card-bg);
                border-radius: var(--border-radius);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                overflow: hidden;
            }
            
            .card-header {
                background: var(--primary-gradient);
                color: white;
                font-weight: 600;
                padding: 12px 15px;
                border-top-left-radius: var(--border-radius);
                border-top-right-radius: var(--border-radius);
            }
            
            .card-body {
                padding: 15px;
            }
            
            .network-card {
                height: 100%;
                display: flex;
                flex-direction: column;
            }
            
            .network-card .card-body {
                flex: 1;
                overflow-y: auto;
            }
            
            .control-button {
                background: var(--secondary-gradient);
                border: none;
                border-radius: var(--border-radius);
                color: white;
                padding: 8px 15px;
                transition: all 0.3s ease;
            }
            
            .control-button:hover {
                opacity: 0.9;
                transform: translateY(-2px);
            }
            
            .status-indicator {
                height: 12px;
                width: 12px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 5px;
            }
            
            .status-active {
                background-color: #4CAF50;
            }
            
            .status-inactive {
                background-color: #F44336;
            }
            
            .stage-indicator {
                font-size: 1.2rem;
                font-weight: bold;
                background: var(--secondary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .custom-tab {
                border-radius: var(--border-radius) var(--border-radius) 0 0;
                padding: 10px 15px;
                font-weight: 500;
            }
            
            .custom-tab--selected {
                background: var(--secondary-gradient);
                color: white;
            }
            
            /* Custom slider styling */
            .rc-slider-track {
                background: var(--primary-gradient);
            }
            
            .rc-slider-handle {
                border: solid 2px var(--highlight-color);
            }
            
            /* Dropdown styling */
            .Select-control {
                background-color: var(--card-bg);
                border-color: rgba(255, 255, 255, 0.1);
                color: var(--text-color);
            }
            
            .Select-menu-outer {
                background-color: var(--card-bg);
                border-color: rgba(255, 255, 255, 0.1);
            }
            
            .Select-option {
                background-color: var(--card-bg);
                color: var(--text-color);
            }
            
            .Select-option.is-selected {
                background-color: var(--highlight-color);
            }
            
            /* Input styling */
            .form-control {
                background-color: var(--card-bg);
                border-color: rgba(255, 255, 255, 0.1);
                color: var(--text-color);
            }
            
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--card-bg);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--highlight-color);
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Define the app layout
app.layout = dbc.Container(
    fluid=True,
    style={"padding": "20px"},
    children=[
        html.Div(id="interval-container", children=[
            dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
        ]),

        # Header Row
        dbc.Row([
            dbc.Col([
                html.H1("Neural Child Dashboard",
                        style={"backgroundImage": "linear-gradient(90deg, #7303c0, #3b8dff)",
                               "WebkitBackgroundClip": "text",
                               "WebkitTextFillColor": "transparent",
                               "fontWeight": "bold",
                               "marginBottom": "20px"}),
            ], width=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("Status: ", style={"marginRight": "5px"}),
                            html.Span(id="status-indicator", className="status-indicator status-inactive"),
                            html.Span(id="status-text", children="Inactive"),
                        ], style={"marginBottom": "10px"}),
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.Span("Step Count: "),
                            html.Span(id="step-count", children="0"),
                        ], style={"marginBottom": "10px"}),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Button("Start Simulation", id="start-button", className="control-button",
                                        style={"marginRight": "10px", "backgroundColor": "#4CAF50"}),
                            html.Button("Stop Simulation", id="stop-button", className="control-button",
                                        style={"marginRight": "10px", "backgroundColor": "#F44336"}),
                            html.Button("Save Models", id="save-button", className="control-button",
                                        style={"backgroundColor": "#3b8dff"}),
                        ]),
                    ], width=12),
                ]),
            ], width=6),
        ]),

        # Main Dashboard Row
        dbc.Row([
            # Left Column - Mind State and Controls
            dbc.Col([
                # Mind State Card
                html.Div(className="card", children=[
                    html.Div(className="card-header", children="Mind State Overview"),
                    html.Div(className="card-body", children=[
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5("Developmental Stage", className="mb-2"),
                                    html.Div(id="developmental-stage", className="stage-indicator", children="INFANT"),
                                ], style={"marginBottom": "15px"}),
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    html.H5("Consciousness Level", className="mb-2"),
                                    dbc.Progress(id="consciousness-progress", value=20,
                                                 style={"height": "10px", "borderRadius": "5px"}),
                                ], style={"marginBottom": "15px"}),
                            ], width=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5("Energy Level", className="mb-2"),
                                    dbc.Progress(id="energy-progress", value=70, color="success",
                                                 style={"height": "10px", "borderRadius": "5px"}),
                                ], style={"marginBottom": "15px"}),
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    html.H5("Mood", className="mb-2"),
                                    dbc.Progress(id="mood-progress", value=50, color="info",
                                                 style={"height": "10px", "borderRadius": "5px"}),
                                ], style={"marginBottom": "15px"}),
                            ], width=6),
                        ]),
                        html.Div([
                            html.H5("Current Focus", className="mb-2"),
                            html.Div(id="current-focus", children="None"),
                        ], style={"marginBottom": "15px"}),
                        html.Div([
                            html.H5("Latest Vocalization", className="mb-2"),
                            html.Div(id="vocalization", children="None"),
                        ], style={"marginBottom": "15px"}),
                    ]),
                ]),

                # Mother Interaction Card
                html.Div(className="card", children=[
                    html.Div(className="card-header", children="Mother's Response"),
                    html.Div(className="card-body", children=[
                        html.Div(id="mother-response",
                                 style={"minHeight": "100px", "fontStyle": "italic"},
                                 children="No recent responses."),
                    ]),
                ]),

                # Training Configuration Card
                html.Div(className="card", children=[
                    html.Div(className="card-header", children="Training Configuration"),
                    html.Div(className="card-body", children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("Step Interval (seconds)"),
                                dcc.Slider(
                                    id="step-interval-slider",
                                    min=0.01,
                                    max=1,
                                    step=0.01,
                                    value=0.1,
                                    marks={0.01: "0.01", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1"},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ], width=12),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Save Interval (steps)"),
                                dbc.Input(
                                    id="save-interval-input",
                                    type="number",
                                    value=100,
                                    min=1,
                                    step=1,
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Checkpoints to Keep"),
                                dbc.Input(
                                    id="checkpoint-count-input",
                                    type="number",
                                    value=5,
                                    min=1,
                                    step=1,
                                ),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Save Directory"),
                                dbc.Input(
                                    id="save-directory-input",
                                    type="text",
                                    value="saved_models",
                                ),
                            ], width=8),
                            dbc.Col([
                                dbc.Checkbox(
                                    id="auto-backup-checkbox",
                                    label="Auto Backup",
                                    value=True,
                                ),
                            ], width=4, style={"display": "flex", "alignItems": "flex-end"}),
                        ]),
                        html.Div(
                            html.Button(
                                "Apply Configuration",
                                id="apply-config-button",
                                className="control-button",
                                style={"marginTop": "15px"},
                            ),
                            style={"textAlign": "right"},
                        ),
                    ]),
                ]),

                # Error Log Card
                html.Div(className="card", children=[
                    html.Div(className="card-header", children="System Log"),
                    html.Div(className="card-body", children=[
                        html.Div(
                            id="error-log",
                            style={
                                "maxHeight": "150px",
                                "overflowY": "auto",
                                "fontFamily": "monospace",
                                "fontSize": "0.8rem",
                                "whiteSpace": "pre-wrap",
                            },
                            children="System ready.",
                        ),
                    ]),
                ]),
            ], width=4),

            # Right Column - Networks and Visualization
            dbc.Col([
                # Tabs for different views
                dcc.Tabs(id="main-tabs", value="networks-tab", className="custom-tabs", children=[
                    # Networks Tab
                    dcc.Tab(
                        label="Neural Networks",
                        value="networks-tab",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=[
                            dbc.Row([
                                # Consciousness Network
                                dbc.Col([
                                    html.Div(className="card network-card", children=[
                                        html.Div(className="card-header", children="Consciousness Network"),
                                        html.Div(className="card-body", children=[
                                            html.Div(id="consciousness-output",
                                                    style={"marginBottom": "10px"}),
                                            html.Hr(),
                                            html.Div([
                                                html.Span("Confidence: "),
                                                dbc.Progress(id="consciousness-confidence", value=0,
                                                            style={"height": "6px", "borderRadius": "3px"}),
                                            ]),
                                        ]),
                                    ]),
                                ], width=6),

                                # Emotions Network
                                dbc.Col([
                                    html.Div(className="card network-card", children=[
                                        html.Div(className="card-header", children="Emotions Network"),
                                        html.Div(className="card-body", children=[
                                            html.Div(id="emotions-output",
                                                    style={"marginBottom": "10px"}),
                                            html.Hr(),
                                            html.Div([
                                                html.Span("Confidence: "),
                                                dbc.Progress(id="emotions-confidence", value=0,
                                                            style={"height": "6px", "borderRadius": "3px"}),
                                            ]),
                                        ]),
                                    ]),
                                ], width=6),
                            ], style={"marginBottom": "15px"}),

                            dbc.Row([
                                # Perception Network
                                dbc.Col([
                                    html.Div(className="card network-card", children=[
                                        html.Div(className="card-header", children="Perception Network"),
                                        html.Div(className="card-body", children=[
                                            html.Div(id="perception-output",
                                                    style={"marginBottom": "10px"}),
                                            html.Hr(),
                                            html.Div([
                                                html.Span("Confidence: "),
                                                dbc.Progress(id="perception-confidence", value=0,
                                                            style={"height": "6px", "borderRadius": "3px"}),
                                            ]),
                                        ]),
                                    ]),
                                ], width=6),

                                # Thoughts Network
                                dbc.Col([
                                    html.Div(className="card network-card", children=[
                                        html.Div(className="card-header", children="Thoughts Network"),
                                        html.Div(className="card-body", children=[
                                            html.Div(id="thoughts-output",
                                                    style={"marginBottom": "10px"}),
                                            html.Hr(),
                                            html.Div([
                                                html.Span("Confidence: "),
                                                dbc.Progress(id="thoughts-confidence", value=0,
                                                            style={"height": "6px", "borderRadius": "3px"}),
                                            ]),
                                        ]),
                                    ]),
                                ], width=6),
                            ]),
                        ],
                    ),

                    # Analytics Tab
                    dcc.Tab(
                        label="Analytics",
                        value="analytics-tab",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=[
                            # Development Graph
                            html.Div(className="card", style={"marginTop": "15px"}, children=[
                                html.Div(className="card-header", children="Development Over Time"),
                                html.Div(className="card-body", children=[
                                    dcc.Graph(
                                        id="development-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "250px"},
                                    ),
                                ]),
                            ]),

                            # Emotions Graph
                            html.Div(className="card", style={"marginTop": "15px"}, children=[
                                html.Div(className="card-header", children="Emotional State Evolution"),
                                html.Div(className="card-body", children=[
                                    dcc.Graph(
                                        id="emotions-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "250px"},
                                    ),
                                ]),
                            ]),

                            # Memory Graph
                            html.Div(className="card", style={"marginTop": "15px"}, children=[
                                html.Div(className="card-header", children="Memory Development"),
                                html.Div(className="card-body", children=[
                                    dcc.Graph(
                                        id="memory-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "250px"},
                                    ),
                                ]),
                            ]),
                        ],
                    ),

                    # Development Tab
                    dcc.Tab(
                        label="Development",
                        value="development-tab",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=[
                            # Developmental Milestones
                            html.Div(className="card", style={"marginTop": "15px"}, children=[
                                html.Div(className="card-header", children="Developmental Milestones"),
                                html.Div(className="card-body", children=[
                                    html.Div(id="milestones-content"),
                                ]),
                            ]),

                            # Beliefs Table
                            html.Div(className="card", style={"marginTop": "15px"}, children=[
                                html.Div(className="card-header", children="Beliefs System"),
                                html.Div(className="card-body", children=[
                                    html.Div(id="beliefs-content"),
                                ]),
                            ]),

                            # Needs Status
                            html.Div(className="card", style={"marginTop": "15px"}, children=[
                                html.Div(className="card-header", children="Needs Status"),
                                html.Div(className="card-body", children=[
                                    html.Div(id="needs-content"),
                                ]),
                            ]),
                        ],
                    ),
                ]),
            ], width=8),
        ]),

        # Hidden div for storing intermediate data
        html.Div(id="intermediate-value", style={"display": "none"}),
    ],
)

# Callback to update the intermediate value with current state
@app.callback(
    Output("intermediate-value", "children"),
    Input("update-interval", "n_intervals"),
)
def update_intermediate_value(n_intervals):
    return json.dumps({
        "is_running": dashboard_data.is_running,
        "step_count": dashboard_data.step_count,
        "mind_state": dashboard_data.mind_state,
        "network_states": dashboard_data.network_states,
        "last_mother_response": dashboard_data.last_mother_response,
        "errors": dashboard_data.errors,
    })

# Callback to update the status indicators
@app.callback(
    [Output("status-indicator", "className"),
     Output("status-text", "children"),
     Output("step-count", "children")],
    Input("intermediate-value", "children"),
)
def update_status_indicators(json_data):
    data = json.loads(json_data) if json_data else {}

    is_running = data.get("is_running", False)
    step_count = data.get("step_count", 0)

    status_class = "status-indicator status-active" if is_running else "status-indicator status-inactive"
    status_text = "Running" if is_running else "Inactive"

    return status_class, status_text, step_count

# Callback to update the mind state overviews
@app.callback(
    [Output("developmental-stage", "children"),
     Output("consciousness-progress", "value"),
     Output("energy-progress", "value"),
     Output("mood-progress", "value"),
     Output("current-focus", "children"),
     Output("vocalization", "children"),
     Output("mother-response", "children")],
    Input("intermediate-value", "children"),
)
def update_mind_state(json_data):
    data = json.loads(json_data) if json_data else {}

    mind_state = data.get("mind_state", {})

    # Extract values with defaults
    developmental_stage = mind_state.get("developmental_stage", "INFANT")
    consciousness_level = int(mind_state.get("consciousness_level", 0.2) * 100)
    energy_level = int(mind_state.get("energy_level", 0.7) * 100)
    apparent_mood = int((mind_state.get("apparent_mood", 0) + 1) * 50)  # Convert -1 to 1 range to 0-100
    current_focus = mind_state.get("current_focus", "None")
    vocalization = mind_state.get("vocalization", "None")

    mother_response = data.get("last_mother_response", "No recent responses.")

    return (
        developmental_stage,
        consciousness_level,
        energy_level,
        apparent_mood,
        current_focus,
        vocalization,
        mother_response or "No recent responses.",
    )

# Callback to update the neural network outputs
@app.callback(
    [Output("consciousness-output", "children"),
     Output("consciousness-confidence", "value"),
     Output("emotions-output", "children"),
     Output("emotions-confidence", "value"),
     Output("perception-output", "children"),
     Output("perception-confidence", "value"),
     Output("thoughts-output", "children"),
     Output("thoughts-confidence", "value")],
    Input("intermediate-value", "children"),
)
def update_network_outputs(json_data):
    data = json.loads(json_data) if json_data else {}

    network_states = data.get("network_states", {})

    # Get states with defaults
    consciousness = network_states.get("consciousness", {})
    emotions = network_states.get("emotions", {})
    perception = network_states.get("perception", {})
    thoughts = network_states.get("thoughts", {})

    # Extract values
    consciousness_output = consciousness.get("text_output", "No data")
    consciousness_confidence = int(consciousness.get("confidence", 0) * 100)

    emotions_output = emotions.get("text_output", "No data")
    emotions_confidence = int(emotions.get("confidence", 0) * 100)

    perception_output = perception.get("text_output", "No data")
    perception_confidence = int(perception.get("confidence", 0) * 100)

    thoughts_output = thoughts.get("text_output", "No data")
    thoughts_confidence = int(thoughts.get("confidence", 0) * 100)

    return (
        consciousness_output,
        consciousness_confidence,
        emotions_output,
        emotions_confidence,
        perception_output,
        perception_confidence,
        thoughts_output,
        thoughts_confidence,
    )

# Callback to update the development graph
@app.callback(
    Output("development-graph", "figure"),
    Input("update-interval", "n_intervals"),
)
def update_development_graph(n_intervals):
    history = dashboard_data.development_history

    if not history:
        # Return empty figure if no data
        return go.Figure().update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Steps",
            yaxis_title="Level",
            title="No development data available",
        )

    # Prepare data
    df = pd.DataFrame(history[-200:])  # Just use last 200 points

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["consciousness_level"],
        mode="lines",
        name="Consciousness",
        line=dict(width=2, color="#9d68f2"),
    ))

    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["energy_level"],
        mode="lines",
        name="Energy",
        line=dict(width=2, color="#4CAF50"),
    ))

    # Highlight developmental stage changes
    stage_changes = []
    prev_stage = None
    for i, row in df.iterrows():
        if row["developmental_stage"] != prev_stage:
            stage_changes.append({
                "step": row["step"],
                "stage": row["developmental_stage"],
            })
            prev_stage = row["developmental_stage"]

    for change in stage_changes:
        fig.add_vline(
            x=change["step"],
            line_width=2,
            line_dash="dash",
            line_color="#3b8dff",
            annotation_text=change["stage"],
            annotation_position="top right",
        )

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Steps",
        yaxis_title="Level",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig

# Callback to update the emotions graph
@app.callback(
    Output("emotions-graph", "figure"),
    Input("update-interval", "n_intervals"),
)
def update_emotions_graph(n_intervals):
    history = dashboard_data.emotion_history

    if not history:
        # Return empty figure if no data
        return go.Figure().update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Steps",
            yaxis_title="Intensity",
            title="No emotion data available",
        )

    # Prepare data
    df = pd.DataFrame(history[-200:])  # Just use last 200 points

    # Group by step and emotion, taking the max intensity for each
    df = df.groupby(["step", "emotion"])["intensity"].max().reset_index()

    # Pivot to get emotions as columns
    df_pivot = df.pivot(index="step", columns="emotion", values="intensity").reset_index()
    df_pivot = df_pivot.fillna(0)  # Fill NaN with 0

    # Create figure
    fig = go.Figure()

    # Add a trace for each emotion
    colors = {
        "joy": "#FFD700",         # Gold
        "sadness": "#4682B4",     # Steel Blue
        "anger": "#DC143C",       # Crimson
        "fear": "#800080",        # Purple
        "disgust": "#006400",     # Dark Green
        "surprise": "#FF8C00",    # Dark Orange
        "trust": "#20B2AA",       # Light Sea Green
        "anticipation": "#FF69B4",# Hot Pink
        "confusion": "#708090",   # Slate Gray
        "interest": "#7B68EE",    # Medium Slate Blue
        "boredom": "#A9A9A9",     # Dark Gray
    }

    for emotion in df_pivot.columns:
        if emotion != "step":
            fig.add_trace(go.Scatter(
                x=df_pivot["step"],
                y=df_pivot[emotion],
                mode="lines",
                name=emotion.capitalize(),
                line=dict(width=2, color=colors.get(emotion, "#CCCCCC")),
            ))

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Steps",
        yaxis_title="Intensity",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig

# Callback to update the memory graph
@app.callback(
    Output("memory-graph", "figure"),
    Input("update-interval", "n_intervals"),
)
def update_memory_graph(n_intervals):
    history = dashboard_data.memory_history

    if not history:
        # Return empty figure if no data
        return go.Figure().update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Steps",
            yaxis_title="Count",
            title="No memory data available",
        )

    # Prepare data
    df = pd.DataFrame(history[-200:])  # Just use last 200 points

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["short_term"],
        mode="lines",
        name="Short-term Memory",
        line=dict(width=2, color="#9d68f2"),
    ))

    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["long_term"],
        mode="lines",
        name="Long-term Memory",
        line=dict(width=2, color="#3b8dff"),
    ))

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Steps",
        yaxis_title="Memory Count",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig

# Callback to update the developmental milestones
@app.callback(
    Output("milestones-content", "children"),
    Input("update-interval", "n_intervals"),
)
def update_milestones(n_intervals):
    # Get developmental milestones data
    milestones = mind.developmental_milestones
    current_stage = mind.state.developmental_stage
    next_stage_value = current_stage.value + 1 if current_stage.value < 5 else 5
    next_stage = DevelopmentalStage(next_stage_value) if next_stage_value <= 5 else None

    # Create content
    content = []

    # Current stage info
    content.append(html.H5(f"Current Stage: {current_stage.name}"))

    # Progress bars for next stage requirements
    if next_stage:
        content.append(html.H6(f"Progress toward {next_stage.name} stage:"))

        thresholds = mind.development_thresholds.get(current_stage, {})

        for metric, threshold in thresholds.items():
            current_value = 0

            if metric == "emotions_experienced":
                current_value = len(milestones.get("emotions_experienced", set()))
            elif metric == "vocabulary_learned":
                current_value = len(milestones.get("vocabulary_learned", set()))
            elif metric in milestones:
                current_value = milestones.get(metric, 0)

            # Calculate percentage
            percentage = min(100, int((current_value / threshold) * 100))

            content.append(html.Div([
                html.Span(f"{metric.replace('_', ' ').title()}: {current_value}/{threshold}"),
                dbc.Progress(
                    value=percentage,
                    style={"height": "10px", "borderRadius": "5px", "marginBottom": "10px"},
                    color="info",
                ),
            ]))
    else:
        content.append(html.Div("Reached maximum developmental stage."))

    # Recent vocabulary
    vocab = list(milestones.get("vocabulary_learned", set()))
    if vocab:
        content.append(html.Hr())
        content.append(html.H6("Recent Vocabulary:"))

        # Display most recent words first
        vocab_sample = sorted(vocab)[-20:]  # Show up to 20 words, most recent first
        vocab_text = ", ".join(vocab_sample)
        content.append(html.Div(vocab_text, style={"marginBottom": "10px"}))
        content.append(html.Div(f"Total vocabulary size: {len(vocab)} words"))

    return content

# Callback to update the beliefs system
@app.callback(
    Output("beliefs-content", "children"),
    Input("update-interval", "n_intervals"),
)
def update_beliefs(n_intervals):
    # Get beliefs data - FIXED
    beliefs = list(mind.belief_network.beliefs.values())

    if not beliefs:
        return html.Div("No beliefs have been formed yet.")

    # Create content
    content = []

    # Table of beliefs
    table_header = [
        html.Thead(html.Tr([
            html.Th("Subject"),
            html.Th("Predicate"),
            html.Th("Object"),
            html.Th("Confidence"),
            html.Th("Stage Formed"),
        ])),
    ]

    rows = []
    for belief in beliefs[-10:]:  # Show last 10 beliefs
        confidence_percent = f"{int(belief.confidence * 100)}%"

        rows.append(html.Tr([
            html.Td(belief.subject),
            html.Td(belief.predicate),
            html.Td(belief.object),
            html.Td(confidence_percent),
            html.Td(belief.developmental_stage.name),
        ]))

    table_body = [html.Tbody(rows)]

    content.append(dbc.Table(
        table_header + table_body,
        bordered=False,
        striped=True,
        size="sm",
        style={"backgroundColor": "transparent"},
    ))

    content.append(html.Div(f"Total beliefs formed: {len(beliefs)}"))

    return content

# Callback to update the needs status
@app.callback(
    Output("needs-content", "children"),
    Input("update-interval", "n_intervals"),
)
def update_needs(n_intervals):
    # Get needs data - FIXED
    needs = mind.need_system.needs

    if not needs:
        return html.Div("No needs data available.")

    # Create content
    content = []

    for name, need in needs.items():
        # Calculate color gradient based on intensity
        color = "#3b8dff" if need.intensity < 0.7 else "#ff9800" if need.intensity < 0.9 else "#f44336"

        content.append(html.Div([
            html.Span(f"{name.capitalize()}: "),
            dbc.Progress(
                value=int(need.intensity * 100),
                style={"height": "10px", "borderRadius": "5px", "marginBottom": "5px"},
                color="info" if need.intensity < 0.7 else "warning" if need.intensity < 0.9 else "danger",
            ),
            html.Div([
                html.Span("Satisfaction: "),
                dbc.Progress(
                    value=int(need.satisfaction_level * 100),
                    style={"height": "6px", "borderRadius": "3px", "marginBottom": "15px"},
                    color="success",
                ),
            ]),
        ]))

    return content

# Callback to update the error log
@app.callback(
    Output("error-log", "children"),
    Input("intermediate-value", "children"),
)
def update_error_log(json_data):
    data = json.loads(json_data) if json_data else {}

    errors = data.get("errors", [])

    if not errors:
        return "System ready. No errors reported."

    # Format errors as text
    error_text = "\n".join([f"[{i+1}] {error}" for i, error in enumerate(errors[-10:])])

    return error_text

# Callback for start button
@app.callback(
    Output("interval-container", "children", allow_duplicate=True),
    Input("start-button", "n_clicks"),
    prevent_initial_call=True,
)
def start_simulation(n_clicks):
    if n_clicks:
        global simulation_thread, simulation_active, dashboard_data

        if simulation_thread is None or not simulation_thread.is_alive():
            simulation_active = True
            dashboard_data.is_running = True
            simulation_thread = threading.Thread(target=run_simulation)
            simulation_thread.daemon = True
            simulation_thread.start()

    return [dcc.Interval(id="update-interval", interval=1000, n_intervals=0)]

# Callback for stop button
@app.callback(
    Output("interval-container", "children", allow_duplicate=True),
    Input("stop-button", "n_clicks"),
    prevent_initial_call=True,
)
def stop_simulation(n_clicks):
    if n_clicks:
        global simulation_active, dashboard_data
        simulation_active = False
        dashboard_data.is_running = False

    return [dcc.Interval(id="update-interval", interval=1000, n_intervals=0)]

# Callback for save button
@app.callback(
    Output("error-log", "children", allow_duplicate=True),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_models_callback(n_clicks):
    if n_clicks:
        success, result = save_models()
        if success:
            return f"Models successfully saved to: {result}"
        return f"Error saving models: {result}"

    return "System ready."

# Callback for applying configuration
@app.callback(
    Output("error-log", "children", allow_duplicate=True),
    [Input("apply-config-button", "n_clicks")],
    [State("step-interval-slider", "value"),
     State("save-interval-input", "value"),
     State("checkpoint-count-input", "value"),
     State("save-directory-input", "value"),
     State("auto-backup-checkbox", "value")],
    prevent_initial_call=True,
)
def apply_configuration(n_clicks, step_interval, save_interval, checkpoint_count, save_directory, auto_backup):
    if n_clicks:
        try:
            # Update dashboard data using Pydantic model
            dashboard_data.training_config = TrainingConfig(
                step_interval=step_interval,
                save_interval_steps=save_interval,
                checkpoint_count=checkpoint_count,
                save_directory=save_directory,
                auto_backup=auto_backup,
            )

            # Update the mind configuration
            config = get_config()
            config.mind.step_interval = step_interval

            return f"Configuration applied successfully. Step interval: {step_interval}s, Save interval: {save_interval} steps"
        except Exception as e:
            return f"Error applying configuration: {e!s}"

    return "System ready."

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app.run_server(debug=True)
