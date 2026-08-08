"""Command-line interface for the NeuralChild project.

This module provides the entry point for running the Neural Child simulation
from the command line, with options for configuration and visualization.
"""

import argparse
import json
import logging
import os
import signal
import time
from datetime import datetime

from config import Config, get_config, load_config
from core.schemas import DevelopmentalStage
from mind.mind_core import Mind
from mind.networks.consciousness import ConsciousnessNetwork
from mind.networks.emotions import EmotionsNetwork
from mind.networks.perception import PerceptionNetwork
from mind.networks.thoughts import ThoughtsNetwork
from mother.mother_llm import MotherLLM

# For visualization import visualization modules if enabled
try:
    import colorama
    from colorama import Back, Fore, Style

    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# State to track whether simulation should continue running
running = True


def signal_handler(sig, frame):
    """Handle interruption signals.

    Args:
        sig: Signal number
        frame: Current stack frame

    """
    global running
    print("\nReceived signal to stop simulation. Shutting down gracefully...")
    running = False


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def colored_text(text: str, color: str = None, style: str = None) -> str:
    """Format text with color and style if colorama is available.

    Args:
        text: Text to format
        color: Color name (RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE)
        style: Style name (BRIGHT, DIM, NORMAL)

    Returns:
        Formatted text

    """
    if not COLORAMA_AVAILABLE:
        return text

    color_map = {
        "RED": Fore.RED,
        "GREEN": Fore.GREEN,
        "YELLOW": Fore.YELLOW,
        "BLUE": Fore.BLUE,
        "MAGENTA": Fore.MAGENTA,
        "CYAN": Fore.CYAN,
        "WHITE": Fore.WHITE,
    }

    style_map = {
        "BRIGHT": Style.BRIGHT,
        "DIM": Style.DIM,
        "NORMAL": Style.NORMAL,
    }

    result = ""

    if color and color in color_map:
        result += color_map[color]

    if style and style in style_map:
        result += style_map[style]

    result += text + Style.RESET_ALL
    return result


def display_simulation_state(
    mind: Mind,
    mother: MotherLLM,
    iteration: int,
    last_response: str | None = None,
):
    """Display the current state of the simulation.

    Args:
        mind: Mind object
        mother: MotherLLM object
        iteration: Current iteration count
        last_response: Last response from mother

    """
    # Clear screen if not in debug mode
    config = get_config()
    if not config.development.debug_mode and os.name == "posix":
        os.system("clear")
    elif not config.development.debug_mode and os.name == "nt":
        os.system("cls")

    # Get observable state
    observable = mind.get_observable_state()

    # Display header
    stage_name = observable.developmental_stage.name
    print(
        colored_text(
            f"=== NEURAL CHILD SIMULATION (Iteration {iteration}) ===", "CYAN", "BRIGHT"
        )
    )
    print(colored_text(f"Developmental Stage: {stage_name}", "YELLOW", "BRIGHT"))

    # Display mind state
    print(colored_text("\n=== MIND STATE ===", "GREEN", "BRIGHT"))
    print(f"Energy Level: {observable.energy_level:.2f}")
    print(f"Mood: {observable.apparent_mood:.2f} (-1.0 to 1.0)")

    # Display needs
    if observable.expressed_needs:
        print(colored_text("\n=== NEEDS ===", "YELLOW"))
        for need, intensity in observable.expressed_needs.items():
            print(f"{need.capitalize()}: {intensity:.2f}")

    # Display emotions
    if observable.recent_emotions:
        print(colored_text("\n=== EMOTIONS ===", "MAGENTA"))
        for emotion in observable.recent_emotions:
            print(f"{emotion.name.value.capitalize()}: {emotion.intensity:.2f}")

    # Display behaviors
    print(colored_text("\n=== BEHAVIORS ===", "BLUE"))
    if observable.vocalization:
        print(f"Vocalization: {observable.vocalization}")
    if observable.age_appropriate_behaviors:
        for behavior in observable.age_appropriate_behaviors:
            print(f"• {behavior}")

    # Display network outputs
    if observable.current_focus and mind.networks.get(observable.current_focus):
        focused_network = mind.networks[observable.current_focus]
        text_output = focused_network.generate_text_output()

        print(
            colored_text(
                f"\n=== FOCUSED NETWORK: {observable.current_focus.upper()} ===", "CYAN"
            )
        )
        print(text_output.text)

    # Display mother's last response
    if last_response:
        print(colored_text("\n=== MOTHER'S RESPONSE ===", "GREEN"))
        print(f"👩 Mother: {last_response}")

    # Display prompt for user
    print(colored_text("\nPress Ctrl+C to stop simulation", "WHITE", "DIM"))


def initialize_networks(mind: Mind, config: Config):
    """Initialize and register neural networks with the mind.

    Args:
        mind: Mind object to register networks with
        config: Configuration object

    """
    network_configs = config.mind.networks

    # Initialize consciousness network
    consciousness_config = network_configs.get("consciousness", {})
    consciousness = ConsciousnessNetwork(
        input_dim=consciousness_config.get("input_dim", 64),
        hidden_dim=consciousness_config.get("hidden_dim", 128),
        output_dim=consciousness_config.get("output_dim", 64),
    )
    mind.register_network(consciousness)

    # Initialize emotions network
    emotions_config = network_configs.get("emotions", {})
    emotions = EmotionsNetwork(
        input_dim=emotions_config.get("input_dim", 32),
        hidden_dim=emotions_config.get("hidden_dim", 64),
        output_dim=emotions_config.get("output_dim", 32),
    )
    mind.register_network(emotions)

    # Initialize perception network
    perception_config = network_configs.get("perception", {})
    perception = PerceptionNetwork(
        input_dim=perception_config.get("input_dim", 128),
        hidden_dim=perception_config.get("hidden_dim", 256),
        output_dim=perception_config.get("output_dim", 64),
    )
    mind.register_network(perception)

    # Initialize thoughts network
    thoughts_config = network_configs.get("thoughts", {})
    thoughts = ThoughtsNetwork(
        input_dim=thoughts_config.get("input_dim", 64),
        hidden_dim=thoughts_config.get("hidden_dim", 128),
        output_dim=thoughts_config.get("output_dim", 64),
    )
    mind.register_network(thoughts)

    # Set starting developmental stage if specified
    if config.mind.starting_stage != "INFANT":
        try:
            starting_stage = DevelopmentalStage[config.mind.starting_stage]
            mind.state.developmental_stage = starting_stage

            # Update all networks
            for network in mind.networks.values():
                network.update_developmental_stage(starting_stage)

            logger.info(f"Set starting developmental stage to {starting_stage.name}")
        except KeyError:
            logger.warning(
                f"Invalid starting stage: {config.mind.starting_stage}, using INFANT"
            )


def run(config_path: str | None = None) -> None:
    """Run the NeuralChild simulation.

    Args:
        config_path: Path to the configuration file

    """
    # Load configuration
    config = load_config(config_path) if config_path else load_config()

    # Initialize components
    mind = Mind()
    mother = MotherLLM()

    # Initialize neural networks
    initialize_networks(mind, config)

    # Initialize visualization if enabled
    if config.visualization.enabled:
        # Here would be initialization of visualization components
        # For now, we'll use console-based visualization
        pass

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    # Variables for tracking simulation state
    last_mother_response = None
    iteration = 0
    simulation_start_time = time.time()

    # Initialize running state
    global running
    running = True

    # Main simulation loop
    try:
        print(colored_text("🧠 Starting NeuralChild simulation...", "CYAN", "BRIGHT"))
        print(colored_text("Press Ctrl+C to stop", "WHITE", "DIM"))

        while running:
            iteration += 1
            step_start_time = time.time()

            # Advance mind simulation
            mind.step()

            # Get observable state and generate mother response
            response = mother.observe_and_respond(mind)
            if response:
                last_mother_response = response.response

                # Log mother's response
                logger.info(f"Mother: {response.response} (Action: {response.action})")

                # Display response if visualization is enabled
                if config.visualization.enabled:
                    print(f"\n👩 Mother: {response.response}")

            # Display simulation state
            if config.visualization.enabled and iteration % 5 == 0:
                display_simulation_state(mind, mother, iteration, last_mother_response)

            # Calculate time to wait for next step
            step_duration = time.time() - step_start_time
            wait_time = max(0.0, config.mind.step_interval - step_duration)

            # Wait for next step
            if wait_time > 0:
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print(
            colored_text(
                "\nInterrupted by user. Stopping NeuralChild simulation...", "YELLOW"
            )
        )

    except Exception as e:
        logger.error(f"Error in simulation: {e!s}", exc_info=True)
        if config.development.crash_on_error:
            raise
        print(colored_text(f"\nError in simulation: {e!s}", "RED", "BRIGHT"))

    finally:
        # Clean up and finalize
        simulation_duration = time.time() - simulation_start_time

        # Save simulation metrics
        if config.development.record_metrics:
            try:
                metrics = {
                    "simulation_duration": simulation_duration,
                    "iterations": iteration,
                    "developmental_stage": mind.state.developmental_stage.name,
                    "final_state": mind.get_observable_state().to_dict(),
                    "timestamp": datetime.now().isoformat(),
                }

                # Create metrics directory if it doesn't exist
                os.makedirs("metrics", exist_ok=True)

                # Save metrics to file
                metrics_file = f"metrics/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=2)

                logger.info(f"Simulation metrics saved to {metrics_file}")
            except Exception as e:
                logger.error(f"Failed to save metrics: {e!s}")

        print(colored_text("\nNeuralChild simulation stopped.", "CYAN"))
        print(
            f"Simulation ran for {simulation_duration:.2f} seconds ({iteration} iterations)"
        )
        print(f"Final developmental stage: {mind.state.developmental_stage.name}")


def main() -> None:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="NeuralChild: A psychological brain simulation"
    )
    parser.add_argument(
        "--config", "-c", type=str, help="Path to the configuration file"
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Enable visualization"
    )
    parser.add_argument(
        "--simulate-llm", "-s", action="store_true", help="Use simulated LLM responses"
    )
    parser.add_argument(
        "--stage",
        "-S",
        type=str,
        choices=["INFANT", "TODDLER", "CHILD", "ADOLESCENT", "MATURE"],
        help="Starting developmental stage",
    )

    args = parser.parse_args()

    # Check for required colorama package
    if not COLORAMA_AVAILABLE:
        print(
            "WARNING: colorama package not found. Install with 'pip install colorama' for better formatting."
        )

    # Apply command-line overrides to config
    temp_config = load_config(args.config) if args.config else load_config()

    if args.debug:
        temp_config.development.debug_mode = True
        temp_config.logging.level = "DEBUG"
        temp_config.setup_logging()

    if args.visualize:
        temp_config.visualization.enabled = True

    if args.simulate_llm:
        temp_config.development.simulate_llm = True

    if args.stage:
        temp_config.mind.starting_stage = args.stage

    # Save temp config
    temp_config_path = "temp_config.yaml"
    temp_config.to_yaml(temp_config_path)

    try:
        # Run simulation with the temporary config
        run(temp_config_path)
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    main()
