"""Shared fixtures for the NeuralChild suite."""

import pytest

from communication.message_bus import GlobalMessageBus
from config import config
from mind.language import LanguageNetwork
from mind.mind_core import Mind
from mind.networks.consciousness import ConsciousnessNetwork
from mind.networks.emotions import EmotionsNetwork
from mind.networks.perception import PerceptionNetwork
from mind.networks.thoughts import ThoughtsNetwork

NETWORK_CLASSES = {
    "consciousness": ConsciousnessNetwork,
    "emotions": EmotionsNetwork,
    "perception": PerceptionNetwork,
    "thoughts": ThoughtsNetwork,
    "language": LanguageNetwork,
}

REGISTERED_BY_CLI = ("consciousness", "emotions", "perception", "thoughts")


def build_network(name: str):
    """Construct the named network at the dimensions declared in config.mind.networks."""
    dims = config.mind.networks[name]
    return NETWORK_CLASSES[name](
        input_dim=dims["input_dim"],
        hidden_dim=dims["hidden_dim"],
        output_dim=dims["output_dim"],
    )


@pytest.fixture(autouse=True)
def isolated_message_bus():
    """Reset the global message bus around each test so Mind subscriptions never accumulate."""
    GlobalMessageBus.reset()
    yield
    GlobalMessageBus.reset()


@pytest.fixture
def mind() -> Mind:
    """A Mind with the networks cli.py registers, each attached through register_network."""
    instance = Mind()
    for name in REGISTERED_BY_CLI:
        instance.register_network(build_network(name))
    return instance


@pytest.fixture(params=sorted(NETWORK_CLASSES))
def network(request):
    """Each of the five networks in turn, constructed but never registered with a Mind."""
    return build_network(request.param)


@pytest.fixture
def checkpoint_dir(tmp_path) -> str:
    """An empty directory for save_state and load_state round trips."""
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    return str(directory)
