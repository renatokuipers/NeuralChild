"""Tests for the neural network base class and the five network subclasses."""

import pytest
import torch
from pydantic import ValidationError

from core.neural_network import GrowthMetrics

METRIC_FIELDS = (
    "connection_density",
    "plasticity",
    "pruning_rate",
    "specialization",
    "integration",
    "adaptability",
)


def test_forward_produces_output_of_declared_width(network):
    """Every network maps a batch of input_dim to a batch of output_dim."""
    output = network.forward(torch.randn(1, network.input_dim))
    assert output.shape == (1, network.output_dim)


@pytest.mark.parametrize("field", METRIC_FIELDS)
def test_growth_metrics_accepts_explicit_values(field):
    """Constructing with an in-range value keeps that value instead of raising."""
    metrics = GrowthMetrics(**{field: 0.8})
    assert getattr(metrics, field) == 0.8


@pytest.mark.parametrize("invalid", (-0.1, 1.5))
@pytest.mark.parametrize("field", METRIC_FIELDS)
def test_growth_metrics_rejects_out_of_range_values(field, invalid):
    """Either bound of [0, 1] is enforced, with a validation error naming the field."""
    with pytest.raises(ValidationError) as excinfo:
        GrowthMetrics(**{field: invalid})
    assert field in str(excinfo.value)


@pytest.mark.parametrize("plasticity", [0.0, 0.5, 1.0])
def test_set_plasticity_updates_derived_learning_rate(network, plasticity):
    """set_plasticity stores the value and recomputes the learning rate from it."""
    network.set_plasticity(plasticity)
    assert network.growth_metrics.plasticity == plasticity
    assert network.learning_rate == pytest.approx(0.01 * (0.5 + plasticity))


def test_a_highly_active_layer_triggers_growth(network):
    """Activity above growth_threshold reaches a grow attempt, which restarts the experience count."""
    network.min_experiences_before_growth = 0
    network.experiences_since_last_growth = 1
    network.growth_metrics.plasticity = 1.0
    network.activity_tracker = {"probe": [network.growth_threshold + 0.1] * 5}
    network._check_for_network_growth("test")
    assert network.experiences_since_last_growth == 0


def test_an_idle_layer_triggers_pruning(network):
    """Activity below pruning_threshold reaches a prune attempt, which restarts the experience count."""
    network.min_experiences_before_growth = 0
    network.experiences_since_last_growth = 1
    network.growth_metrics.pruning_rate = 1.0
    network.activity_tracker = {"probe": [network.pruning_threshold / 2] * 5}
    network._check_for_network_growth("test")
    assert network.experiences_since_last_growth == 0


def test_unregistered_network_has_nonzero_developmental_weight(network):
    """A network that was never registered with a Mind still carries a usable stage weight."""
    assert network.state.developmental_weights[network.developmental_stage] > 0.0


def test_unregistered_network_updates_parameters_when_learning(network):
    """experiential_learning on an unregistered network changes at least one parameter."""
    before = [parameter.detach().clone() for parameter in network.parameters()]
    network.experiential_learning(torch.randn(1, network.input_dim))
    after = list(network.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_experience_count_tracks_calls_even_when_nothing_can_be_learned(network):
    """Every call counts as an experience, including with effective_lr pinned to zero."""
    network.state.developmental_weights[network.developmental_stage] = 0.0
    before = [parameter.detach().clone() for parameter in network.parameters()]
    for _ in range(3):
        network.experiential_learning(torch.randn(1, network.input_dim))
    assert network.experience_count == 3
    assert all(torch.equal(b, a) for b, a in zip(before, network.parameters()))


def test_clone_with_growth_returns_a_larger_network(network):
    """Each subclass clones itself at scaled dimensions instead of raising."""
    clone = network.clone_with_growth(growth_factor=1.2)
    assert clone.input_dim > network.input_dim
    assert clone.output_dim > network.output_dim


def test_clone_with_growth_records_the_event(network):
    """A grown clone carries a growth record describing the expansion."""
    clone = network.clone_with_growth(growth_factor=1.2)
    assert len(clone.growth_history) > len(network.growth_history)
