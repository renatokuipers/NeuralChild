"""Tests for the Mind coordinator, its belief network, memory and persistence."""

import json
import os

from hypothesis import given
from hypothesis import strategies as st

from core.schemas import Belief, DevelopmentalStage, Need, NetworkMessage
from mind.mind_core import BeliefNetwork, Mind

CHECKPOINT_FILES = ("mind_state.json", "memories.json", "beliefs.json", "needs.json")


def sensory_input(index: int) -> dict:
    """Build a distinct sensory input payload for the given index."""
    return {
        "type": "sensory_input",
        "visual": [0.1 * index] * 64,
        "auditory": [0.05 * index] * 64,
    }


def test_registered_network_carries_the_documented_learning_rate(mind):
    """register_network applies the stage immediately, giving INFANT an effective rate of 0.00280."""
    network = mind.networks["emotions"]
    weight = network.state.developmental_weights[network.developmental_stage]
    assert round(network.learning_rate * weight, 5) == 0.0028


def test_process_input_forms_a_memory_and_counts_the_interaction(mind):
    """A memory is formed for every input, whether or not a network consumed it."""
    mind.process_input(sensory_input(1))
    assert mind.developmental_milestones["memories_formed"] == 1
    assert mind.developmental_milestones["interactions_count"] == 1
    assert len(mind.short_term_memory) == 1


def test_short_term_memory_is_capped_by_developmental_stage(mind):
    """Short-term memory holds at most 3 + 2 * stage entries, which is 5 at INFANT."""
    for index in range(12):
        mind.process_input(sensory_input(index))
    cap = 3 + 2 * mind.state.developmental_stage.value
    assert len(mind.short_term_memory) == cap


def test_belief_network_assigns_an_identity_on_add():
    """add_belief keys a belief under an identity the belief itself carries."""
    network = BeliefNetwork()
    belief = Belief(subject="ball", predicate="is", object="red")
    belief_id = network.add_belief(belief)
    assert belief_id in network.beliefs
    assert getattr(belief, "id", None) == belief_id


def test_belief_network_indexes_evidence_and_relations_under_the_same_identity():
    """The evidence index and the relationship table are keyed by the belief's own id."""
    network = BeliefNetwork()
    belief = Belief(
        subject="ball", predicate="is", object="red", supporting_memories=["mem-1"]
    )
    belief_id = network.add_belief(belief)
    assert network.evidence_index["mem-1"] == [belief_id]
    assert network.belief_relationships[belief_id] == {}


def test_belief_network_from_dict_restores_the_identity_it_keyed_by():
    """A reloaded belief takes its id from the key it was stored under, not from its payload."""
    restored = BeliefNetwork.from_dict(
        {
            "beliefs": {
                "belief-1": {
                    "id": "stale-identity",
                    "subject": "mother",
                    "predicate": "provides",
                    "object": "comfort",
                    "confidence": 0.75,
                }
            }
        }
    )
    assert restored.beliefs["belief-1"].id == "belief-1"


def test_a_belief_message_forms_a_belief_the_mind_keeps(mind):
    """A belief message off the bus survives into the network and counts as a milestone."""
    mind.message_bus.publish(
        NetworkMessage(
            sender="thoughts",
            receiver="mind",
            message_type="belief",
            content={
                "subject": "ball",
                "predicate": "is",
                "object": "red",
                "confidence": 0.6,
            },
        )
    )
    mind.process_messages()
    assert [belief.object for belief in mind.get_beliefs_about("ball")] == ["red"]
    assert mind.developmental_milestones["beliefs_formed"] == 1


def test_belief_network_round_trips_through_dict():
    """to_dict emits JSON-safe data that from_dict reconstructs without loss."""
    network = BeliefNetwork()
    belief = Belief(
        subject="mother",
        predicate="provides",
        object="comfort",
        confidence=0.75,
        developmental_stage=DevelopmentalStage.INFANT,
    )
    belief_id = network.add_belief(belief)

    serialized = network.to_dict()
    json.dumps(serialized)
    restored = BeliefNetwork.from_dict(serialized)

    assert set(restored.beliefs) == {belief_id}
    recovered = restored.beliefs[belief_id]
    assert (recovered.subject, recovered.predicate, recovered.object) == (
        "mother",
        "provides",
        "comfort",
    )
    assert recovered.confidence == 0.75
    assert recovered.developmental_stage == DevelopmentalStage.INFANT


def test_save_state_writes_a_checkpoint_that_loads_back(mind, checkpoint_dir):
    """save_state writes four complete files and load_state restores what they hold."""
    for index in range(3):
        mind.process_input(sensory_input(index))
    mind.belief_network.beliefs["belief-1"] = Belief(
        subject="ball", predicate="is", object="red", confidence=0.6
    )
    memories_before = mind.developmental_milestones["memories_formed"]

    assert mind.save_state(checkpoint_dir) is True

    for filename in CHECKPOINT_FILES:
        path = os.path.join(checkpoint_dir, filename)
        assert os.path.exists(path), f"{filename} was never written"
        with open(path) as handle:
            json.load(handle)

    reloaded = Mind()
    reloaded.load_state(checkpoint_dir)
    assert reloaded.developmental_milestones["memories_formed"] == memories_before
    assert len(reloaded.short_term_memory) == len(mind.short_term_memory)
    assert "belief-1" in reloaded.belief_network.beliefs
    assert set(reloaded.need_system.needs) == set(mind.need_system.needs)


@given(
    deltas=st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=25,
    )
)
def test_need_intensity_never_leaves_the_unit_range(deltas):
    """No sequence of intensity updates can push a need outside [0, 1]."""
    need = Need(name="comfort")
    for delta in deltas:
        need.update_intensity(delta)
        assert 0.0 <= need.intensity <= 1.0
