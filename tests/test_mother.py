"""Tests for the MotherLLM caregiver and the ObservableState boundary it sees."""

from datetime import datetime, timedelta

import pytest

from core.schemas import DevelopmentalStage
from mother import mother_llm
from mother.mother_llm import MotherLLM, MotherResponse

RESPONSE_TYPES = ("comfort", "play", "rest", "teach")

OBSERVABLE_FIELDS = {
    "apparent_mood",
    "energy_level",
    "current_focus",
    "recent_emotions",
    "expressed_needs",
    "developmental_stage",
    "vocalization",
    "age_appropriate_behaviors",
}


@pytest.fixture(autouse=True)
def unreachable_llm(monkeypatch):
    """Cut the LLM transport for every test here, so none of them depend on a running server."""

    def refuse(*args, **kwargs):
        raise RuntimeError("LM Studio is not running")

    monkeypatch.setattr(mother_llm, "chat_completion", refuse)


@pytest.fixture
def mother() -> MotherLLM:
    """A caregiver whose rate gate has already elapsed."""
    instance = MotherLLM()
    instance.last_response_time = datetime.now() - timedelta(seconds=60)
    return instance


def test_observable_state_exposes_nothing_internal(mind):
    """The mother's only view of the child carries no memories, beliefs or network state."""
    observable = mind.get_observable_state()
    assert set(observable.model_dump()) == OBSERVABLE_FIELDS


def test_mother_reads_only_the_observable_state(mind, mother, monkeypatch):
    """observe_and_respond reaches the mind through get_observable_state exactly once."""
    original = mind.get_observable_state
    calls = []

    def counted():
        calls.append(1)
        return original()

    monkeypatch.setattr(mind, "get_observable_state", counted)
    mother.observe_and_respond(mind)
    assert len(calls) == 1


def test_mother_stays_silent_inside_the_rate_gate(mind):
    """A caregiver that has just spoken produces nothing until the interval elapses."""
    fresh = MotherLLM()
    fresh.last_response_time = datetime.now()
    assert fresh.observe_and_respond(mind) is None


def test_mother_responds_when_a_need_is_urgent(mind, mother, unreachable_llm):
    """An expressed need above 0.7 draws a usable response while chat_completion raises."""
    assert any(
        intensity > 0.7
        for intensity in mind.get_observable_state().expressed_needs.values()
    )
    response = mother.observe_and_respond(mind)
    assert isinstance(response, MotherResponse)
    assert response.response.strip()
    assert response.action in RESPONSE_TYPES


def test_responding_records_the_interaction(mind, mother):
    """A produced response is appended to interaction history with the observation that caused it."""
    mother.observe_and_respond(mind)
    assert len(mother.interaction_history) == 1
    entry = mother.interaction_history[0]
    assert set(entry) == {"observation", "response", "timestamp"}


@pytest.mark.parametrize("stage", list(DevelopmentalStage))
@pytest.mark.parametrize("response_type", RESPONSE_TYPES)
def test_template_bank_covers_every_stage_and_response_type(
    mother, stage, response_type
):
    """Each stage defines a template for each response type, which is why the LLM is optional."""
    assert mother._get_template_response(stage, response_type)
