# mind_core.py

Central coordinator for the simulation: owns a registry of `NeuralNetwork` instances, a two-tier memory store, a belief graph, and a need/motivation system, and advances all of them one tick at a time via `Mind.step()`. Also holds developmental-stage progression logic (INFANT → MATURE) and persistence — four JSON files written directly here, plus per-network model files delegated to `network.save_model`/`load_model`. 2479 lines, four top-level classes plus a module-level logger.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `MemoryCluster` | pydantic model | Labelled group of memory IDs with importance, timestamps, optional `centroid`; `access`, `add_memory`, `remove_memory`, `to_dict`. |
| `BeliefNetwork` | pydantic model | Belief store + bidirectional relationship map + memory→belief evidence index + contradiction list; add/remove/relate, `add_contradiction`, `update_with_new_evidence`, `resolve_contradictions`, `get_related_beliefs`, `to_dict`/`from_dict`. |
| `NeedMotivationSystem` | pydantic model | Seven default needs with per-stage priority profiles; time-driven `update_needs`, `satisfy_need`, `get_dominant_need`, `get_expressed_needs`, `update_developmental_stage`, `get_need_trend`, `to_dict`. |
| `Mind` | class | Plain (non-pydantic) coordinator. `register_network`, `process_input`, `step`, `get_state`, `get_observable_state`, memory/belief queries, `save_state`, `load_state`. |
| `logger` | module constant | `logging.basicConfig(level=INFO)` is executed at import time (mind_core.py:40). |

## Key behaviour

- `Mind.__init__` (mind_core.py:854) starts at `consciousness_level=0.2`, `energy_level=0.7`, stage INFANT, empty vocabulary; subscribes to `GlobalMessageBus` with `MessageFilter(receiver="mind", min_priority=0.3)` (mind_core.py:876-880).
- `step()` (mind_core.py:1051) fixed order: drain bus → per-network `generate_text_output()` + optional `autonomous_step()` + drain `state.parameters["pending_messages"]` → needs → memory consolidation → belief update → network growth → mind state → developmental check → self-reflection. `simulation_time` accumulates wall-clock duration of the step itself, not simulated time.

```
input_data ─► process_input ─► perception net (128-dim: 64 visual ++ 64 auditory)
                            ├─► language net (len(tokens) x 10, TODDLER+ only)
                            └─► _form_memory ─► short_term (cap 3 + 2*stage.value)
                                                    │ consolidation
                                                    ▼
                                              long_term ─► _cluster_memory (TODDLER+)
                                                        └► belief_network.update_with_new_evidence
```

- Timing gates, all early-return without updating their tracker when not due: needs `config.mind.need_update_interval` (1262), consolidation `config.mind.memory_consolidation_interval` (1324), growth `config.mind.network_growth_check_interval` (1594), development `config.mind.development_check_interval` (1695); belief update uses a hardcoded 60.0 s (1528).
- Need dynamics (`update_needs`, mind_core.py:636): base rate 0.0003/s scaled by stage priority; `stimulation` multiplied by `1 + satisfaction`, `rest` by `2 - satisfaction`, `autonomy` by `0.2 * stage.value`; satisfaction decays 0.0002/s; both histories capped at 100 entries.
- Consolidation criteria (1331): |valence| > 0.6, or strength > 1.5, or age > 300 s. Forget threshold 0.1, raised to 0.3 for INFANT and 0.2 for TODDLER; decay 0.01, halved for `emotionally_significant`, x0.7 for `well_reinforced`.
- Stage advance (`_check_developmental_progress`, 1690) requires every threshold in `development_thresholds` for the current stage to be met; `config.mind.development_acceleration > 1.0` adds a per-check probability of `(mean_progress/100) * (accel-1) * 0.1` to skip the requirements (1738).
- `_process_mind_message` (1133) handles `emotion`, `belief`, `consciousness`, `language_output`, `need`. Message types are matched by string literal.
- `_self_reflection` (1783) is inert until `self_awareness_level >= 0.3`. That field starts at 0.1 (946) and is only raised by an incoming `consciousness` message (1215) or by `load_state` restoring a saved value (2346). Corrective actions additionally need `self_awareness * 0.2 * stage.value > 0.5` and stage CHILD or later (1811).
- Growth: `growth_schedule` is 0.0 for INFANT, 0.001/0.002/0.003/0.001 thereafter, applied as a per-check `random.random()` probability, calling `clone_with_growth(growth_factor=1.2)` and swapping the instance in `self.networks` (1617-1620).

## Imports

Third-party: `numpy`, `torch`, `pydantic` (`BaseModel`, `Field`, `validator`, `root_validator`). Standard library: `typing`, `datetime`, `random`, `logging`, `os`, `json`, `uuid`, `copy`, `threading`, `time`.

Project-internal: `core.schemas` (`NetworkMessage`, `Memory`, `Belief`, `Need`, `DevelopmentalStage`), `mind.schemas` (`MindState`, `ObservableState`, `Emotion`, `EmotionType`, `LanguageAbility`), `core.neural_network` (`NeuralNetwork`, `GrowthMetrics`), `communication.message_bus` (`GlobalMessageBus`, `MessageFilter`), `config` (`config`).

## Defects and gaps

- **Infinite recursion at CHILD stage** — mind_core.py:1921-1923: when vocabulary has fewer than 10 entries the CHILD branch calls `_generate_age_appropriate_vocalization()` again with an unchanged stage, re-entering the same branch. Comment claims it falls back to toddler speech; it cannot. Only escapes if a `language` network supplies `recent_utterances` (1891).
- **Save/load path mismatch** — `save_state` builds the network path as `{name}.{format}` with `format` defaulting to `"pytorch"` (2204, 2219); `load_state` builds `{name}.pt` (2465) and takes a `format` parameter (2284) it never uses. The two paths cannot agree under the defaults, and the miss is logged as "Network model not found" (2471).
- **Cluster restore always raises `TypeError`** — `MemoryCluster.to_dict` emits an `"id"` key (mind_core.py:76); `load_state` passes `id=cluster_id, **cluster_dict` (2413-2418) after popping only the stage and the two timestamps, so `id` is supplied twice. The `except` at 2421 downgrades it to a warning, so clusters never survive a reload. `BeliefNetwork.from_dict` has the same `id=belief_id, **belief_data` shape (498-504); whether it collides depends on `Belief.to_dict`, which is not in this file.
- **Asymmetric memory stage round-trip** — `save_state` serializes memories via `memory.dict()` with no enum→name conversion (2247-2254), while `load_state` reads the field with `DevelopmentalStage[stage_name]` (2363, 2385), i.e. by member *name*. Whether that lookup actually fails depends on how `DevelopmentalStage` and `Memory` serialize (not in this file); if it does, the per-memory `except` at 2376/2398 reduces it to a warning and the memory is dropped.
- **Evidence index cleanup targets the wrong container** — 1387-1389 iterates `evidence_index.values()` (lists of belief IDs) looking for `memory.id`; the key holding that memory is never removed. Correct target is the dict key.
- **Dead branch in `_find_relevant_beliefs`** — 344-345 tests `isinstance(value, dict)` on the output of `_flatten_dict`, which by construction contains no dict values (422-442).
- **Self-reflection's extra consolidation is always a no-op** — 1813-1815 calls `_consolidate_memories()` again, but that method's own gate (1324) was either just satisfied and its tracker reset to now (1391) earlier in the same `step()`, or was not due; either way the second call returns immediately.
- `_process_mind_message` has no `development` branch, so the message built and self-dispatched at 1757-1771 is silently discarded.
- Hash-based language embedding at 1034-1036 comments "consistent embedding" but `hash()` on `str` is salted per process, so the same token maps to different vectors across runs. `hash_val % 1000` also caps the value at three digits while the format spec pads to width 10, so seven of the ten embedding dimensions are always zero.
- `process_input` normalizes visual/auditory payloads to 64 elements only when they are `list` (1000-1011), but concatenates unconditionally at 1014; a non-list payload skips normalization and `+` stops meaning concatenation.
- `_form_belief_relationships` sums up to 0.6 + 0.3 + 0.4 + 0.8 = 2.1 (1562-1576) and passes the unclamped total to `add_relationship` (1587), whose documented range is 0.0–1.0 (181).
- `current_focus` is only recomputed while falsy (1638), so once set it never tracks the most active network again.
- `_update_mind_state` line 1658 truncates `vocabulary_learned` to the language network's reported `vocabulary_size` by slicing an unordered set — arbitrary words are destroyed on every step.
- Unused local assignments: `old_confidence` (253, 267), `old_conf1`/`old_conf2` (298-299), `old_conf` (315), `emotional_valence` (379), `affected_beliefs` (1535).
- Unused imports: `Set`, `Union`, `Callable` (7), `timedelta` (8), `copy` (16), `threading` (17), `time` (18), `validator`, `root_validator` (19), `GrowthMetrics` (35).
- `network.input_dim` / `output_dim` are read at 1612 outside the `try`, so a network lacking those attributes crashes `step()` rather than being caught by the handler at 1626.
- Redundant guard: `if "language" in self.networks` at 1038 is already guaranteed by the enclosing condition at 1018.
- `generate_text_output()` is called unguarded (1061) while `autonomous_step` gets a `hasattr` check (1064) — inconsistent contract assumptions about `NeuralNetwork`.
- `language_output` handling (1232-1243) records a need-expression memory but never satisfies or adjusts the need it identified.
- Comment at 1616 asserts "all subclasses implement clone_with_growth" — not verifiable from this file, and the code still wraps the call in a broad `try/except`.
- Both `save_state` (2280) and `load_state` (2476) catch bare `Exception` and reduce every failure to a log line plus a boolean.
- `load_state` is annotated `-> None` (2284) but returns `False`/`True` (2295, 2474, 2479), and its docstring documents a boolean return.
- Never referenced anywhere in this file: `BeliefNetwork.remove_belief` (136), `NeedMotivationSystem.get_dominant_need` (723), `NeedMotivationSystem.get_need_trend` (809).
- `MemoryCluster.centroid` is declared (48) and serialized (79) but never assigned by any code path here; `_calculate_memory_match` (1453) uses type/field equality instead.

## Notes

- Pydantic v1 idioms throughout: inner `class Config` (54, 102, 539), `validator`/`root_validator` imported but never applied (19), `memory.dict()` (2248). Which pydantic version is actually installed is not determinable from this file.
- `Mind` is a plain class; only the three helper containers are pydantic models.
- `DevelopmentalStage` is treated as 1-based ordinal arithmetic (`stage.value - 1` at 1676, `value + 1` at 1703, `3 + value*2` at 1313); changing the enum's numbering silently rescales memory capacity, language estimates, and stage transitions.
- Cross-file consistency of the emotion key format cannot be checked here: `_form_memory` writes `emotional_context` keyed by `EmotionType.name` (1278) while `_calculate_evidence_impact` reconstructs it with the value-based `EmotionType(emotion_str)` (399) and swallows `ValueError`. If names and values differ, the entire emotional-congruence block (405-416) is dead. Unverifiable from this file alone.
