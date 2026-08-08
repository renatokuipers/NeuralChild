# emotions.py

Defines `EmotionsNetwork`, a subclass of `NeuralNetwork` that holds a dict-based emotional state and a small feed-forward torch module. It converts inbound "perception" messages into valence-driven emotion updates, decays emotions on an autonomous tick, and re-parameterises reactivity/regulation when the developmental stage changes. It also renders a stage-dependent text summary and attempts a dimension-scaled clone.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `EmotionsNetwork` | class | Emotion network; ctor takes `input_dim=32`, `hidden_dim=64`, `output_dim=32`; calls super with `name="emotions"` (emotions.py:30-38) |
| `.forward(x)` | method | Runs `emotion_processor`, scales by `reactivity`, dampens toward 0.5 when `regulation > 0.5` (emotions.py:75-94) |
| `.process_message(msg)` | method | Handles `"perception"` and `"query"` types; returns `VectorOutput` or `None` (emotions.py:96-150) |
| `.autonomous_step()` | method | Per-emotion decay, deletes emotions that hit exactly 0.0 (emotions.py:270-299) |
| `.update_developmental_stage(stage)` | method | Sets reactivity/regulation from a lookup table and seeds new emotion keys at 0.0 (emotions.py:301-348) |
| `.generate_text_output()` | method | `TextOutput` describing emotions with intensity > 0.3 (emotions.py:350-404) |
| `.clone_with_growth(growth_factor=1.2, min_dim=8)` | method | Intended larger clone; raises before returning — see Defects (emotions.py:406-456) |
| `logger` | constant | Module logger, used only in `clone_with_growth` (emotions.py:19, 450) |

Private helpers: `_update_emotional_state`, `_update_emotion`, `_remember_emotional_event`.

## Key behaviour

- `emotion_processor` is `Linear(input_dim,hidden) → ReLU → Dropout(0.2) → Linear(hidden,hidden) → ReLU → Linear(hidden,output_dim) → Sigmoid` (emotions.py:41-49). `hidden_dim` is never stored on `self`.
- Initial state: JOY 0.3, FEAR 0.1, SURPRISE 0.2, TRUST 0.3; `reactivity` 0.8, `regulation` 0.2 (emotions.py:52-66).
- Perception path builds a single-slot `input_dim` vector: index `int((valence+1)/2 * (input_dim//3))` set to `intensity`, unsqueezed to shape `[1, input_dim]`; forward returns `[1, output_dim]` and `response[0]` is used (emotions.py:113-125). With `input_dim=32` the divisor is 10, so only indices 0-10 are ever addressable for valid valence.

```
message(perception) --> stimulus_vector[32] --> forward --> response[32]
                                                    |
                          _update_emotional_state(valence buckets)
                          _remember_emotional_event(ring buffer)
                                                    |
                                      update_state{emotional_state, pending_messages}
```

- Valence buckets (emotions.py:162-196): `>0.3` raises JOY (`mean*intensity`) and TRUST (`*0.7`), lowers SADNESS/FEAR/ANGER by `0.1*intensity`; `<-0.3` and `<-0.7` picks ANGER or FEAR by a 50/50 `random.random()` coin flip, otherwise SADNESS, plus JOY `-0.1*intensity`; the neutral band raises SURPRISE (`*0.5`), plus INTEREST (`*0.3`) at ≥ TODDLER and ANTICIPATION (`*0.2`) at ≥ CHILD.
- `_update_emotion` clamps to [0,1] after `change * reactivity`; when `change > 0 and regulation > 0.5` it multiplies by `2 - regulation` (emotions.py:230-238).
- Every `_update_emotional_state` call appends a serialized `NetworkMessage` (receiver `"mind"`, type `"emotion"`, priority 0.8) onto `state.parameters["pending_messages"]`, filtering emotions to `v > 0.2` (emotions.py:204-221).
- Memory cap is `5 + developmental_stage.value * 5`; the slice keeps the newest entries and drops the oldest (emotions.py:261-263); memory entries record ISO timestamp and emotions with `v > 0.1`.
- Decay rates per autonomous step: SURPRISE 0.03, FEAR 0.015, JOY 0.005, all others 0.01 — fixed per call, independent of wall time (emotions.py:277-290).
- Stage table (emotions.py:313-319): INFANT 0.8/0.2, TODDLER 0.7/0.4, CHILD 0.6/0.6, ADOLESCENT 0.5/0.7, MATURE 0.4/0.9 (reactivity/regulation).
- Text output: INFANT → top emotion only; TODDLER → top two; all other stages → top three plus a regulation clause when stage value ≥ CHILD. Confidence is `max(0.5, <lowest significant intensity>)` (emotions.py:400-404).

## Imports

Third-party: `torch`, `torch.nn`, `numpy`, plus stdlib `typing`, `random`, `datetime`, `logging`.
Project-internal: `core.neural_network.NeuralNetwork`; `core.schemas` (`NetworkMessage`, `VectorOutput`, `TextOutput`, `DevelopmentalStage`); `mind.schemas.EmotionType`.

## Defects and gaps

- `clone_with_growth` cannot execute. The first failure is at emotions.py:418: `self.emotion_processor[1]` indexes the `ReLU` (the `Linear` is index 0), and `ReLU` has no `.out_features`, so an `AttributeError` is raised before anything else runs. The same bad index recurs at emotions.py:443 and 452.
- Even past that, `copy` is used at emotions.py:429, 430, 435, 439 and `NeuralGrowthRecord` at emotions.py:440, neither of which is imported anywhere in this file — two further `NameError`s block the method.
- `clone_with_growth` contains no weight transfer at all; were it reachable, the "clone" would be a freshly randomly-initialised module with only the Python-side emotional state and counters carried over.
- Nothing in this file trains the network: no loss, optimizer, backward pass, or parameter update exists, so `emotion_processor` is a fixed random projection and `response.mean()` is near-constant across stimuli — the valence branch, not the network, determines every emotion change.
- `numpy as np` is imported at emotions.py:11 and never used.
- The inline comment at emotions.py:234 claims higher regulation dampens increases, but the code multiplies by `2 - regulation` (emotions.py:235), a factor of 1.1–1.5 for regulation in (0.5, 0.9] — it amplifies increases instead.
- `Dropout(0.2)` is active whenever the module is in the default train mode; `process_message` wraps `forward` in `torch.no_grad()` but never calls `eval()`, so inference outputs are stochastic (emotions.py:44, 121-122).
- `except ValueError: pass` at emotions.py:147-148 silently swallows an unknown emotion name and falls through to `return None`.
- `pending_messages` is only ever appended to (emotions.py:219-221); nothing in this file trims or clears it, so the state parameter grows without bound.
- The truthiness checks on `sorted_emotions` at emotions.py:377 and 387 can never be false — the empty case already returned at emotions.py:359-364, so the list always holds at least one item.
- The `"perception"` branch requires both `"stimulus"` and `"valence"` keys (emotions.py:108); a message carrying only one falls through to `return None` with no log or error.
- `valence` and `intensity` are read from message content without range validation (emotions.py:110-111). At `input_dim=32`, `valence_idx` (emotions.py:117) exceeds the tensor for `valence ≳ 5.4`, raising `IndexError` at emotions.py:118; `valence < -1` yields a negative index that silently wraps to the end of the vector instead of erroring.
- The dampening in `forward` (emotions.py:91-92) is inverted in practice. It is applied *after* scaling by `reactivity`, and the branch only runs at stages where `regulation > 0.5` — CHILD/ADOLESCENT/MATURE, whose reactivity is 0.6/0.5/0.4 (emotions.py:316-318). Almost every scaled value is therefore below the hardcoded 0.5 centre, `sign` is negative, and the term is *added*: at MATURE it compresses the output into roughly [0.4, 0.48], raising the mean response rather than suppressing it.
- Zero-valued emotion keys are permanent. The deletion at emotions.py:293-294 sits inside the `current > 0` guard (emotions.py:289), so entries already at 0.0 are skipped forever. Two paths create them: the stage unlocks at emotions.py:331-348 seed keys at 0.0, and `_update_emotion` with a negative `change` on an absent emotion (e.g. SADNESS/ANGER on positive valence, emotions.py:168-170) inserts it clamped to 0.0.
- Arithmetic at emotions.py:261 (`developmental_stage.value * 5`) requires `DevelopmentalStage.value` to be numeric, while emotions.py:375/381 compare enum members directly — the numeric assumption is unverifiable from this file alone.

## Notes

- Base-class members `input_dim`, `output_dim`, `state.parameters`, `update_state`, `developmental_stage`, `growth_metrics`, `experience_count`, `growth_history`, and `NetworkMessage.to_dict()` are used but defined elsewhere; their existence and semantics are unverifiable from this file alone.
- Emotion members referenced beyond the four seeded ones — SADNESS, ANGER, INTEREST, ANTICIPATION, DISGUST, CONFUSION, BOREDOM — are assumed to exist on `EmotionType`.
- Negative changes bypass the regulation branch entirely, so decreases scale only with `reactivity`.
