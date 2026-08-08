# schemas.py

Pydantic data models for the mind simulation layer: emotion vocabulary, language-ability metrics, the mind's internal state, and the externally observable projection of that state. Pure declarative schema plus a handful of serialization/description helpers — no simulation logic, no I/O, no side effects. Every model exposes a hand-written `to_dict` rather than relying on pydantic's own serializers.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `EmotionType` | str Enum | 11 members: joy, sadness, anger, fear, disgust, surprise, trust, anticipation, confusion, interest, boredom (mind/schemas.py:14) |
| `Emotion` | model | `name: EmotionType`, `intensity` clamped 0..1, `timestamp` defaulting to `datetime.now()` (mind/schemas.py:28) |
| `Emotion.to_dict` | method | Dict with enum `.value`, float intensity, ISO-8601 timestamp (mind/schemas.py:34) |
| `LanguageAbility` | model | `vocabulary_size: int = 0` plus three 0..1 floats: sentence_complexity, understanding_level, expression_level (mind/schemas.py:42) |
| `LanguageAbility.generate_vocalization` | method | Maps vocabulary_size to one of six fixed English labels (mind/schemas.py:49) |
| `LanguageAbility.to_dict` | method | Flat dict of the four fields (mind/schemas.py:68) |
| `MindState` | model | Internal state: consciousness_level, emotional_state map, current_focus, energy_level, developmental_stage, optional language_ability, timestamp (mind/schemas.py:77) |
| `MindState.to_dict` | method | Serializes all seven fields; stage as `.name`, emotion keys as `.value` (mind/schemas.py:87) |
| `ObservableState` | model | External projection: apparent_mood (-1..1), energy_level, current_focus, recent_emotions, expressed_needs, developmental_stage, vocalization, age_appropriate_behaviors (mind/schemas.py:99) |
| `ObservableState.to_dict` | method | Serializes all eight fields, recursing into each `Emotion.to_dict` (mind/schemas.py:110) |
| `ObservableState.get_developmental_description` | method | Formats a fixed five-entry stage→prose lookup into one sentence (mind/schemas.py:123) |

## Key behaviour

- Validation is entirely declarative via `Field(ge=..., le=...)`. `intensity`, `energy_level`, `consciousness_level`, and the three language floats are bounded 0..1; `apparent_mood` is bounded -1..1. `vocabulary_size` is unbounded.
- Required-vs-default split: `Emotion` requires name+intensity; `MindState` requires consciousness_level+energy_level; `ObservableState` requires apparent_mood+energy_level. Everything else has a default, so bare construction of these three models raises ValidationError.
- `generate_vocalization` is a pure if/elif ladder over `vocabulary_size` only. Exactly 0 returns "pre-linguistic sounds"; below 10, "single words"; below 50, "simple phrases"; below 200, "simple sentences"; below 500, "complex sentences"; 500 or more, "fluent speech".

- Serialization graph: `ObservableState.to_dict` → `Emotion.to_dict` per element of `recent_emotions` (mind/schemas.py:116). `MindState.to_dict` → `LanguageAbility.to_dict` when `language_ability` is not None, else emits null (mind/schemas.py:95). No other cross-model calls.
- Enum handling is inconsistent across the file: `EmotionType` is emitted as `.value` (mind/schemas.py:37, mind/schemas.py:91), `DevelopmentalStage` as `.name` (mind/schemas.py:94, mind/schemas.py:118).
- `datetime.now()` is used as the default factory for `Emotion.timestamp` and `MindState.timestamp` — naive local time, no timezone.
- Both `MindState` and `ObservableState` carry `current_focus`, `energy_level`, and `developmental_stage`; there is no conversion function between them in this file.

## Imports

Third-party: `pydantic` (`BaseModel`, `Field`).

Standard library: `typing` (`Dict`, `List`, `Optional`, `Set`), `datetime` (`datetime`), `enum` (`Enum`).

Project-internal: `core.schemas` → `DevelopmentalStage`.

## Defects and gaps

- mind/schemas.py:8 — `Set` is imported from `typing` and never used anywhere in the file.
- mind/schemas.py:137 — `stage_descriptions.get(self.developmental_stage)` has no default. Any `DevelopmentalStage` member outside the five hardcoded keys yields the string "Developmentally equivalent to None" instead of raising. Whether the enum has exactly those five members is unverifiable from this file alone.
- mind/schemas.py:129-135 — the five stage descriptions and their age ranges are hardcoded in a method body; adding an enum member silently degrades to the None case above.
- mind/schemas.py:55-66 — thresholds 10/50/200/500 are magic numbers with no configuration hook.
- mind/schemas.py:44 — `vocabulary_size` has no `ge=0` constraint. A negative value skips the `== 0` branch at mind/schemas.py:55 and falls into `< 10`, returning "single words" for nonsense input.
- mind/schemas.py:50 — the docstring says the vocalization is generated "based on current language ability", but `sentence_complexity`, `understanding_level`, and `expression_level` are never read by the method; only `vocabulary_size` is.
- mind/schemas.py:94 / mind/schemas.py:37 — `.name` vs `.value` asymmetry within one file. Whether `DevelopmentalStage` member names equal their values, and therefore whether the emitted dict round-trips back through the model, is unverifiable from this file alone.
- mind/schemas.py:80 / mind/schemas.py:105 — the float values in `MindState.emotional_state` and `ObservableState.expressed_needs` carry no `ge`/`le` bounds, unlike the scalar `Emotion.intensity` at mind/schemas.py:31. The same "emotion intensity" quantity is therefore validated in one representation and unvalidated in the other. `expressed_needs` keys are unconstrained `str`, not an enum.
- `ObservableState` has no `timestamp` field while `Emotion` and `MindState` both do, so observations cannot be ordered from the payload alone.
- Defined but never referenced within this file: `LanguageAbility.generate_vocalization`, `MindState.to_dict`, `ObservableState.to_dict`, `ObservableState.get_developmental_description`. Whether external modules call them is unverifiable from this file alone.

## Notes

- The hand-written `to_dict` methods duplicate what `model_dump(mode="json")` would produce, minus the `.name` deviation for `DevelopmentalStage`. Any field added to a model must be added to its `to_dict` manually or it is silently dropped from serialized output.
- No `__all__`, no module-level constants, no runtime code executes on import beyond enum/model class construction.
- `ObservableState.vocalization` is a free-form `Optional[str]`; nothing in this file wires it to `LanguageAbility.generate_vocalization`.
