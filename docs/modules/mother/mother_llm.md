# mother_llm.py

Implements the "Mother" caregiver agent that observes a `Mind` object's externally visible state and emits a nurturing response. Responses come either from a hardcoded template bank keyed by developmental stage and response type, or from an LLM call via `chat_completion`, with a template fallback on failure. The file is self-contained apart from five project imports; all templates, techniques, weights and thresholds are literal in-source data.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `MotherResponse` | pydantic model | Fields `understanding`, `response`, `action` (all required str), `development_focus` (optional str), `timestamp` (defaults to `datetime.now`). |
| `MotherResponse.to_dict` | method | Flat dict of the five fields; timestamp as ISO string. |
| `MotherLLM` | class | Stateful caregiver; no constructor arguments. |
| `MotherLLM.observe_and_respond` | method | Takes a `Mind`, returns `MotherResponse` or `None`; appends to `interaction_history` and advances `last_response_time` on success. |
| `MotherLLM.interaction_history` | attribute | List of `{observation, response, timestamp}` dicts, trimmed to the last 100 (mother_llm.py:302-303). |
| `MotherLLM.personality` | attribute | Five float traits (patience 0.85, warmth 0.9, playfulness 0.8, teaching_focus 0.7, emotional_support 0.9). |
| `MotherLLM.developmental_techniques` | attribute | `stage -> {language, emotional, cognitive, physical} -> list[str]`, keyed by `DevelopmentalStage` enum members. |
| `MotherLLM.response_templates` | attribute | `"INFANT"/"TODDLER"/"CHILD"/"ADOLESCENT"/"MATURE" -> {comfort, play, rest, teach} -> 4 strings each`. |

All remaining methods (`_load_response_templates`, `_should_respond`, `_construct_situation`, `_determine_response_type`, `_select_development_focus`, `_select_technique`, `_generate_response`, `_get_template_response`, `_fallback_response`) are private helpers, each referenced within this file.

## Key behaviour

```
observe_and_respond(mind)
  |- elapsed < response_interval (10.0 s)? -> None          [mother_llm.py:257]
  |- mind.get_observable_state()
  |- _should_respond(state) false? -> None                  [mother_llm.py:310]
  |- _construct_situation -> natural-language string
  |- _determine_response_type -> comfort|play|rest|teach
  |- _select_development_focus -> language|emotional|cognitive|physical
  |- _select_technique(stage, focus) -> phrase
  `- _generate_response
        |- template exists AND random() < 0.7 -> MotherResponse(template)
        `- else chat_completion(structured_output=True)
              |- dict has understanding/response/action -> MotherResponse
              |- otherwise -> _fallback_response
              `- any Exception -> _fallback_response
```

- Rate gate is a hard 10.0 s wall-clock interval; `last_response_time` only advances when a response is actually produced (mother_llm.py:305), so a suppressed turn leaves the gate open.
- `_should_respond` short-circuits true on any need intensity > 0.7, any of fear/sadness/anger at intensity > 0.6, or any non-empty `vocalization`. Otherwise it is a per-stage coin flip: INFANT 0.8, TODDLER 0.6, CHILD 0.4, everything else 0.3 (mother_llm.py:334-345).
- `_determine_response_type` precedence: strongest expressed need > 0.5 (only if named comfort/play/rest) → negative emotion (sadness/fear/anger/disgust) > 0.5 → "comfort" → probabilistic "teach" when energy > 0.3 and mood > -0.3, with probability `0.3 + stage.value * 0.1` → "play" when energy > 0.6 → "rest" when energy < 0.3 → stage default (INFANT "comfort", else "teach").
- `_construct_situation` mood bands: < -0.5 very distressed, < -0.2 somewhat upset, > 0.5 very happy, > 0.2 cheerful, else content. Energy bands: < 0.3 tired, > 0.7 energetic. Needs are only narrated when intensity > 0.4 (mother_llm.py:396).
- `_select_development_focus` builds four weights at 1.0, then boosts per stage (INFANT emotional/physical 1.5, cognitive 0.7; TODDLER language 1.5, physical 1.3; CHILD cognitive 1.5, language 1.3; ADOLESCENT emotional 1.5, cognitive 1.3), normalizes, and does a cumulative-sum draw. MATURE has no branch, so it stays uniform.
- `_select_technique` falls back to a flat pool of every technique across all stages and focus areas when the requested `(stage, focus)` pair is missing (mother_llm.py:533-539).
- Only the LLM path consumes `situation` (mother_llm.py:598) and `technique` (mother_llm.py:587); the template path discards both and stamps `understanding` as a fixed `"Child appears to need {response_type}."` string (mother_llm.py:571).
- Template lookup keys on `stage.name`, which must be the uppercase strings present in `response_templates`; `developmental_techniques` instead keys on the enum member itself.

## Imports

Third-party: `pydantic` (`BaseModel`, `Field`). Standard library: `typing`, `datetime`, `random`, `json`, `logging`.

Project-internal: `mind.mind_core.Mind`, `mind.schemas.ObservableState`, `core.schemas.DevelopmentalStage`, `utils.llm_module.chat_completion`, `config.config`.

## Defects and gaps

- `json` is imported at mother_llm.py:10 and never used.
- `config` is imported at mother_llm.py:18 and never used; `response_interval = 10.0` (mother_llm.py:53) and the history cap of 100 (mother_llm.py:302) are hardcoded rather than configured.
- `self.personality` (mother_llm.py:54-60) is populated but never read by any code in this file — it influences nothing.
- `stage_name` is assigned and unused at mother_llm.py:276 and again at mother_llm.py:682. (The one at mother_llm.py:580 is used in the prompt.)
- `_generate_response` returns a `MotherResponse` on every path (template, LLM success, LLM-shape-mismatch fallback, exception fallback). Therefore `if response:` at mother_llm.py:293 is always true and the trailing `return None` at mother_llm.py:308 is unreachable, contradicting the docstring's "None if generation fails".
- The `return random.choice(focus_areas)` fallback at mother_llm.py:519 is effectively unreachable: the weights are normalized to sum to 1.0 immediately before the loop, so `r <= cumulative` is satisfied on the final iteration except under float rounding.
- Bare `except Exception` at mother_llm.py:627 logs and silently substitutes a canned response — an LLM outage is indistinguishable from a template-driven turn in `interaction_history`.
- mother_llm.py:607 assumes `chat_completion` returns a mapping. If it returns a string, `"understanding" in response` performs a substring test that can pass, and the subsequent subscript raises `TypeError`, which is then swallowed by the handler at mother_llm.py:627 rather than reported as a contract violation.
- `teach_probability` at mother_llm.py:444 multiplies `state.developmental_stage.value` by 0.1, requiring the enum values to be numeric and ordered. Unverifiable from this file alone; a string-valued enum raises `TypeError` here.
- Emotion matching compares `emotion.name.value` against lowercase literals at mother_llm.py:326 and mother_llm.py:438; any casing change in the emotion enum silently disables both branches.
- `_determine_response_type` only maps needs named exactly `comfort`, `play`, or `rest` (mother_llm.py:432); a strong need with any other name is ignored and falls through to the emotion check.
- `_load_response_templates` returns a hardcoded literal; its docstring claim of loading is aspirational, acknowledged by the note at mother_llm.py:104.
- `MotherResponse.timestamp` is set at construction, but `interaction_history` stores a second, separately generated timestamp (mother_llm.py:298), so the two can differ.
- The stage-specific emergency branches in `_fallback_response` (mother_llm.py:683-710) are dead code for every stage the templates cover: `_determine_response_type` can only return `comfort`, `play`, `rest` or `teach`, and every one of the five template stages defines all four, so `_get_template_response` at mother_llm.py:671 always returns a template and the method returns at mother_llm.py:674 first. Only a `DevelopmentalStage` member whose `name` is absent from `response_templates` would reach them; whether such a member exists is unverifiable from this file alone.
- For the same reason the LLM is reached on roughly 30% of responding turns (mother_llm.py:569) — the template gate almost never fails on the "no template" condition, only on the random draw.

## Notes

- Randomness is unseeded (`random` module global state), so `observe_and_respond` is nondeterministic across runs at six separate decision points: `_should_respond`, `_determine_response_type`, `_select_development_focus`, `_select_technique`, the 0.7 template gate, and `_get_template_response`'s choice of phrasing.
- `Mind` is imported solely for the type annotation on `observe_and_respond`; the only method invoked on it is `get_observable_state()`.
- Methods called on `ObservableState` in this file: `get_developmental_description()`, `to_dict()`, and attribute access on `expressed_needs`, `recent_emotions`, `vocalization`, `apparent_mood`, `energy_level`, `current_focus`, `age_appropriate_behaviors`, `developmental_stage`. Whether those exist is unverifiable from this file alone.
