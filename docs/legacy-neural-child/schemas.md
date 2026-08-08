# schemas.py

A single-model Pydantic module (36 lines) defining `MotherResponse`, the structured payload describing one mother-side utterance plus its scalar ratings. The file contains only declarations: no functions, no validators, no methods, no side effects. Which producer fills the model and which consumer reads it is unverifiable from this file alone.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `MotherResponse` | Pydantic model (`BaseModel`) | Seven fields: one required text field plus six defaulted metric/label fields; carries a JSON-schema example block. schemas.py:4 |

No `__all__`, no module-level constants, no helper functions.

## Key behaviour

Field table (all declarations at schemas.py:6-17):

| Field | Annotation | Default | Notes |
| --- | --- | --- | --- |
| `content` | `str` | none — required | Only field whose omission raises a validation error |
| `emotional_context` | `Optional[dict]` | `{joy:0.5, trust:0.5, fear:0.1, surprise:0.3}` | Unparameterized `dict`; keys/values unvalidated |
| `reward_score` | `float` | `0.7` | Only non-zero scalar default |
| `success_metric` | `float` | `0.0` | |
| `complexity_rating` | `float` | `0.0` | |
| `self_critique_score` | `float` | `0.0` | |
| `cognitive_labels` | `Optional[list]` | `[]` | Unparameterized `list`; element type unvalidated |

- Validation surface is minimal: any payload whose `content` is a string validates. The four scalars accept numeric strings under Pydantic v2's default lax coercion, and the two container fields accept any dict-like / list-like input.
- No bounds are declared. Scalars accept negatives, values above 1.0, and `inf`/`nan` (float coercion accepts `float("nan")`); the 0.0-1.0 ranges implied by the example are convention only.
- `model_config` (schemas.py:19-36) sets `json_schema_extra.examples` only. It changes emitted JSON Schema; it has zero effect on runtime validation, and nothing in this file checks an instance against the example.
- The example's `content` string embeds a bracketed token, `[HUG]` (schemas.py:22), implying an out-of-band markup convention that this file neither defines nor parses.
- Default emotional values are asymmetric (joy/trust high, fear low) and do not sum to 1, so they are independent intensities rather than a distribution — no normalization exists here.

```
MotherResponse(...)
  |
  +-- content ............... REQUIRED  -> ValidationError if absent
  +-- emotional_context ..... default dict  (may also be set to None)
  +-- reward_score .......... default 0.7   <-- silent positive reward
  +-- success_metric ........ default 0.0
  +-- complexity_rating ..... default 0.0
  +-- self_critique_score ... default 0.0
  +-- cognitive_labels ...... default []    (may also be set to None)
```

## Imports

- Third-party: `pydantic.BaseModel` (schemas.py:1).
- Standard library: `typing.Optional` (schemas.py:2).
- Project-internal: none.

## Defects and gaps

- schemas.py:13 — `reward_score` defaults to `0.7`, a hardcoded positive magnitude, while the other three scalars default to `0.0` (schemas.py:14-16). A payload that omits the field is indistinguishable from one that asserted a strongly positive reward, and nothing in this file justifies the constant.
- schemas.py:7, 17 — `Optional[dict]` / `Optional[list]` explicitly admit `None` while defaulting to a non-`None` value. `emotional_context=None` validates cleanly, so a validated instance is not guaranteed to hold a dict or a list. The `Optional` annotation contradicts the intent expressed by the defaults; `dict` / `list` (non-Optional) is what the defaults imply.
- schemas.py:7-12, 17 — mutable literal defaults. Safe under Pydantic v2 (defaults are deep-copied per instance), but only because of that framework guarantee; the pattern is a trap if the model is ever converted to a dataclass or plain class.
- schemas.py:7, 17 — bare `dict` / `list` annotations perform no key, value, or element validation. `emotional_context={"nonsense": "text"}` and `cognitive_labels=[{}, 3]` both validate cleanly. The four default emotion keys (schemas.py:8-11) are therefore an implicit vocabulary the type does not enforce.
- schemas.py:13-16 — no `Field(ge=..., le=...)` constraints on any scalar despite the 0.0-1.0 semantics the example implies. Out-of-range values propagate silently.
- schemas.py:19 — `model_config` as a plain dict is Pydantic v2 syntax. Under Pydantic v1 this class attribute would not configure anything (v1 requires an inner `Config` class), so the file is v2-only in effect. No version pin is visible in this file.
- No `Field` descriptions, aliases, or `extra` policy are set, so schema consumers get bare type names and unknown keys follow whatever the global/default `extra` behaviour is (`ignore` by default in v2).

## Notes

- The docstring at schemas.py:5 calls this a schema for "mother-child interactions", but every field describes the mother/teacher side only (content plus ratings of that content). There is no child-side field, no turn id, no timestamp, and no correlation key — the model cannot represent an interaction pair on its own.
- The four scalars (`reward_score`, `success_metric`, `complexity_rating`, `self_critique_score`) are structurally identical unbounded floats; their distinct meanings exist only in their names.
- Because only `content` is required, a payload carrying nothing but a content string still constructs a fully populated object of defaults. Validation cannot distinguish a complete payload from a bare one — every metric field will be present either way.
