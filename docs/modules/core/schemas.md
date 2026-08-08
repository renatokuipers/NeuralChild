# schemas.py

Pydantic data-model module defining the shared vocabulary passed between neural networks in NeuralChild: messages, network state, memories, outputs, beliefs, and needs. Also defines the `DevelopmentalStage` enum that most models carry as a tag. Contains no I/O, no networking, and no neural computation — only declarative models plus a handful of small mutator/formatter methods.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `DevelopmentalStage` | Enum | Five members INFANT=1, TODDLER=2, CHILD=3, ADOLESCENT=4, MATURE=5. Plain `Enum`, not `IntEnum`. schemas.py:13 |
| `NetworkMessage` | model | sender/receiver/content plus message_type ("standard"), timestamp, priority (1.0), stage. schemas.py:21 |
| `NetworkMessage.to_dict` | method | Manual dict with `timestamp.isoformat()` and stage rendered as `.name`. schemas.py:35 |
| `NetworkState` | model | name, active (True), last_update, free-form `parameters`, `developmental_weights` keyed by enum. schemas.py:47 |
| `NetworkState.to_dict` | method | Same manual serialization; converts enum dict keys to their `.name`. schemas.py:59 |
| `Memory` | model | id, content, creation/last-access times, strength (1.0), emotional_valence (0.0), tags, stage. schemas.py:69 |
| `Memory.access` | method | Sets last_access_time=now, strength = min(5.0, strength+0.1). schemas.py:84 |
| `Memory.decay` | method | strength = max(0.0, strength − amount), amount defaults 0.01. schemas.py:90 |
| `Memory.is_forgotten` | method | True when strength < 0.1. schemas.py:94 |
| `VectorOutput` | model | source, `data: List[float]`, timestamp, stage. No dimension declared. schemas.py:98 |
| `TextOutput` | model | source, text, confidence (1.0), timestamp, stage. schemas.py:109 |
| `Belief` | model | subject/predicate/object triple, confidence (0.5), timestamps, supporting_memories (memory id strings), stage. schemas.py:121 |
| `Belief.update_confidence` | method | confidence = (confidence + new_evidence) / 2; refreshes last_update_time. schemas.py:136 |
| `Belief.to_natural_language` | method | Prefixes the triple with a confidence phrase; returns "prefix subject predicate object". schemas.py:142 |
| `Need` | model | name, intensity and satisfaction_level both `Field(ge=0.0, le=1.0, default=0.5)`, last_update. schemas.py:154 |
| `Need.update_intensity` | method | intensity clamped into [0,1]; refreshes last_update. schemas.py:165 |
| `Need.satisfy` | method | satisfaction += amount (capped at 1.0); intensity −= amount*0.5 (floored at 0.0). schemas.py:170 |

## Key behaviour

- Five of seven models carry a `developmental_stage` field defaulting to `INFANT`. `NetworkState` and `Need` do not. `Need` is the only model with validated numeric bounds.

```
DevelopmentalStage ──as field──> NetworkMessage, Memory,
                                 VectorOutput, TextOutput, Belief
                   ──as key────> NetworkState.developmental_weights
                   ──absent────> Need
```

- Memory strength dynamics are entirely caller-driven: nothing in this file schedules `decay`, so the "decays over time" claim in the field comment (schemas.py:79) depends on an external loop. The two mutators together hold strength inside [0.0, 5.0], and forgetting starts below 0.1 — so a memory needs ~90 decay steps at the default 0.01 to fall from its default 1.0 to the forget threshold.
- `Belief.to_natural_language` thresholds: >0.8 "I'm sure that ", >0.5 "I think that ", otherwise "I'm not sure, but maybe ". Default confidence 0.5 lands in the *else* branch, so every freshly created belief verbalizes as uncertain.
- `update_confidence` is a plain arithmetic mean of old confidence and new evidence, not a Bayesian update, despite the comment at schemas.py:138. Each call halves the weight of all prior evidence, so the value is strongly recency-biased and asymptotic — it never reaches 0 or 1.
- Only `NetworkMessage` and `NetworkState` define `to_dict`; `Memory`, `Belief`, `VectorOutput`, `TextOutput`, `Need` have none, so the module offers no uniform serialization entry point.
- No `model_config`, no validators, no `__all__`. Field constraints on `Need` are therefore construction-time only — the mutator methods assign attributes directly and (in pydantic v2 without `validate_assignment`) bypass the ge/le checks.

## Imports

- Third-party: `pydantic` (`BaseModel`, `Field`), `torch`.
- Stdlib: `typing` (`Dict`, `Any`, `List`, `Optional`, `Union`), `datetime.datetime`, `enum` (`Enum`, `auto`).
- Project-internal: none.

## Defects and gaps

- schemas.py:11 — `import torch` is never referenced anywhere in the file. Importing this module therefore pays the full torch import cost, and a broken torch install fails the import for zero benefit.
- schemas.py:8 — `Optional` and `Union` imported, never used. schemas.py:10 — `auto` imported, never used (all enum members are explicit integers).
- schemas.py:172-174 — `Need.satisfy` clamps satisfaction only at the top (`min(1.0, …)`) and intensity only at the bottom (`max(0.0, …)`). A negative `amount` drives satisfaction_level below 0.0 and intensity above 1.0, silently violating the declared `ge=0.0, le=1.0` field constraints since assignment is not validated.
- schemas.py:130/136 — `Belief.confidence` has no bound declared and `update_confidence` does not clamp `new_evidence`; an out-of-range evidence value propagates straight into the confidence used by `to_natural_language`.
- schemas.py:144 — `confidence_text = ""` is dead; all three branches of the following if/elif/else reassign it unconditionally.
- schemas.py:13 — `DevelopmentalStage` subclasses plain `Enum`, so ordering comparisons (`stage >= DevelopmentalStage.CHILD`) raise TypeError even though the members are numbered 1–5 as if intended for progression checks.
- schemas.py:88/90/96 — strength ceiling 5.0 and increment 0.1 (line 88), decay default 0.01 (line 90 signature), and forget threshold 0.1 (line 96) are hardcoded with no configuration hook; changing the decay cadence silently changes memory lifetime.
- schemas.py:47-57 — `NetworkState` is the only non-`Need` model without a `developmental_stage` field, so a network's own stage cannot be read off its state; only the per-stage weight mapping is present.
- Both `to_dict` methods (schemas.py:35, schemas.py:59) hand-build dicts that duplicate what pydantic's own JSON-mode serialization already produces, and they will silently drop any field added to the model later.
- schemas.py:57 — `developmental_weights` uses enum objects as dict keys. `to_dict` handles this, but any direct `json.dumps` of the raw field (or a pydantic v1 `.dict()`) yields non-string keys / enum objects.
- schemas.py:105 — `VectorOutput.data` is an unconstrained `List[float]` with no declared dimensionality, so mismatched vector sizes between producing and consuming networks cannot be caught here.
- Every method in the file (both `to_dict`s, `access`, `decay`, `is_forgotten`, `update_confidence`, `to_natural_language`, `update_intensity`, `satisfy`) is defined but never referenced within this file. Whether external callers use them is unverifiable from this file alone.

## Notes

- The module uses no pydantic-v2-only API (`Field(default_factory=…)` and `ge`/`le` exist in both), so it parses under v1 and v2 — but the assignment-validation behaviour of the mutator methods differs between them, which is worth pinning before relying on the `Need` bounds.
- `timestamp` / `creation_time` fields all use `datetime.now()` (naive local time), not `datetime.now(timezone.utc)`; ordering across DST boundaries or machines is unreliable.
