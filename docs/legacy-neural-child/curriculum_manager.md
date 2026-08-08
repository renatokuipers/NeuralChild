# curriculum_manager.py

Defines an 18-level developmental-stage ladder and a stateful controller that scores incoming metrics against the current stage's target profile, then advances or regresses the stage. Roughly 515 of the 656 lines are a hand-written static table of per-stage numeric thresholds and symbolic behaviour vocabularies. No I/O, no persistence, no logging — purely in-memory state on a single object.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `StageMetrics` | dataclass | 10 numeric/duration targets for a stage; `to_tensor()` flattens to a 10-element float32 tensor (curriculum_manager.py:9) |
| `StageMetrics.to_tensor` | method | Emits `[curiosity, attention_hours, emo_stability, social, abstraction, language, memory, problem_solving, motor, self_awareness]`; index 1 is `total_seconds()/3600` (curriculum_manager.py:23) |
| `StageBehaviors` | dataclass | Two `Set[str]` (allowed_actions, required_skills) + four `List[str]` vocabularies; pure data, no methods (curriculum_manager.py:37) |
| `DevelopmentalStage` | Enum | 18 members, contiguous int values 0 (NEWBORN) → 17 (MATURE_ADULT) (curriculum_manager.py:47) |
| `DevelopmentalSystem` | class | Holds `current_stage`, `stage_history`, `regression_count`, `last_transition` (curriculum_manager.py:67) |
| `DevelopmentalSystem.evaluate_current_stage` | method | Cosine similarity between a caller-supplied metrics dict and the current stage's target tensor; returns a float (curriculum_manager.py:597) |
| `DevelopmentalSystem.check_regression` | method | True when `performance < mean(last 5 history) * 0.7`; increments `regression_count` as a side effect (curriculum_manager.py:618) |
| `DevelopmentalSystem.update_stage` | method | Scores, records, then may regress or advance; returns a human-readable string or `None` (curriculum_manager.py:626) |
| `DevelopmentalSystem.get_stage_requirements` | method | Current stage's metrics `__dict__` plus behaviours with sets converted to lists (curriculum_manager.py:644) |

## Key behaviour

- `_initialize_stages` (curriculum_manager.py:75) populates all 18 enum members; no member is missing, so no `KeyError` is reachable from `self.stage_definitions[self.current_stage]`.
- Target metric values are non-decreasing across stages except `emotional_stability`, which dips at EARLY_ADOLESCENCE 0.85 / MIDDLE_ADOLESCENCE 0.83 after LATE_ELEMENTARY 0.89 (curriculum_manager.py:396, 420).
- `attention_span` targets span `timedelta(minutes=5)` → `timedelta(hours=8)`, i.e. tensor index 1 ranges 0.083 → 8.0 while all nine other components stay in [0.1, 0.99].
- Both tensors are shaped (10,), unsqueezed to (1,10); `cosine_similarity` returns shape (1,) and `.item()` unwraps it (curriculum_manager.py:612).
- Advancement gate: `performance > 0.85` **and** `(now - last_transition).days >= 30` (curriculum_manager.py:636) **and** `current_stage.value < len(DevelopmentalStage) - 1`, i.e. `< 17` (curriculum_manager.py:637).
- Regression gate: `check_regression` returns False outright until `stage_history` holds at least 5 entries (curriculum_manager.py:619). Three *cumulative* hits (`regression_count >= 3`) drop one stage and reset the counter; fewer hits return only a warning string (curriculum_manager.py:630). The counter is not reset by a non-regressing update, so the three hits need not be consecutive or recent.

```
update_stage(metrics)
  ├─ performance = cosine_sim(metrics, target)     # 627 calls 597
  ├─ stage_history.append(performance)             # 628  (before the check)
  ├─ check_regression? ──yes──┬─ count>=3 → stage-1, count=0, return "Regression detected…"
  │                           └─ else     → return "Warning: Potential regression detected"
  └─ no ─ perf>0.85 AND days_since_transition>=30 AND value<17
                └─ stage+1, last_transition=now, count=0, return "Advanced to …"
          otherwise → None
```

## Imports

Third-party: `torch`, `numpy`. Standard library: `enum.Enum`, `dataclasses.dataclass`, `typing` (Dict, List, Optional, Set), `datetime.datetime`, `datetime.timedelta`. No project-internal imports — this file is self-contained.

## Defects and gaps

- **Wall-clock gate makes progression untestable.** `last_transition` is set to `datetime.now()` in `__init__` (curriculum_manager.py:73) and the advance branch requires `.days >= 30` (curriculum_manager.py:636). A freshly constructed system cannot leave NEWBORN for 30 real days; reaching MATURE_ADULT needs ≥510 real days. Nothing in the file injects or overrides the clock.
- **Cosine similarity makes the 0.85 threshold measure direction, not attainment.** The target vector is entirely non-negative and cosine is scale-invariant, so a metrics vector proportional to the target scores 1.0 at any magnitude — uniformly tiny inputs score identically to uniformly large ones. Nothing compares absolute levels against the stage thresholds.
- **No validation of caller metrics.** The incoming values are read with dict-get defaults and placed straight into the tensor (curriculum_manager.py:600-611); no range, sign, or key check exists, so out-of-range or negative inputs propagate into the score.
- **History is contaminated before comparison.** `update_stage` appends the new score (curriculum_manager.py:628) and only then calls `check_regression` (curriculum_manager.py:629), so the current sample sits inside the 5-sample mean it is compared against, making the drop trigger stricter than the stated 30% and self-damping.
- **Regression at the lowest stage reports a move that did not happen.** The previous-stage computation clamps at zero (curriculum_manager.py:631), so at NEWBORN the stage is unchanged, yet the branch still resets the counter and returns the regression-detected message naming NEWBORN (curriculum_manager.py:633-634) — a state-change report for a no-op.
- **Silent key-name mismatch.** `evaluate_current_stage` reads `'curiosity'`, `'abstraction'`, `'language'`, `'memory'` (curriculum_manager.py:601-607) while the corresponding `StageMetrics` fields are `curiosity_threshold`, `abstraction_level`, `language_complexity`, `memory_retention`. A caller passing field-named keys gets `0.0` defaults with no warning.
- **Unit mismatch on index 1.** The target's `attention_span` is in hours (up to 8.0) but `metrics.get('attention_span', 0.0)` is an unconverted caller value with no documented unit (curriculum_manager.py:602).
- **Regression does not reset the transition clock.** The regress branch (curriculum_manager.py:631-634) leaves `last_transition` untouched, so the very next `update_stage` with `performance > 0.85` can re-advance immediately, allowing stage oscillation.
- **`Dict[str, any]`** at curriculum_manager.py:644 uses the builtin `any` function, not `typing.Any`. It does not raise at runtime but is meaningless as an annotation.
- **`get_stage_requirements` exposes the live stage table.** `metrics.__dict__` (curriculum_manager.py:647) is the shared `StageMetrics` instance dict, not a copy, so a caller mutating the returned `'metrics'` mapping rewrites the stage definition for every later evaluation. It also carries `attention_span` as a `timedelta`, leaving the result non-JSON-serializable despite the set-to-list flattening done alongside it.
- **`stage_history` grows unbounded** and is never cleared on a stage transition (curriculum_manager.py:628), so post-transition scores are compared against scores computed against a different stage's target.
- **`StageBehaviors` is inert.** `allowed_actions` and `required_skills` are never checked, enforced, or compared anywhere in this file — they are only copied out by `get_stage_requirements`.
- **No `__init__` argument for starting stage or loading saved state**; the class always begins at NEWBORN with empty history.

## Notes

- Methods with no caller inside this file: `update_stage` (curriculum_manager.py:626) and `get_stage_requirements` (curriculum_manager.py:644). Whether external modules call them is unverifiable from this file alone.
- `check_regression` mutates `regression_count` while also returning a boolean — calling it for its return value alone has a side effect.
- Uses PEP 585 builtin generics (`tuple[...]` at curriculum_manager.py:75), so Python 3.9+ is required.
- `numpy` is pulled in for a single mean over a short Python list (curriculum_manager.py:620), and `torch` only for two tensor constructions and one cosine-similarity call — both are heavy dependencies for the arithmetic actually performed.
- Comments in the stage table carry emoji and human-age annotations ("0-3 months", "21+ years"); those ages have no representation in code — nothing maps simulated or real time to a stage.
