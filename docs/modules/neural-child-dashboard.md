# neural-child-dashboard.py

Single-file Dash/Plotly web dashboard for the NeuralChild simulation. It owns module-level global instances of `Mind`, `MotherLLM` and four neural networks, runs the simulation in a background daemon thread, and mirrors state into a Pydantic store read by 15 Dash callbacks (11 of them driven by a 1 s interval, 4 by buttons). Also contains checkpoint save/load to disk and a full inline CSS theme.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `SimulatedInput` | pydantic model | visual/auditory/language/source/type; sensory vectors coerced to exactly 64 floats; at least one of visual/auditory/language required |
| `TrainingConfig` | pydantic model | save_interval_steps>=1, save_directory, checkpoint_count>=1, step_interval 0.01–10.0, auto_backup |
| `DashboardData` | pydantic model | mutable global store: mind_state, network_states, three history lists, step_count, is_running, errors, training_config |
| `DateTimeEncoder` | class (JSONEncoder) | serializes `datetime` as ISO-8601 |
| `bootstrap_mind(mind)` | function | feeds 5 hardcoded "maternal/environment" experiences into `mind.process_input` |
| `generate_environmental_input()` | function | random choice of visual / auditory / language / combined stimulus dict |
| `process_mother_response(text)` | function | wraps mother text plus a 64-float auditory vector scaled by `min(1.0, len(text)/100)` |
| `run_simulation()` | function | thread target; loops while the `simulation_active` global stays true |
| `save_models(checkpoint_name=None)` | function | returns `(bool, path_or_error)`; writes checkpoint dir + dashboard_data.json, prunes old checkpoints |
| `load_models(checkpoint_dir)` | function | returns `(bool, message)`; restores mind state and dashboard JSON |
| `dashboard_data`, `mind`, `mother`, `networks` | module globals | shared mutable state, constructed at import time (lines 97-107) |
| `app` | Dash app | DARKLY theme, `suppress_callback_exceptions=True`, custom `index_string`; served by `app.run_server(debug=True)` at line 1580 |

## Key behaviour

- Import-time side effects: `Mind()`, `MotherLLM()`, four networks are instantiated and registered at lines 98-110 — before any callback or `__main__` guard.
- Simulation loop (neural-child-dashboard.py:239-341), per iteration: `mind.step()` → every 3rd step inject environmental input → every 10th step `mother.observe_and_respond(mind)` and re-inject its text as sensory input → snapshot observable/mind state → collect `generate_text_output()` from each of the 4 networks → append to development/emotion/memory histories → truncate each history to the last 1000 entries → conditional `save_models()` → `time.sleep(training_config.step_interval)` (default 0.1s).
- Mood is mapped from [-1,1] to a 0-100 progress bar via `(mood + 1) * 50` (line 1038); consciousness/energy/confidence are multiplied by 100 and truncated to int.
- UI refresh is a fixed 1000 ms `dcc.Interval`. One callback (line 990) serializes a subset of the store to a hidden div; four callbacks fan out from that div (status, mind state, network outputs, error log). The three graph callbacks plus milestones, beliefs and needs bypass the div and read the globals directly.
- Graphs render only the last 200 history rows. Development graph draws a dashed vline at every developmental-stage change detected by scanning the frame.
- Needs colouring thresholds are hardcoded at 0.7 / 0.9 (lines 1452, 1459).

```
[start-button] -> thread(run_simulation) -+-> mind.step()  -----> dashboard_data (globals)
                                          |                              |
[stop-button]  -> simulation_active=False +                              v
                                                    dcc.Interval(1s) -> hidden div -> 4 callbacks
                                                                     \-> 6 direct-read callbacks
```

## Imports

Third-party: dash (`dcc`, `html`, `Input`, `Output`, `State`, `callback`, `ctx`), dash_bootstrap_components, plotly.graph_objects, plotly.express, pandas, numpy, pydantic. Stdlib: datetime, os, sys, threading, time, json, random, typing, logging, shutil (imported inside `save_models`).

Project-internal (not read): `config` (`load_config`, `Config`, `get_config`), `mind.mind_core.Mind`, `mother.mother_llm.MotherLLM`, `mind.networks.{consciousness,emotions,perception,thoughts}`, `mind.schemas.EmotionType`, `core.schemas.DevelopmentalStage`.

## Defects and gaps

- neural-child-dashboard.py:19 — comment says "Add project root to path" but `dirname(dirname(abspath(__file__)))` appends the parent of the directory holding this file, i.e. one level *above* the file's own directory. Whether lines 22-30 resolve is decided by the layout outside this file — unverifiable here.
- neural-child-dashboard.py:394-429 — `load_models` has no caller in this file; there is no load button or callback. Dead within this module.
- neural-child-dashboard.py:422 — `dashboard_data.__dict__.update(data_dict)` writes straight into the Pydantic instance dict, bypassing all validation despite `validate_assignment = True`.
- neural-child-dashboard.py:409-418 — `load_models` converts history `timestamp` strings back to `datetime`, but no code in this file ever reads `timestamp` as a datetime (the graphs plot `step` only). Dead work that also re-introduces objects only `DateTimeEncoder` can re-serialize.
- neural-child-dashboard.py:1330-1331 — line 1330 clamps `next_stage_value` to at most 5, so the `if next_stage_value <= 5 else None` on 1331 can never yield `None`. The "Reached maximum developmental stage" fallback at 1367 is then reachable only if the enum member itself is falsy. `5` is hardcoded as the maximum stage, and at the top stage `next_stage` equals `current_stage`, so the UI shows "Progress toward <current stage>".
- neural-child-dashboard.py:1356 — `int((current_value / threshold) * 100)` divides by an unchecked value taken from `mind.development_thresholds`; a zero threshold raises inside the once-per-second callback.
- neural-child-dashboard.py:236-237 — `bootstrap_mind(mind)` runs *outside* the `try` and outside the `while`. If it raises, the thread dies while `dashboard_data.is_running` is still `True` (set at 232), so the UI reports "Running" indefinitely. Normal loop exit also never resets `is_running`; only the stop callback does.
- neural-child-dashboard.py:1452 — the `color` local computed from the 0.7/0.9 thresholds is never referenced; the progress bars at 1459 recompute the same thresholds with Bootstrap colour names instead. Dead assignment.
- neural-child-dashboard.py:335-341 — blanket `except Exception` logs to a list and sleeps 1s inside a `while` loop; a permanently failing `mind.step()` spins forever with no back-off or abort.
- neural-child-dashboard.py:329 — `save_models()` return value is discarded in the loop, so an auto-backup failure only surfaces as an appended error string, yet `last_saved_step` is still advanced on line 330.
- neural-child-dashboard.py:343 — the `checkpoint_name` parameter is never supplied by either caller (lines 329, 1532); the pruning filter on line 374 only matches `checkpoint_`-prefixed dirs, so named checkpoints would never be pruned.
- neural-child-dashboard.py:432 — `DateTimeEncoder` is defined *after* its use site at line 371 (resolves at call time, but the ordering is fragile if `save_models` is ever called during module import).
- neural-child-dashboard.py:991 — this `json.dumps` does not pass `cls=DateTimeEncoder` (unlike line 371), and it serializes whatever `network.state.parameters` holds (stored at line 283); if that value is not JSON-encodable the 1-second callback raises. The actual type is unverifiable from this file.
- neural-child-dashboard.py:46 — validator signature `(cls, v, values, **kwargs)`: under Pydantic v2 the second positional argument is a `ValidationInfo`, not a values dict. Harmless only because `values` is unused.
- neural-child-dashboard.py:1564-1565 — `config.mind.step_interval` is assigned but the loop reads `dashboard_data.training_config.step_interval` (line 333); the effect of that assignment is unverifiable from this file alone.
- Unused imports: `plotly.express as px` (5), `numpy as np` (7), `timedelta` (8), `callback`/`ctx` (2), `Union`/`Set` (15), `load_config` and `Config` (22), `EmotionType` (29). Module-level `import random` (14) is shadowed by redundant local re-imports at lines 125, 169, 212.
- neural-child-dashboard.py:109 — loop variable `name` is unused in the network registration loop.
- Nested `class Config:` blocks (63, 76, 93) are the Pydantic v1 config style; under v2 these emit deprecation warnings.
- No lock guards `dashboard_data`, `mind`, or `networks` between the simulation thread and the Dash callback threads.

## Notes

- `app.run_server(...)` (line 1580) is the Dash 2.x spelling; `debug=True` enables the reloader, which re-executes the module and would construct a second `Mind` and second network set in the child process.
- `update_error_log` (line 1478) fires every second on the same `error-log` output that the save and apply-config callbacks write to, so their success messages are overwritten within ~1s.
- Both start and stop callbacks replace `interval-container` with a brand-new `dcc.Interval`, resetting `n_intervals` to 0 on every click.
- With `validate_assignment = True`, each history truncation (lines 296, 310, 322) re-validates up to 1000 dict entries every simulation step.
