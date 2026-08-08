# Visualization

> **STATUS — CURRENT: the dashboard cannot start.**
> `neural-child-dashboard.py:1580` (the last line) calls `app.run_server(debug=True)`.
> The installed Dash is **4.0.0**, which removed `run_server`; even *accessing* the attribute raises
> `dash.exceptions.ObsoleteAttributeException: app.run_server has been replaced by app.run`.
> **One-line fix:** `app.run(debug=True)`.
>
> The rest of the module is healthy — imported headlessly it loads, builds its layout and registers
> **15 callbacks** without error. `run_server` is the only thing between it and a running server.

The dashboard is the project's only visualization surface. `visualization/` is an empty package
(see [§8](#8-the-visualization-package)); all output is intended to land here.

---

## 1. Launch paths

| Path | What it does | Verdict |
| --- | --- | --- |
| `run_dashboard.py` | probes 6 packages, pip-installs missing ones, spawns the dashboard as a subprocess | CURRENT: reaches launch, then the child dies on `run_server` |
| direct invocation | runs `neural-child-dashboard.py` in the current interpreter | CURRENT: same failure, but the traceback is visible |

**Correct command (once `run_server` → `run` is fixed):**

```
F:/envs/5070_11/python.exe E:/python-projects/NeuralChild/neural-child-dashboard.py
```

`run_dashboard.py` (61 lines) has two behaviours worth knowing before you use it:

- **It pip-installs at runtime.** `check_and_install_packages()` (run_dashboard.py:21) probes six
  packages with `importlib.util.find_spec`, then `pip install`s the missing ones into
  `sys.executable` — no virtualenv check, no `--user`, no version pins, no confirmation. In
  `F:/envs/5070_11` all six already import, so it installs nothing.
- **It probes a wrongly-named file first.** run_dashboard.py:43 looks for
  `neural_child_dashboard.py` (underscores), which **does not exist** here, then falls through to
  the real `neural-child-dashboard.py` at run_dashboard.py:50. The fallback works, but the
  not-found message at run_dashboard.py:55 names only the hyphenated candidate.

It also discards the child's exit code — `main()` returns 0 whether the dashboard ran or crashed
instantly, so today it exits **successfully** while the dashboard is dying.

Installed in `F:/envs/5070_11`: dash 4.0.0, dash-bootstrap-components 2.0.4, plotly 5.24.1,
pandas 3.0.3, pydantic 2.13.4.

---

## 2. Architecture

CURRENT. A single 1580-line file. State lives in module-level globals constructed at **import
time** (neural-child-dashboard.py:97-110) — before any callback runs and before the `__main__`
guard. A background daemon thread mutates them; Dash callbacks read them on a 1 s poll.

```
 import time (97-110)   dashboard_data=DashboardData()  mind=Mind()  mother=MotherLLM()
                        networks={consciousness, emotions, perception, thoughts} -> mind

 [Start] --> Thread(run_simulation, daemon=True)
                   |  mind.step(); mind.process_input(...); mother.observe_and_respond(mind)
                   v
            dashboard_data (mind_state, network_states, 3 history lists, errors)
                   |  NO LOCK — sim thread and callback threads race
                   v
   dcc.Interval(1000 ms) --+--> update_intermediate_value() --> hidden div
                           |         +--> status / mind-state / network-outputs / error-log  (4)
                           +------------> 6 callbacks reading the globals directly
                                          (3 graphs, milestones, beliefs, needs)
```

15 callbacks: 11 driven by the interval (directly or via the hidden div), 4 by buttons
(start, stop, save, apply-config).

**No `LanguageNetwork` is registered** — only the four above. This matters in [§7](#7-panels-that-render-empty-or-frozen).
`LanguageNetwork` is never instantiated anywhere in the repo
(see [docs/modules/mind/language.md](modules/mind/language.md)).

---

## 3. Simulation loop

CURRENT — `run_simulation()`, neural-child-dashboard.py:227-341.

| Phase | Behaviour |
| --- | --- |
| Bootstrap | if `step_count == 0`, feed 5 hardcoded maternal/environment inputs. Runs **outside** the `try` and outside the `while` |
| Every step | `mind.step()`, then snapshot observable + mind state, then collect `generate_text_output()` from all 4 networks |
| Every 3rd step | inject `generate_environmental_input()` — random visual / auditory / language / combined stimulus |
| Every 10th step | `mother.observe_and_respond(mind)`; the reply is re-injected as sensory input with a 64-float auditory vector scaled by `min(1.0, len(text)/100)` |
| History | append one row each to development / emotion / memory history, then truncate each to the last **1000** entries |
| Autosave | when `auto_backup` and `step_count % save_interval_steps == 0` (default **100**) |
| Pace | `time.sleep(step_interval)` — default **0.1 s** |

The mother leg is the one part of this loop verified working end to end against LM Studio
(see [docs/modules/mother/mother_llm.md](modules/mother/mother_llm.md)).

---

## 4. Panels and callbacks

| Panel | Callback (line) | Data source | Shows |
| --- | --- | --- | --- |
| Status / step count | `update_status_indicators` (1007) | hidden div | running flag, step counter |
| Mind State Overview | `update_mind_state` (1029) | hidden div | stage, consciousness, energy, mood, focus, vocalization |
| Mother's Response | `update_mind_state` (1029) | hidden div | `last_mother_response` |
| Network cards ×4 | `update_network_outputs` (1066) | hidden div | per-network text + confidence bar |
| Development graph | `update_development_graph` (1106) | `development_history` | consciousness + energy vs step; dashed vline per stage change |
| Emotions graph | `update_emotions_graph` (1188) | `emotion_history` | intensity per emotion vs step |
| Memory graph | `update_memory_graph` (1265) | `memory_history` | short-term vs long-term counts |
| Milestones | `update_milestones` (1326) | `mind.developmental_milestones` | progress bars toward next stage + vocabulary list |
| Beliefs | `update_beliefs` (1388) | `mind.belief_network.beliefs` | last 10 beliefs as a table |
| Needs | `update_needs` (1440) | `mind.need_system.needs` | intensity + satisfaction bars per need |
| System Log | `update_error_log` (1478) | hidden div | last 10 error strings |

Graphs render only the **last 200** history rows. Mood is mapped from `[-1,1]` to `0-100` via
`(mood + 1) * 50`.

---

## 5. Persistence

CURRENT — `save_models()` (343) and `load_models()` (394).

```
saved_models/                                   <- training_config.save_directory
  checkpoint_<YYYYmmdd_HHMMSS>_step_<n>/
    {consciousness,emotions,perception,thoughts}.pytorch   <- mind.save_state(), format="pytorch"
    mind_state.json                             <- stage, milestones, language_ability, metrics
    memories.json                               <- TRUNCATED, see below
    beliefs.json   needs.json                   <- NEVER WRITTEN, see below
    dashboard_data.json                         <- store, histories sliced to last 100
```

- **Rotation:** dirs matching prefix `checkpoint_` are sorted by `getctime`; oldest removed beyond
  `checkpoint_count` (default 5). `checkpoint_name` is never passed by either caller, and a named
  checkpoint would not match the prefix filter, so it would never be pruned.
- **`DateTimeEncoder`** (432) serializes `datetime` as ISO-8601, but is defined *after* its use site
  at line 371. The hidden-div `json.dumps` at line 991 does **not** pass it.
- **Loading is dead in the UI.** `load_models()` has no button and no callback, and writes via
  `dashboard_data.__dict__.update(...)`, bypassing pydantic validation entirely.

**Checkpoints are written broken.** Measured by calling `mind.save_state()` directly: it returns
`False`. It writes the four `.pytorch` files and `mind_state.json`, then raises `Object of type
DevelopmentalStage is not JSON serializable` while dumping `memories.json` (`memory.dict()` leaves
the enum un-serialized). So `memories.json` is **truncated mid-write** and `beliefs.json` /
`needs.json` are **never created**; a following `load_state()` fails parsing it and returns `False`.

The dashboard does not notice — `save_models()` discards the `mind.save_state()` return value
(line 360), writes `dashboard_data.json`, and returns `True`. The Save button then reports
**"Models successfully saved to: …"** over a partial, unloadable checkpoint.

---

## 6. Known issues

| # | Issue | Location |
| --- | --- | --- |
| 1 | `app.run_server()` — fatal on Dash 4.0.0; blocks all startup. `debug=True` would also enable the reloader, building a **second** `Mind` and network set | 1580 |
| 2 | Runtime `pip install` into the active interpreter, unpinned, unconfirmed | run_dashboard.py:32 |
| 3 | Runner probes a nonexistent `neural_child_dashboard.py` first, names the wrong path on failure, and discards the child exit code (always exits 0) | run_dashboard.py:43,55,59 |
| 4 | Blanket `except Exception` in the sim loop appends a string and sleeps 1 s — a permanently failing `mind.step()` spins forever, no back-off, no abort | 335-341 |
| 5 | `bootstrap_mind()` runs outside the `try`; if it raises, the thread dies while `is_running` stays `True` and the UI reports "Running" forever | 236-237 |
| 6 | Save reports success over a broken checkpoint (§5) | 360 |
| 7 | No lock guards `dashboard_data` / `mind` / `networks` across the sim and callback threads | — |
| 8 | `update_error_log` fires every second on the same output the save and apply-config callbacks write to — success messages vanish within ~1 s | 1478 |
| 9 | Milestone percentage divides by an unchecked `development_thresholds` value; a zero threshold would raise inside a 1 Hz callback | 1356 |
| 10 | Start/stop callbacks replace `interval-container` with a fresh `dcc.Interval`, resetting `n_intervals` to 0 on every click | 1508, 1522 |
| 11 | `sys.path.append(dirname(dirname(abspath(__file__))))` appends the *parent* of the repo; imports work only because the script's own directory is `sys.path[0]` | 19 |
| 12 | With `validate_assignment = True`, each history truncation re-validates up to 1000 dicts every step | 296, 310, 322 |
| 13 | "Reached maximum developmental stage" is unreachable — `next_stage_value` is clamped to ≤5 before the `is None` test, so at MATURE the UI reads "Progress toward MATURE" | 1330-1331, 1367 |

---

## 7. Panels that render empty or frozen

The messaging fabric is **partially** live, and which half is dead determines which panels move.

```
 CONSCIOUSNESS  autonomous_step() -queues-> pending_messages, drained every step by
                Mind._retrieve_network_messages (1102-1131) -> _process_mind_message
                -> "consciousness" branch (1204)   LIVE: consciousness_level, current_focus

 PERCEPTION     autonomous_step() -queues-> receiver="thoughts" -> message_bus.publish (1126)
                DEAD END: "mind" is the only subscriber (877)

 EMOTIONS       _update_emotional_state() -queues emotion msg-+  both sit INSIDE process_message(),
 THOUGHTS       process_message()         -queues belief  msg-+  which NOTHING ever calls, so the
                "emotion" (1139) and "belief" (1162) branches never fire
```

So `_process_mind_message` **is** on the hot path and **does** receive messages — but never an
`emotion` or `belief` one. `Mind.state.emotional_state` therefore keeps the four seed values set in
`Mind.__init__` (mind_core.py:859-864) forever, and `emotions_experienced` is never incremented.

Measured by driving the same globals headlessly — replicating `run_simulation()`'s loop (bootstrap
+ 120 steps, environmental input every 3rd step) — since the dashboard itself cannot start:

| Panel | State | Why |
| --- | --- | --- |
| **Beliefs System** | **permanently empty** — "No beliefs have been formed yet." | beliefs are only created during memory consolidation and via the dead `belief` branch |
| **Emotions graph** | **frozen, not empty** — 4 flat lines at exactly 0.30 (joy, trust, fear, surprise) | the seed values; all exceed the 0.2 `recent_emotions` threshold, so rows are logged every step but never change. Before Start it reads "No emotion data available" |
| **Milestones → Emotions Experienced** | **frozen at 0/3** | only the dead `emotion` branch increments it |
| **Milestones → Recent Vocabulary** | **never rendered** | `process_input` gates its vocabulary update on `"language" in self.networks`; only 4 networks are registered, so `vocabulary_learned` stays empty and the section is skipped |
| **Memory graph → long-term** | **flat at 0** | consolidation needs emotional valence > 0.6, strength > 1.5, or age > 300 s; short-term is capped at `3 + stage.value*2` = **5** at INFANT, so rows are evicted long before they age in |
| **Memory graph → short-term** | sawtooths 0→5 | live, but bounded by that cap |
| **Developmental Stage** | **frozen at INFANT** | the INFANT→TODDLER gate needs 3 distinct emotions experienced; that counter cannot leave 0 |

Panels that **do** carry live data: the four network cards (text + confidence), Mother's Response,
Needs Status (7 needs seeded at construction), the development graph, and consciousness / energy /
mood / focus / vocalization in the Mind State Overview — `current_focus` and `consciousness_level`
move precisely because the consciousness branch above is live.

Net effect: of the three Development-tab panels, only **Needs Status** shows anything real.

---

## 8. The `visualization/` package

CURRENT: `visualization/` exists on disk and is **empty** — its two files (`__init__.py`,
`obsidian_api.py`) were deleted and the Obsidian export path was abandoned. Nothing in the repo
imports it. All visualization output is intended to land in the dashboard described above.

PLANNED / NOT IMPLEMENTED: the richer instrumentation sketched in [docs/ideas.md](ideas.md) —
sensory rendering, generated media, multi-character interaction views — has no code behind it.

## See also

- [neural-child-dashboard.md](modules/neural-child-dashboard.md) · [run_dashboard.md](modules/run_dashboard.md) — per-file reports
- [mind_core.md](modules/mind/mind_core.md) — the `Mind` the dashboard drives
- [mother_llm.md](modules/mother/mother_llm.md) — the working mother leg
