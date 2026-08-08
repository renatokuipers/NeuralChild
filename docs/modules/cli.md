# cli.py

Command-line entry point for the NeuralChild simulation. Parses arguments, mutates a loaded config, serialises it to a temporary YAML file, then runs a blocking step loop that advances a `Mind`, asks a `MotherLLM` to respond, and prints a console dashboard. Also owns signal handling, colorama-optional text colouring, and end-of-run metrics dumping.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `running` | module global (bool) | Loop sentinel; set False by `signal_handler`, reset True at cli.py:260 |
| `COLORAMA_AVAILABLE` | constant (bool) | True only if `import colorama` succeeded (cli.py:28-34) |
| `logger` | constant | `logging.getLogger(__name__)` (cli.py:37); handler setup is delegated to `Config.setup_logging`, invoked only on the `--debug` path (cli.py:361) |
| `signal_handler(sig, frame)` | function | Prints a shutdown line and clears the `running` global |
| `setup_signal_handlers()` | function | Binds `signal_handler` to SIGINT and SIGTERM |
| `colored_text(text, color, style)` | function | Wraps text in colorama codes; returns text unchanged when colorama missing |
| `display_simulation_state(mind, mother, iteration, last_response)` | function | Conditionally clears the terminal, then prints stage/energy/mood/needs/emotions/behaviours/focused-network/mother-response |
| `initialize_networks(mind, config)` | function | Builds four networks from `config.mind.networks`, registers them, optionally overrides starting stage |
| `run(config_path)` | function | Full simulation lifecycle: load config, build Mind + MotherLLM, loop, then finalise and write metrics |
| `main()` | function | Argparse front end; writes and deletes `temp_config.yaml`, calls `run` |

## Key behaviour

- `main` loads config, applies overrides for `--debug` / `--visualize` / `--simulate-llm` / `--stage`, writes it to `temp_config.yaml` in the current working directory (cli.py:373-374), then `run` **re-loads the config from that file** (cli.py:235). Overrides therefore survive only if `to_yaml` round-trips them.
- `--debug` is the only path that calls `temp_config.setup_logging()` (cli.py:361); without it nothing in this file configures logging handlers.
- Network construction fallbacks when a key is absent from `config.mind.networks`:

| Network | input_dim | hidden_dim | output_dim | Lines |
|---|---|---|---|---|
| consciousness | 64 | 128 | 64 | cli.py:180-184 |
| emotions | 32 | 64 | 32 | cli.py:189-193 |
| perception | 128 | 256 | 64 | cli.py:198-202 |
| thoughts | 64 | 128 | 64 | cli.py:207-211 |

- Loop cadence: each iteration calls `mind.step()`, then `mother.observe_and_respond(mind)`, then sleeps `max(0.0, config.mind.step_interval - elapsed)` (cli.py:291-296). The full dashboard renders only when `config.visualization.enabled` is true **and** the iteration count is divisible by 5 (cli.py:287); with visualization disabled the loop prints nothing per-iteration and the mother's replies reach only the logger (cli.py:280).
- Finalisation always runs: duration, iteration count, final stage printed; if `config.development.record_metrics` is set, a JSON dict (duration, iterations, stage name, `get_observable_state().to_dict()`, ISO timestamp) is written to `metrics/simulation_<YYYYmmdd_HHMMSS>.json` (cli.py:313-331).

```
main() --argparse--> config overrides --to_yaml--> temp_config.yaml
                                                        |
                                                   run(path)
                                                        |
   load_config -> Mind() -> MotherLLM() -> initialize_networks -> setup_signal_handlers
                                                        |
        +---------------------- while running ----------+
        |   mind.step() -> mother.observe_and_respond() -> print/dashboard -> sleep
        +-----------------------------------------------+
                                                        |
                                       finally: metrics JSON + summary print
                                                        |
                                       main finally: os.remove(temp_config.yaml)
```

## Imports

- Third-party: `yaml`, `colorama` (optional, guarded by try/except).
- Standard library: `argparse`, `time`, `sys`, `os`, `logging`, `signal`, `json`, `datetime`, `typing`.
- Project-internal: `config` (`load_config`, `Config`, `get_config`), `mind.mind_core.Mind`, `mother.mother_llm.MotherLLM`, `mind.networks.consciousness.ConsciousnessNetwork`, `mind.networks.emotions.EmotionsNetwork`, `mind.networks.perception.PerceptionNetwork`, `mind.networks.thoughts.ThoughtsNetwork`, `core.schemas.DevelopmentalStage`.

## Defects and gaps

- Unused imports: `sys` (cli.py:9), `yaml` (cli.py:15), `Dict`/`Any`/`List` (cli.py:13), `Back` (cli.py:30) — none is referenced anywhere in this file.
- `except KeyboardInterrupt` (cli.py:298) is effectively dead: `setup_signal_handlers()` runs at cli.py:251, before the `try` opens at cli.py:263, and replaces the default SIGINT handler with `signal_handler`, which sets `running = False` rather than raising.
- `display_simulation_state` accepts `mother` (cli.py:101) and never uses it.
- Screen clearing (cli.py:115-118, skipped when `development.debug_mode` is set) runs at the top of every dashboard render, wiping the per-iteration mother lines printed at cli.py:284 for the preceding iterations; only the most recent `last_mother_response` survives, re-printed inside the dashboard at cli.py:162-164.
- `initialize_networks` only propagates the developmental stage to networks when `config.mind.starting_stage != "INFANT"` (cli.py:215); an explicit `--stage INFANT` skips `update_developmental_stage` on every network entirely.
- Broad `except Exception` at cli.py:301 terminates the whole simulation on any single-step error unless `crash_on_error` re-raises; there is no per-iteration recovery.
- Metrics write failure is caught and only logged (cli.py:332-333) — the run still reports success at cli.py:335-337.
- The closing summary at cli.py:337 re-reads `mind.state.developmental_stage` outside the metrics `try/except`, so if the run died because `mind` is in a bad state that access raises from inside the `finally` and masks the original error.
- `temp_config.yaml` and `metrics/` are hardcoded relative paths (cli.py:324, cli.py:327, cli.py:373); two concurrent runs collide on the temp file, and one run's `finally` deletes the other's config.
- Cross-object contracts assumed but unverifiable from this file alone: `config.mind.networks` consumed with `.get()` (cli.py:179-206) and `mind.networks` with both `.get()` and `[...]` (cli.py:154-155) as if plain dicts; the `observe_and_respond` return accessed as `.response`/`.action` (cli.py:277-280); `get_observable_state()` accessed both attribute-wise (cli.py:124-154) and via `.to_dict()` (cli.py:319).
- `initialize_networks` assigns `mind.state.developmental_stage` directly (cli.py:218), bypassing any Mind-level transition logic; the lookup `DevelopmentalStage[config.mind.starting_stage]` (cli.py:217) assumes the five argparse choices at cli.py:346 are enum member names, with a `KeyError` fallback at cli.py:225-226 that only logs "using INFANT" — it neither assigns a stage nor calls `update_developmental_stage`, so the message describes an action the code does not take.
- cli.py:351-353 labels colorama "required" (comment at cli.py:351) but only prints a warning and continues — contradicts the optional-import guard at cli.py:28-34.
- `display_simulation_state` reads its config from the module-level `get_config()` (cli.py:114) rather than the `config` object `run` is holding, so the two can diverge unless `load_config` also populates that global — unverifiable from this file alone.
- `colored_text` appends `Style.RESET_ALL` unconditionally (cli.py:96), emitting a reset escape even when neither `color` nor `style` matched the maps.
- `if config.visualization.enabled:` at cli.py:245-248 is a `pass` with a comment saying visualization init "would be" here — no behaviour.
- `if wait_time > 0` (cli.py:295) can only ever skip a zero-length sleep, since `max(0.0, ...)` at cli.py:292 already floors the value — the guard buys nothing.

## Notes

- `run()` can be imported and called directly with a config path; it does not depend on `main()`.
- The module docstring advertises "options for configuration and visualization", but the only visualization path implemented is console printing.
