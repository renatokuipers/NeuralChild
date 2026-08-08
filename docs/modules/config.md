# config.py

Pydantic-based configuration schema and global config singleton for NeuralChild. Defines six nested settings models rolled up into one `Config` root model, plus YAML/JSON persistence and root-logger setup. Exposes a module-level mutable global (`config`) with `load_config`/`get_config` accessors. Failure paths log and swallow rather than raising: `from_yaml` falls back to defaults, while `to_yaml`/`to_json` return normally whether or not the write succeeded.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `logger` | constant | Module logger, `logging.getLogger(__name__)` (config.py:15) |
| `ServerConfig` | model | LLM/embedding endpoint URLs (default `localhost:1234`), optional Obsidian API URL (config.py:17) |
| `ModelConfig` | model | LLM + embedding model names, `temperature` bounded 0.0–2.0, `max_tokens` default -1 (config.py:23) |
| `VisualizationConfig` | model | Enable flags, `update_interval` 1.0 s, hex colour map, per-network display bool map (config.py:30) |
| `MindConfig` | model | Simulation timing intervals, per-network dimension dict, acceleration factor, feature flags (config.py:55) |
| `LoggingConfig` | model | Level/handlers/format; `validate_log_level` coerces unknown levels to INFO (config.py:88) |
| `DevelopmentConfig` | model | Debug/simulate/profile/crash/metrics booleans plus free-form `experimental_features` (config.py:105) |
| `Config` | class | Root aggregate of the six sections, all with `default_factory` (config.py:116) |
| `Config.from_yaml` | classmethod | Load path → `Config`; returns defaults on missing file, empty file, or any exception (config.py:126) |
| `Config.to_yaml` | method | Dump `model_dump()` to YAML, creating parent dirs; logs and swallows errors (config.py:154) |
| `Config.to_json` | method | Same as above, JSON with indent 2 (config.py:171) |
| `Config.setup_logging` | method | Rebuilds root-logger handlers from `self.logging` (config.py:188) |
| `config` | constant | Module-level `Config()` instance built at import time (config.py:224) |
| `load_config` | function | Rebinds global `config` from YAML path (default `"config.yaml"`), calls `setup_logging`, returns it (config.py:226) |
| `get_config` | function | Returns the current global `config` (config.py:243) |

## Key behaviour

- Every model field has a default or `default_factory`; `Config()` with no arguments is always valid, so a completely absent config file is indistinguishable from a valid one at the call site.
- `from_yaml` has three fallback-to-defaults branches: file missing (config.py:135), `yaml.safe_load` returning `None` for an empty file (config.py:144), and the blanket `except Exception` at config.py:149 which also absorbs Pydantic validation errors.
- Bounded fields (Pydantic `Field` constraints): `temperature` 0.0–2.0; `learning_rate` 0.0001–0.1; `step_interval` 0.01–10.0 s; `need_update_interval` ≥0.1 s; `memory_consolidation_interval` ≥1.0 s; `development_check_interval` ≥5.0 s; `network_growth_check_interval` ≥10.0 s; `development_acceleration` 0.1–10.0.
- Default network dimensions in `MindConfig.networks` (config.py:65-71), as raw ints in an untyped `Dict[str, Any]`:

| Network | input_dim | hidden_dim | output_dim |
|---|---|---|---|
| consciousness | 64 | 128 | 64 |
| emotions | 32 | 64 | 32 |
| perception | 128 | 256 | 64 |
| thoughts | 64 | 128 | 64 |
| language | 96 | 192 | 48 |

- `setup_logging` builds a handler list first, then strips *all* existing root handlers (config.py:214-215) before attaching the new ones, so an empty build leaves the root logger with zero handlers.
- Load/apply flow:

```
load_config(path)
  └─ Config.from_yaml(path)
       ├─ missing file ──────────┐
       ├─ empty YAML ────────────┼─→ Config()  (defaults, warning logged)
       └─ any exception ─────────┘
       └─ model_validate(dict) ──→ Config
  └─ config.setup_logging()
       ├─ console_logging → StreamHandler
       ├─ file_logging   → FileHandler (failure → print, continue)
       └─ replace root handlers wholesale
  └─ rebind module global `config`, return it
```

## Imports

- Third-party: `pydantic` (`BaseModel`, `Field`, `model_validator`), `yaml`.
- Standard library: `typing` (`Optional`, `Dict`, `Any`, `List`), `os`, `logging`, `json`.
- Project-internal: none.

## Defects and gaps

- config.py:8 — `List` is imported but never used anywhere in the file.
- config.py:149 — bare `except Exception` in `from_yaml` swallows malformed YAML *and* schema validation errors and silently substitutes defaults; an out-of-range value or unparseable file produces a running system on default settings with only an ERROR log line. Note the fallback is all-or-nothing: one bad field discards every other valid setting in the file.
- No model sets `model_config`/`extra`, so Pydantic's default "ignore extra" applies to all seven models. A misspelled section or key in the YAML is dropped silently — no exception, no log line — and the corresponding default is used, so this failure mode is invisible even in the ERROR path above.
- config.py:150-151 vs config.py:239 — `from_yaml`'s diagnostics are emitted *before* `setup_logging` runs. On the first `load_config` call the root logger still has no handlers, so `logging.lastResort` (WARNING threshold) applies and `logger.info("Using default configuration")` at config.py:151 is discarded; the reason defaults were chosen is the part most likely to be lost.
- config.py:168 and config.py:185 — `to_yaml`/`to_json` catch every exception, log, and return `None` regardless; callers cannot distinguish a successful save from a failed one, and a partially written file is left on disk.
- config.py:200-207 — file-handler setup failure is reported with `print`, not `logger`, and execution continues; the caller gets no signal that file logging is inactive.
- config.py:194-219 — if both `console_logging` and `file_logging` are false, `handlers` is empty and the root logger ends up with zero handlers; the confirmation message at config.py:221 is then dropped by the `lastResort` fallback (WARNING threshold), so the "Logging configured" claim is never actually visible.
- config.py:190 — `getattr(logging, self.logging.level, logging.INFO)` resolves the level by attribute lookup on the `logging` module. It is only safe because `validate_log_level` runs at construction; a post-construction mutation of `.level` (or `model_construct`) is not re-validated and any module attribute name would be returned as the "level".
- config.py:96-103 — the validator mutates `self.level` inside an `mode="after"` validator instead of rejecting; invalid input is downgraded to a warning, so bad config never fails loudly.
- config.py:77 — `starting_stage` is a plain `str` with no enum or membership validation; any string passes.
- config.py:28 — `max_tokens` has no `Field` constraint despite the `-1 for unlimited` sentinel convention; arbitrary negatives are accepted.
- config.py:65 — `networks` is `Dict[str, Dict[str, Any]]`; nothing validates that a user-supplied entry contains `input_dim`/`hidden_dim`/`output_dim` or that they are positive ints.
- config.py:33 — `update_interval` has no lower bound, unlike every comparable interval in `MindConfig`.
- config.py:154 (`to_yaml`) and config.py:171 (`to_json`) are defined but never referenced anywhere within this file.
- config.py:62 — the field comment ends with the leftover authoring note `# seconds - add this line!`.
- config.py:224 / config.py:236 — `load_config` rebinds the module global rather than mutating the existing instance, so the import-time `Config()` object at config.py:224 and the post-load object are different instances. Whether that matters depends on how importers bind the name; not verifiable from this file alone.

## Notes

- `DevelopmentConfig` flags (`simulate_llm`, `profile_performance`, `crash_on_error`, `record_metrics`) and every `features_enabled`/`network_display` toggle are pure data here — this file contains no code that reads them.
- `to_yaml`/`to_json` serialise via `model_dump()`, which yields plain scalars/dicts, so `yaml.dump` and `json.dump` both round-trip without custom encoders.
- `os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)` is safe for bare filenames because `abspath` guarantees a non-empty dirname.
