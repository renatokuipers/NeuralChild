# legacy neural-child/config.yaml

Documents the **legacy neural-child repository's** `config.yaml` (59 lines). All line citations below use the `legacy-config.yaml:N` prefix and refer only to that file; whether any other file of the same name in this workspace shares its content is unverifiable from this file alone. It is a static YAML data file declaring hardware/device switches, model dimensions, an 18-entry developmental stage threshold ladder, memory capacities, emotional and ethical scalars, and optimizer hyperparameters. No loader, schema, or defaults logic — data only.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `hardware` | section | `cuda` bool, `mixed_precision` bool, `memory_optimization` bool, `device` string `"cuda"` (legacy-config.yaml:2-6) |
| `model_params` | section | `base_dim` 128, `hidden_dim` 1024, `num_layers` 12, `num_heads` 16, `dropout` 0.1 (legacy-config.yaml:8-13) |
| `curriculum.stage_thresholds` | mapping | 18 named developmental stages → float threshold in 0.2–0.99 (legacy-config.yaml:16-34) |
| `memory` | section | `short_term_capacity` 1000, `long_term_capacity` 50000, `replay_batch_size` 32, `consolidation_interval` 3600, `working_memory_size` 10 (legacy-config.yaml:36-41) |
| `emotional_regulation` | section | `emotion_dim` 4, `context_window` 5, `memory_dim` 32 (legacy-config.yaml:43-46) |
| `ethical_constraints` | section | `harm_threshold` 0.4, `fairness_weight` 0.9, `honesty_bias` 0.75, `anxiety_threshold` 0.7 (legacy-config.yaml:48-52) |
| `training` | section | learning rate in bare exponent form (loads as a string under PyYAML — see defects), weight decay 0.01, gradient clip norm 1.0, warmup steps 1000, checkpoint interval 100 (legacy-config.yaml:54-59) |

## Key behaviour

- Seven top-level sections. Each is a flat map of scalars except `curriculum` (legacy-config.yaml:15), which holds a single child key `stage_thresholds` (legacy-config.yaml:16) that in turn holds the 18 stage entries — two levels of nesting for one leaf map. No anchors, aliases, merge keys, multi-document separators, or environment interpolation — a plain safe-load yields a nested dict of scalars.
- The stage ladder reads as a monotonic progression, but the declared values are non-decreasing rather than strictly increasing: `late_childhood` and `early_elementary` are both 0.9 (legacy-config.yaml:26-27).
- Threshold spacing narrows as stages advance: five 0.10 steps up to `early_preschool` 0.7, four 0.05 steps through `late_childhood` 0.9, a 0.00 step into `early_elementary`, two 0.02 steps, then five 0.01 steps to `mature_adult` 0.99 (legacy-config.yaml:17-34). Later stages need far finer progress signals to be distinguishable.
- Key insertion order is the only encoding of stage sequence. Nothing marks a stage as first, last, or terminal.

```
legacy neural-child/config.yaml
├── hardware              → device / precision switches
├── model_params          → 128 base, 1024 hidden, 12L x 16H, p=0.1
├── curriculum
│   └── stage_thresholds  → newborn 0.20 ─┐ 18 stages,
│                            ...          ├ non-decreasing,
│                            mature_adult 0.99 ─┘ tie at 0.9
├── memory                → 1k STM / 50k LTM / 10 working, 3600 interval
├── emotional_regulation  → 4-dim emotion, 5 context, 32 memory
├── ethical_constraints   → 4 scalar gates
└── training              → lr / wd / clip / warmup / ckpt
```

## Imports

None. A YAML data file with no includes, no Python object tags, and no external references.

## Defects and gaps

- legacy-config.yaml:55 — the learning rate is written in bare exponent form (three, `e`, minus four) with no decimal point. That is not a valid implicit float under the YAML 1.1 resolver PyYAML uses, which requires a decimal point and a signed exponent. Under a PyYAML safe-load it resolves to a **string**, not a float. Loaders using the YAML 1.2 core schema parse it as a float, so the loaded type is loader-dependent. What any consumer does with a string here is unverifiable from this file alone.
- legacy-config.yaml:26-27 — duplicate threshold 0.9 for `late_childhood` and `early_elementary`. A strict greater-than advancement check stalls; a greater-or-equal check advances two stages at once. The ladder cannot resolve between them.
- legacy-config.yaml:3 and legacy-config.yaml:6 — `cuda: true` and `device: "cuda"` duplicate the same knowledge in two forms with no stated precedence. Setting one false while leaving the other unchanged yields a contradictory config this file cannot detect.
- legacy-config.yaml:40 — the consolidation interval (3600) carries no unit. Seconds is the obvious reading; steps or episodes are equally plausible and nothing disambiguates.
- legacy-config.yaml:59 — the checkpoint interval (100) is likewise unitless: steps vs. epochs vs. episodes.
- legacy-config.yaml:9 and legacy-config.yaml:10 — `base_dim` and `hidden_dim` are declared independently with no stated relation; changing one silently desynchronizes any code assuming a fixed ratio.
- legacy-config.yaml:1 — header comment reads "NeuralMind Configuration" while the project is neural-child. Stale or copied name.
- Nine trailing comments (legacy-config.yaml:6, 9, 13, 37, 38, 41, 44, 52, 55) assert their value was "added to match" code elsewhere. Four name a `.py` file (a memory module twice, an emotional-regulation module, a self-supervised trainer); two name an implementation (a dynamic-neural-child class, a defense-mechanisms implementation); three say only "the code". None of these targets is verifiable from this file alone — treat them as claims, not facts.
- No seed, no output or checkpoint paths, no logging section, no batch size for the main training loop (only a replay batch size under memory), and no schema or version key. Whether those settings exist elsewhere or are simply absent cannot be determined from this file.
- legacy-config.yaml:9-12 — `num_heads` 16 divides both 1024 (64 per head) and 128 (8 per head) evenly, but nothing records that as a constraint, so an edit to `hidden_dim` can silently violate it.
- The stage thresholds carry no stated metric or comparison direction. Nothing in the file names the quantity compared against 0.2–0.99, so the tie at 0.9 and the 0.01 spacing cannot be judged against any tolerance.

## Notes

- Every value is a plain scalar with no ranges, enums, or bounds. A dropped or mistyped key surfaces as a lookup error or a wrong-typed value at first use rather than at load time, unless validation exists outside this file.
- The 18-stage ladder is the largest and most fragile block. Treat it as the primary migration target and re-derive strictly increasing thresholds when porting.
- Nothing here was cross-checked against consumer code. Every threshold, dimension, and interval is reported as declared; which values are actually read at runtime, and under what key path, cannot be determined from this file.
