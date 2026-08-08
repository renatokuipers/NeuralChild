# neural_network.py

Abstract base class for every neural network in the mind simulation, plus two Pydantic records that
describe growth state. It layers developmental-stage bookkeeping, a hand-rolled SGD/"Hebbian" training
loop, gradient-magnitude-driven layer growth/pruning hooks, and multi-format save/load onto `nn.Module`.
All actual layer construction, growth, and pruning is left to subclasses; the base implementations log
and return False.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `logger` | constant | Module logger (`logging.getLogger(__name__)`), used for every error/warn path. |
| `GrowthMetrics` | model | Six floats in [0,1] (connection_density, plasticity, pruning_rate, specialization, integration, adaptability) with a stage-preset updater and `to_dict`. |
| `NeuralGrowthRecord` | model | One growth/prune/merge/noise event: timestamp, event_type, layer_affected, old/new shape, growth_factor, trigger, developmental_stage; `to_dict` serializes stage by `.name`. |
| `NeuralNetwork` | class | `nn.Module` + `ABC`. Abstract: `forward`, `process_message`, `generate_text_output`. Concrete: state/stage updates, learning, growth checks, persistence, merging, metrics. |

## Key behaviour

- Construction (neural_network.py:139-176) sets `input_dim`/`output_dim` but builds no layers; stage starts at
  `INFANT`; `state.developmental_weights` is seeded to **0.0 for every stage**.
- Learning rate is `0.01 * (0.5 + plasticity)` — recomputed in `update_developmental_stage` (:251) and
  `set_plasticity` (:1055). Effective LR is `learning_rate * developmental_weights[current_stage]` (:290, :852).
- `update_developmental_stage` (:220-258): sets weights 0.2/0.4/0.6/0.8/1.0 for stages at or below the new
  stage, 0.1 above it, refreshes `GrowthMetrics` from a hardcoded per-stage table (:49-90), then calls
  `_grow_network_for_new_stage` when `stage.value > 1`.
- `experiential_learning` (:260-337): forward → pseudo-target `where(out > 0.8*rowmax, out*1.1, out*0.9)` when no
  target → MSE → manual `param -= effective_lr * param.grad` under `no_grad`. Per-parameter mean |grad| is
  appended to `activity_tracker[name]`, capped at 50 samples. `last_activations` capped at 100 entries.
- `batch_learning` (:820-887) duplicates that loop over a stacked batch; `evaluate` (:889-958) runs `eval()` in
  chunks of 32 and returns average_loss, average/variance/max/min activation, and an entropy-derived confidence.
- Growth check thresholds: candidate for growth when mean |grad| > `growth_threshold` 0.7, for pruning when
  < `pruning_threshold` 0.1; gated by `min_experiences_before_growth` 100.

```
experiential_learning / batch_learning
  └─ experiences_since_last_growth >= 100 ?
       └─ _check_for_network_growth(trigger)
            ├─ mean|grad| per param  → growth_candidates (>0.7) / prune_candidates (<0.1)
            ├─ random() < plasticity     → _grow_layer()  → logs, records "grow",  returns False
            └─ random() < pruning_rate   → _prune_layer() → logs, records "prune", returns False
                 (counter reset happens only inside _grow_layer/_prune_layer)
```

- Persistence: `save_model`/`load_model` dispatch on a `Literal["pytorch","torchscript","onnx"]` format.
  PyTorch saves a metadata dict (state_dict, stage, state params, LR, experience count, activations, metrics,
  serialized history, dims, class name, ISO time). TorchScript and ONNX write the graph plus a
  `<path>.metadata.pt` companion; their loaders restore metadata only.
- `merge_with` (:977-1023) rejects a different class, weighted-averages parameters with matching names and
  shapes, records a "merge" event, and bumps `integration` by 0.1 (clamped to 1.0).

## Imports

Third-party: `torch`, `torch.nn`, `numpy`, `pydantic` (`BaseModel`, `Field`, `validator`).
Stdlib: `abc` (`ABC`, `abstractmethod`), `typing`, `datetime`, `os`, `logging`, `copy`, `json`.
Project-internal: `core.schemas` — `NetworkState`, `NetworkMessage`, `VectorOutput`, `TextOutput`,
`DevelopmentalStage`.

## Defects and gaps

- **`random` is never imported** yet used four times: :382, :386, :400, :402. Any call to
  `_check_for_network_growth` that reaches a growth or prune candidate raises `NameError`, i.e. the entire
  growth path is dead at runtime.
- **Learning is a no-op until a stage update happens.** `developmental_weights` is initialized to 0.0 for all
  stages (:153-155), so `effective_lr` is 0 and the `if self.training and effective_lr > 0` guard (:293, :855)
  never fires until `update_developmental_stage` is called.
- **Pseudo-target is not detached** (:283, :843): the self-supervised target is built from `output` and stays in
  the autograd graph, so `loss.backward()` differentiates through the target as well instead of treating it as a
  constant.
- **Threshold/unit mismatch.** `growth_threshold` 0.7 and `pruning_threshold` 0.1 are compared against the mean
  of stored mean-absolute-gradient samples (:310, :366-373) — an unbounded, scale-dependent quantity — while the
  docstrings and the unused `min_layer_utilization` (:174) frame the same numbers as a 0-1 "utilization" fraction.
- **`check_range` validator body is dead** (:34-40): `Field(ge=0.0, le=1.0)` already rejects out-of-range values
  before the validator runs, so `if not 0.0 <= v <= 1.0` can never be true. Its first line (:37) unconditionally
  reads `kwargs['field'].name`, which only works because Pydantic v1 injects `field` into `**kwargs`; the
  installed Pydantic version is not verifiable from this file.
- **Assignment bypasses validation.** `update_for_developmental_stage` (:95) and `_load_pytorch` (:700-702) use
  `setattr` on the model; without `validate_assignment`, checkpoint values outside [0,1] are accepted silently.
- **Comment contradicts code** at :399: "not in the same step as growth" — `_grow_layer` always returns False
  (:435), so `growth_occurred` is always False in the base class and pruning is always attempted afterwards.
- **Growth counter never resets on a no-op check.** `_check_for_network_growth` returns early (:350, :354, :377)
  without touching `experiences_since_last_growth`; only `_grow_layer`/`_prune_layer` reset it (:433, :460). Once
  the counter crosses 100 the check runs on every subsequent experience.
- **Unreachable early return** at :353-354: both callers (:325-326, :883-884) already assert the same condition.
- **Success reported after doing nothing.** `_load_torchscript` (:750) and `_load_onnx` (:787) return True even
  when the companion metadata file is absent and nothing was restored. `_load_torchscript` never copies
  parameters out of the loaded graph (:725-729), and `_load_onnx` never opens `path` at all — it reads only
  `path + ".metadata.pt"` (:769-771).
- **`_restore_growth_history` mutates its argument** via `event_dict.pop` (:803, :807), destroying the caller's
  checkpoint dict, and swallows per-event failures with a warning (:817-818), silently dropping history entries.
- **NaN entropy is silently masked** (:949-951): `normalized_outputs` can be negative (nothing constrains the
  output sign), and `torch.log` of a negative value yields NaN. `min(1.0, nan)` returns 1.0 in Python, so
  `results["confidence"]` reports a plausible-looking 0.0 rather than surfacing the failure.
- **Shape assumptions.** `torch.max(..., dim=1)` (:282, :842) and `sum(dim=1)` (:949) require 2-D output;
  `_save_torchscript`/`_save_onnx` build sample input as `torch.zeros(1, self.input_dim)` (:557, :595),
  which breaks for non-flat-vector networks.
- **Dead configuration.** `max_growth_factor` (:173) and `min_layer_utilization` (:174) are set and never read
  anywhere in this file. `clone_with_growth`'s `growth_factor`/`min_dim` params (:960) are unused — it only raises.
- **Unused imports**: `copy` (:16), `json` (:17), and the `Set`/`Union` names from `typing` (:9).
- **Magic ordinal assumptions**: `stage.value > 1` (:254) and `s.value <= stage.value` (:242) assume
  `DevelopmentalStage` is an int enum starting at 1 with INFANT lowest — not verifiable from this file.
- `np.mean(...)` result is stored into `state.parameters` (:333) as a numpy scalar, which is not JSON-serializable.

## Notes

- Methods defined here but never called anywhere within this file: `experiential_learning`, `batch_learning`,
  `evaluate`, `update_developmental_stage`, `get_growth_history`, `get_developmental_capacity`, `save_model`,
  `load_model`, `clone_with_growth`, `merge_with`, `apply_gaussian_noise`, `set_plasticity`,
  `get_complexity_metrics`. Whether external callers exist is unverifiable from this file alone.
- `torch.load` is called without `weights_only` (:668, :734, :771); on recent PyTorch the changed default can
  reject these pickled metadata dicts. `_save_pytorch` also passes the private
  `_use_new_zipfile_serialization` kwarg (:542).
- Save/load wrap everything in broad `except Exception` returning False (:511, :580, :626, :654, :710, :752, :789);
  failures are logged, not raised, so callers cannot distinguish missing file from corrupt checkpoint.
- Subclasses must override `_grow_layer`, `_prune_layer`, `_grow_network_for_new_stage`, and `clone_with_growth`
  for any of the advertised growth behaviour to exist.
