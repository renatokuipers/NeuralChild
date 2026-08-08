# consciousness.py

Defines `ConsciousnessNetwork`, an RNN-based subclass of `NeuralNetwork` that folds per-network activation reports into a single scalar "awareness level" plus a learned self-model vector. It handles inter-network messages, runs a periodic autonomous tick that fluctuates and republishes awareness, and rescales its own parameters by developmental stage. Located at mind/networks/consciousness.py.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `ConsciousnessNetwork` | class | Subclass of `NeuralNetwork`; ctor takes input_dim=64, hidden_dim=128, output_dim=64 |
| `.forward(x)` | method | Runs 2-layer RNN over input, returns output_dim tensor; mutates `self.hidden` and `self.awareness_level` |
| `.process_message(msg)` | method | Returns a `VectorOutput` or `None`, depending on message content/type |
| `.autonomous_step()` | method | Integrates activations, decays hidden state, jitters awareness, queues a message dict in state |
| `._integrate_activations()` | method | Recomputes `awareness_level` as mean activation × integration_capacity, capped per stage |
| `.update_developmental_stage(stage)` | method | Calls super, then sets self_awareness / integration_capacity from a stage table |
| `.generate_text_output()` | method | Returns `TextOutput` with prose describing awareness and stage |
| `.clone_with_growth(growth_factor=1.2, min_dim=8)` | method | Constructs a larger `ConsciousnessNetwork` and copies scalar state (not weights) |
| `logger` | constant | Module-level `logging.getLogger(__name__)` |

## Key behaviour

- Tensor flow in `forward` (consciousness.py:86-127): 2D input is unsqueezed to [batch, 1, input_dim]; RNN yields output [batch, seq, hidden_dim]; the last timestep [batch, hidden_dim] feeds `output_layer` → [batch, output_dim].
- `self.hidden` is reassigned from the RNN on every call (consciousness.py:104) and persists between invocations, carrying both a fixed batch size and the previous graph.
- Self-model blending only activates when `self_awareness > 0.1` (consciousness.py:116); the INFANT value is exactly 0.1, so it is off at the starting stage.
- `awareness_level` is written twice by different formulas: `sigmoid(result.mean())` in `forward` (consciousness.py:113) and mean-activation × integration_capacity in `_integrate_activations` (consciousness.py:269). Whichever ran last wins.
- Stage caps on awareness (consciousness.py:259-265): INFANT 0.3, TODDLER 0.5, CHILD 0.7, ADOLESCENT 0.9, MATURE 1.0; unknown stage falls back to 0.3.
- Stage parameter table (consciousness.py:283-304): self_awareness 0.1/0.3/0.5/0.7/0.9 and integration_capacity 0.2/0.4/0.6/0.8/1.0 across INFANT→MATURE.
- `autonomous_step` decays hidden by ×0.95, adds uniform jitter in ±0.025, clamps awareness to [0.1, 1.0], then appends a serialized `NetworkMessage` (priority 0.7, receiver "mind") to a `pending_messages` list in state (consciousness.py:210-240).
- `recent_inputs` in state is truncated to the last 4 entries plus the new one, i.e. max length 5 (consciousness.py:124).

```
process_message dispatch (consciousness.py:139-199)
  content["vector_data"] non-empty ─► pad/truncate to input_dim ─► forward ─► VectorOutput(data=output)
                                       └─ records network_activations[sender], sets attending_to
  elif type == "activation_update" ──► store float activation ─► _integrate_activations ─► VectorOutput([awareness]*output_dim)
  elif type == "query"
        query_type == "awareness_level" ─► VectorOutput([awareness]*output_dim)
        query_type == "self_model" AND self_awareness > 0.3 ─► zero-input RNN step ─► self_model(hidden[-1])
  otherwise ─► None
```

## Imports

- Third-party: `torch`, `torch.nn`, `numpy`, plus stdlib `typing`, `random`, `datetime`, `logging`.
- Project-internal: `NeuralNetwork` from core.neural_network; `NetworkMessage`, `VectorOutput`, `TextOutput`, `DevelopmentalStage` from core.schemas.

## Defects and gaps

- `copy` is used at consciousness.py:392 and consciousness.py:396 but never imported — `clone_with_growth` raises `NameError` on first call.
- `NeuralGrowthRecord` is used at consciousness.py:397 but never imported — a second missing name, never actually reached because the `copy` failure at line 392 aborts the method first.
- Hidden-state init hardcodes `torch.zeros(2, x.size(0), 128)` (consciousness.py:101) instead of the ctor's `hidden_dim`; the leading 2 does match the hardcoded `num_layers=2` (consciousness.py:44). Any `hidden_dim != 128` makes the first `forward` raise a shape error inside the RNN — exactly what `clone_with_growth` produces (default new hidden dim 153).
- `hidden_dim` is never stored on the instance; the only later reader is `self.rnn.hidden_size` (consciousness.py:374, 400, 413).
- `clone_with_growth` copies only scalar state; no RNN, self_model, or output_layer weights are transferred, so the "clone" is randomly initialized. `network_activations` is also not copied.
- `self.hidden` is never detached (consciousness.py:104, scaled again at 212), so the retained autograd graph accumulates across calls, and a batch-size change between calls raises inside the RNN.
- `awareness_level` is set from `result` before the self-model blend (consciousness.py:113 vs 117-119), so the recorded level describes a vector different from the one actually returned whenever `self_awareness > 0.1`.
- The `if "vector_data" in ...` / `elif message_type == ...` chain (consciousness.py:139-176) is keyed on content first, so an `activation_update` or `query` message that also carries a non-empty `vector_data` never reaches its own branch.
- A `query` of type `self_model` at TODDLER (self_awareness 0.3) fails the strict `> 0.3` test at consciousness.py:187 and silently returns `None`.
- `_integrate_activations` applies `min(max_awareness, integrated_activation)` with no lower bound (consciousness.py:269); negative reported activations drive `awareness_level` negative until the next `autonomous_step` clamp.
- The comment at consciousness.py:310 says "Increase hidden state dimensionality", but the code only sets `self.hidden = None`; no dimension changes.
- The stage-ordering test `stage.value >= DevelopmentalStage.CHILD.value` (consciousness.py:311) assumes ordered comparable enum values; whether the enum values are ints or strings is unverifiable from this file alone.
- `attending_to` is kept twice, as an instance attribute and as a state parameter (both written at consciousness.py:155-156, read separately at 232 and 334), and is never cleared — after the first vector message the text output claims focus on that sender indefinitely.
- Acknowledgment vectors are a scalar repeated `output_dim` times (consciousness.py:173, 184) rather than any real representation.
- The `self_model` query path pushes a batch-1 zero vector through the RNN and overwrites the persistent `self.hidden` (consciousness.py:189-191), so a read-only query perturbs continuity state and raises if `self.hidden` currently holds a different batch size.
- Unused imports: `numpy as np` (line 12), `datetime` (line 11), and `Dict`, `Any`, `List`, `Tuple` from typing (line 9) — only `Optional` is referenced.
- Defined but never called within this file: `process_message`, `autonomous_step`, `update_developmental_stage`, `generate_text_output`, `clone_with_growth`. Whether external callers exist is unverifiable from this file alone.

## Notes

- `update_state`, `self.state.parameters`, `self.developmental_stage`, `self.input_dim`, `self.output_dim`, `self.name`, `growth_metrics`, `experience_count`, and `growth_history` are all assumed to come from the base class; none are defined here, and their semantics (in particular whether `update_state` merges or replaces the parameter dict) cannot be checked from this file.
- `NetworkMessage.to_dict()` (consciousness.py:239) is assumed to exist on the schema type; unverifiable here.
