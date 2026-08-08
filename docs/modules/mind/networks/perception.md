# perception.py

Defines `PerceptionNetwork`, a `NeuralNetwork` subclass that fuses a visual and an auditory input stream into a single perceptual vector, and derives a symbolic perception record (valence, salience, description) whose vocabulary is gated by developmental stage. It also emits outbound `NetworkMessage` objects by stashing them in a `pending_messages` state key that nothing in this file consumes. Stage changes rewrite two scalar capacities (`object_recognition`, `pattern_recognition`) that in turn control noise injection and autonomous pattern firing.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `PerceptionNetwork` | class | Two-stream sensory network with attention, stage-gated perception vocabulary. |
| `__init__(input_dim=128, hidden_dim=256, output_dim=64)` | method | Builds submodules; `hidden_dim` is not stored on the instance. |
| `forward(x)` | method | (B, input_dim) → (B, output_dim) in [0,1]; mutates `self.attentional_focus`. |
| `process_message(msg)` | method | Handles `sensory_input` and `attention_request`; returns `VectorOutput` or `None`. |
| `_extract_perception(output, content)` | method | Tensor + raw content → dict of perception fields. |
| `_remember_perception(perception)` | method | Appends to ring buffer sized by stage, mirrors to state. |
| `autonomous_step()` | method | Probabilistically queues a `pattern` message to `thoughts`. |
| `update_developmental_stage(stage)` | method | Calls super, then sets the two capacity scalars from a lookup table. |
| `generate_text_output()` | method | `TextOutput` describing the latest perception, confidence 0.4/0.6/0.7. |
| `clone_with_growth(growth_factor=1.2, min_dim=8)` | method | Intended to return a scaled clone; raises at runtime (see Defects). |

## Key behaviour

Dimensions with defaults (input 128 / hidden 256 / output 64):

| Stage | Shape |
|---|---|
| input split | visual `x[:, :64]`, auditory `x[:, 64:]` |
| each processor | 64 → 128 → 64 (Linear, ReLU, Dropout 0.2, Linear, ReLU) |
| concat | 128 (= hidden_dim // 2) |
| attention | Linear(128, 2), softmax over dim=1 |
| integration | 128 → 128 → 64, Sigmoid |

```
sensory_input msg ──► pad/truncate each modality to input_dim//2 ──► concat list
      │                                                                  │
      ▼                                                             torch.tensor
 attention_request ──► set focus ──► VectorOutput(one-hot halves)         │
                                                                          ▼
   forward: split ─► visual/auditory MLP ─► cat ─► [attention if stage>INFANT]
                                                   ─► integration ─► [+noise if
                                                      object_recognition<0.5]
                                                          │
                       _extract_perception ◄──────────────┘
                             │
              _remember_perception          NetworkMessage→"emotions" (priority 0.7)
                                                    into state["pending_messages"]
```

- perception.py:115 gates attention on `developmental_stage.value > INFANT.value`; perception.py:126 flips `attentional_focus` inside `forward` based on mean attention weights.
- perception.py:132-135 adds Gaussian noise scaled by `0.5 - object_recognition`, then clamps to [0,1]. Active only for INFANT (0.2) and TODDLER (0.4); CHILD+ (0.6/0.8/0.9) get none.
- perception.py:281 valence = 2 × (mean of first output half − mean of second half), clamped to [-1, 1]. Because perception.py:281 always writes the key, the guard at perception.py:218 is always true, so every processed `sensory_input` queues an `emotions` message.
- perception.py:341 recent-perception cap = `3 + stage.value * 3`.
- perception.py:357 autonomous pattern firing needs >2 stored perceptions and `pattern_recognition > 0.3`, so TODDLER (exactly 0.3) is excluded; CHILD+ only. The "detection" is `random.random() < pattern_recognition` (perception.py:359) — recent perceptions are never inspected, only counted.
- Stage capacity table at perception.py:391-412: INFANT .2/.1, TODDLER .4/.3, CHILD .6/.5, ADOLESCENT .8/.7, MATURE .9/.9.
- Description vocabulary branches use enum identity for INFANT/TODDLER (perception.py:297, 306) but `.value >=` for CHILD (perception.py:315); ADOLESCENT/MATURE fall into the CHILD branch.

## Imports

- Third-party: `torch`, `torch.nn`, `numpy`, plus stdlib `typing`, `random`, `datetime`, `logging`.
- Project-internal: `NeuralNetwork` from `core.neural_network`; `NetworkMessage`, `VectorOutput`, `TextOutput`, `DevelopmentalStage` from `core.schemas`.

## Defects and gaps

- `clone_with_growth` cannot run. The first failing statement is perception.py:481, reading `self.visual_processor[1].out_features`: index 1 is the `ReLU`, not a `Linear`, so it raises `AttributeError`. Index 0 is the layer whose `out_features` is `hidden_dim // 2`. The same expression recurs at perception.py:506 and 515.
- Past that point the method would still fail twice with `NameError`: `copy.deepcopy` at perception.py:495, 498, 502 with no `copy` import, and `NeuralGrowthRecord` at perception.py:503 with no import for that name.
- perception.py:492-502 transfers only scalars, the perception list and growth history; there is no weight transfer, so a successful clone would still return a freshly initialised network.
- `hidden_dim` is a constructor arg that is never stored (perception.py:30); the clone instead reverse-engineers it from layer shapes.
- perception.py:59-67 assume `hidden_dim % 4 == 0`: integration/attention expect `hidden_dim // 2` inputs while the concat produces `2 * (hidden_dim // 4)`. Any hidden_dim not divisible by 4 gives a shape mismatch at `forward`. `clone_with_growth` computes `int(hidden * growth_factor)` (perception.py:481) with no such rounding.
- Odd `input_dim` also breaks `forward`: perception.py:105 slices auditory as `x[:, input_dim // 2:]`, one element wider than the `input_dim // 2` its first `Linear` accepts (perception.py:51). perception.py:480 can produce an odd `new_input_dim`.
- perception.py:126 mutates `attentional_focus` without calling `update_state`, so `state["attentional_focus"]` silently diverges from the attribute (only perception.py:248 keeps them in sync).
- perception.py:101 assigns `batch_size` and never uses it.
- perception.py:271 and perception.py:284 compute the same number: outputs are post-Sigmoid (and clamped) non-negatives, so `output.abs().mean()` equals `output.mean()`. `salience` is a duplicate of `output_summary`.
- perception.py:447 and perception.py:454 test `if self.attentional_focus:` — the attribute is initialised to `"visual"` and only ever assigned `"visual"`/`"auditory"`, so the false branch is unreachable.
- perception.py:252-253 returns a vector of length `2 * (output_dim // 2)`, one element short of `output_dim` for odd `output_dim`.
- `numpy as np` (perception.py:12) and `List`, `Tuple` (perception.py:9) are imported and never used.
- Nothing in this file calls `eval()`, so unless the base class or a caller switches modes (unverifiable from this file alone), the Dropout layers at perception.py:44 and 53 stay active and inference through `process_message` is stochastic beyond the deliberate noise term.
- `self.growth_metrics`, `self.experience_count`, `self.growth_history` (perception.py:498-502) are read but never initialised here; whether the base class provides them is unverifiable from this file alone.
- The comment at perception.py:356 claims the code looks for patterns in recent perceptions; the body only counts them and rolls a random number.

## Notes

- `stage.value` is used both as an ordered comparand (perception.py:115, 315) and as an integer multiplicand (perception.py:341), so `DevelopmentalStage` must be int-valued; this cannot be confirmed without `core.schemas`.
- `process_message` returns `None` for any message type other than the two handled, and also when `sensory_input` carries neither a `visual` nor an `auditory` key.
- Outbound messages are never sent directly; they are appended to `state["pending_messages"]` as dicts via `to_dict()` (perception.py:233, 376). Nothing in this file consumes that key, so whether anything drains it is unverifiable from this file alone.
- perception.py:414 only updates the capacities when the stage is a key of the table; an unlisted stage leaves both scalars unchanged and writes no state.
