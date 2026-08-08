# self_supervised_trainer.py

Defines `AutonomousTrainer`, a single-step training driver that couples a child model, a memory object exposing `record_experience` and a `replay_optimizer.sample_batch` sampler, and a network returning a `moral_score`. One call performs a live forward pass, records the experience, samples a replay batch, and applies one AdamW update. There is no loop, scheduler, checkpointing, or evaluation path in this file.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `AutonomousTrainer` | class | Holds child model, memory, moral net, and one AdamW optimizer (lr 3e-4, self_supervised_trainer.py:13). |
| `AutonomousTrainer.training_step(inputs)` | method | Runs one forward + record + replay-batch update; returns the replay loss as a float. |
| `AutonomousTrainer._process_batch(batch_size=32)` | method | Samples memory, computes the combined replay loss tensor; returns a zero tensor when the sample is empty. |
| `AutonomousTrainer._safe_tensor_op(tensor, requires_grad=False)` | method | Clone/detach/optionally re-enable grad, move to `self.child.device`. Never called in this file. |

## Key behaviour

- Module import has a global side effect: anomaly detection is switched on process-wide at self_supervised_trainer.py:4, which makes every backward pass in the whole process substantially slower.
- `training_step` (15-58) order: `zero_grad` → clone/detach/`requires_grad_` on `inputs` → `child(inputs)` → `moral_net(outputs)['moral_score']` → record experience under `no_grad` → `_process_batch()` → `backward()` → grad clip at `max_norm=1.0` → `optimizer.step()` → overwrite `child.current_beliefs` with detached outputs.
- The live forward pass results (`outputs`, `moral_feedback`) contribute **nothing** to the loss. The gradient graph built at line 26 is constructed and discarded; all learning signal comes from the replay batch.
- `record_experience` is called with five positional values: cloned inputs, cloned outputs, `moral_feedback.item()`, `time.time()`, and a clone of `child.emotional_state` (31-37).
- Replay samples are treated as flat vectors with a fixed layout, sliced by hardcoded offsets (69-82).

| Slice | Width | Interpreted as |
| --- | --- | --- |
| `sample[:768]` | 768 | model input |
| `sample[768:896]` | 128 | past state |
| `sample[896]` | 1 (scalar) | reward |

- Replay loss (88-106) = `0.7 * consistency + 0.3 * moral + 0.1 * ewc`, where consistency is `1 - mean(cosine(current_outputs, past_states))`, moral is MSE between squeezed moral scores and rewards, and "EWC" is the plain L2 norm of parameters whose name contains `_plasticity`.

```
inputs ─► child ─► outputs ─► moral_net ─► score ─┐
                                                  ├─► memory.record_experience (no_grad, discarded graph)
                        emotional_state, time() ──┘

memory.replay_optimizer.sample_batch(32)
   └─► samples ─► [inputs | past_states | rewards]
                       └─► child ─► current_outputs ─┬─► cosine vs past_states ─► consistency (0.7)
                                                     └─► moral_net ─► MSE vs rewards (0.3)
                            child params matching '_plasticity' ─► L2 ─► "ewc" (0.1)
                                                     └─► backward ─► clip(1.0) ─► AdamW.step
```

## Imports

- Third-party: `torch`, `torch.nn` (as `nn`).
- Standard library: `time`.
- Project-internal: none. `child_model`, `memory`, and `moral_net` arrive only as constructor arguments.

## Defects and gaps

- Empty-replay crash: when `sample_batch` returns no samples, `_process_batch` returns a fresh `torch.tensor(0.0)` with `requires_grad=False` (self_supervised_trainer.py:66); `replay_loss.backward()` at line 43 then raises "element 0 of tensors does not require grad". There is no guard for the empty-memory case.
- `grad_norm` is assigned at line 46 and never read — clipping happens, but the reported norm is dropped.
- `indices` from `sample_batch` is bound at line 64 and never used; nothing feeds updated priorities or TD errors back to the replay optimizer, so a prioritized sampler would never re-weight.
- `_safe_tensor_op` (110-115) is defined but referenced nowhere in this file; the same clone/detach idiom is instead inlined at lines 23, 70, 75, 80.
- Cosine similarity at line 88 compares `current_outputs` against 128-wide `past_states`, while the model is fed 768-wide inputs. Unless the child's output width is exactly 128, `CosineSimilarity(dim=-1)` raises on the size mismatch. Cannot be confirmed from this file alone.
- `moral_feedback.item()` at line 34 requires a single-element tensor, so `training_step` only accepts input whose moral score is scalar; a batched score raises. The batch shape depends on `moral_net` and is unverifiable from this file alone.
- `moral_net` is used in the replay loss (line 91) but its parameters are not in the optimizer (line 13 registers only `child_model.parameters()`). `backward()` at line 43 therefore accumulates gradients on `moral_net` without bound: `zero_grad()` at line 20 clears only the optimizer's parameters, and no step ever consumes or clears them.
- `moral_scores.squeeze()` at line 92 collapses a batch of one to a 0-dim tensor while `rewards` stays 1-dim, producing a broadcast-shape MSE (silently wrong loss, plus a torch warning).
- Named "EWC" (95-99) but computes no Fisher information and keeps no reference parameters — it is L2 weight decay restricted to `_plasticity` names. If no parameter matches, `sum` yields the integer 0 and the term silently vanishes.
- `consistency_loss` (89) builds `torch.ones_like(similarity).mean()` purely to obtain the constant 1.0.
- Both docstrings claim the code "avoids in-place operations" (17, 62) while `requires_grad_` and `clip_grad_norm_` are in-place calls; the claim describes intent, not the code.
- Hardcoded slice offsets 768 and 896 (69-82), `batch_size=32` (60), `max_norm=1.0` (48), and the 0.7/0.3/0.1 loss weights (103-105) are all inline with no configuration path. Nothing validates that a sample is at least 897 elements wide.
- Line 23 detaches the caller's `inputs` and re-enables grad on the copy, so any graph the caller built upstream is severed inside `training_step`.
- `self.child.device` is read at lines 66, 72, 77, 82, 115. `nn.Module` exposes no `.device` attribute by default, so this depends on the child class defining one — unverifiable from this file alone.
- `requires_grad_(True)` on replay inputs (line 70) serves no purpose: no gradient w.r.t. the inputs is ever consumed, and it only adds graph overhead.

## Notes

- Anomaly detection (line 4) is a debugging aid left enabled at import; it slows every backward pass in the process, not just this module's.
- The class is stateless between calls apart from the optimizer's moment buffers and the `child.current_beliefs` overwrite at line 56 — that write is the only durable effect on the child besides parameter updates.
- No exception handling anywhere in the file; all failures propagate to the caller.
