# training_system.py

Training loop scaffolding for the "developmental child" model: loss/gradient statistics tracking, checkpoint retention with stability filtering, early stopping, and a trainer that combines four weighted loss terms per step. `DevelopmentalTrainer` owns an AdamW optimizer plus a cosine-warm-restart scheduler and drives an episode loop that pulls stimuli from an injected "mother" object. Nothing in this file is executed at import time; there is no `__main__` block.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `MovingAverageMonitor` | class | Rolling buffers (default window 50) for loss, grad norm, LR, per-component loss; anomaly checks and episode summary. |
| `MovingAverageMonitor.update_stats` | method | Appends one step's numbers, returns a stats dict incl. `loss_ma`, `loss_std`, `grad_ma`, per-component mean/std. |
| `MovingAverageMonitor.check_loss_spike` | method | True if z-score of current loss > threshold; when std==0 falls back to `loss > mean * threshold`. |
| `MovingAverageMonitor.check_gradient_issues` | method | True on NaN/Inf; once ≥2 samples are buffered, also on `grad_norm < 1e-7` or z-score > 3.0. |
| `MovingAverageMonitor.log_error` | method | Appends `{step, type, message, timestamp}` to `error_log`. |
| `MovingAverageMonitor.summarize_episode` | method | Aggregates a list of step-stats; per-component linear `trend` via `np.polyfit(deg=1)`. |
| `CheckpointManager` | class | Saves `.pt` files to a directory, tags "stable" ones, prunes the rest to `max_checkpoints`. |
| `CheckpointManager.save_checkpoint` | method | Writes `checkpoint_{step}.pt` with model state + stats, returns path string. |
| `CheckpointManager.load_last_stable` | method | Loads newest stable checkpoint into the model, returns its data or `None`. |
| `CheckpointManager.load_best_checkpoint` | method | Loads the checkpoint minimizing `metric` (default `'loss'`); returns data or `None`. |
| `EarlyStopping` | class | Patience counter (default 5) plus a volatility trigger over the last 10 losses. |
| `EarlyStopping.check` / `.reset` | method | `check` returns stop-flag; `reset` clears counter, best loss, history. |
| `DevelopmentalTrainer` | class | Wires child/memory/emotion/curriculum/mother/metacognition + config into a training loop. |
| `DevelopmentalTrainer.training_step` | method | One optimizer step; returns stats dict, or an error dict on any handled failure. |
| `DevelopmentalTrainer.train_episode` | method | Runs up to `num_steps` steps, returns `summarize_episode` output. |
| `DevelopmentalTrainer.update_loss_weights` | method | Blends stage weights with inverse-mean-loss weights under momentum 0.95. |

## Key behaviour

- Config defaults (training_system.py:212-222): device `'cuda'`, lr 3e-4, weight decay 0.01, grad clip 1.0, warmup 1000, checkpoint interval 100, MA window 50, patience 5, spike threshold 2.0.
- Scheduler is `CosineAnnealingWarmRestarts(T_0=warmup_steps, T_mult=2)` — `warmup_steps` is used as the restart period, not a warmup (training_system.py:228).
- Initial loss weights are fixed at moral .3 / attachment .3 / emotional .2 / cognitive .2 (training_system.py:238) and updated each step by EMA with momentum 0.95 toward `0.5*base + 0.5*inverse_mean` (training_system.py:250-252).
- `training_step` control flow:

```
zero_grad
  -> curriculum.get_stage_requirements()
  -> child(stimulus)                [RuntimeError -> log 'forward_pass' -> error dict]
  -> memory.retrieve_memories + emotional_regulation.regulate
                                    [RuntimeError -> log 'emotional_regulation' -> error dict]
  -> _compute_losses(4 terms)       [RuntimeError -> log 'loss_computation' -> error dict]
  -> update_loss_weights            (NOT guarded)
  -> total_loss = sum(w_k * L_k)
  -> check_loss_spike               -> error dict if spiking (before backward)
  -> backward + clip_grad_norm_ + check_gradient_issues
                                    [RuntimeError -> log 'backward_pass' -> error dict]
  -> optimizer.step, scheduler.step
  -> memory.record_experience
  -> curriculum.update_stage({success_rate=1-loss, emotional_stability, cognitive_complexity, social_awareness})
  -> monitoring.update_stats -> checkpoint every 100 successful steps
```

- Any error path calls `_handle_training_error`, which tries `load_last_stable()`; success returns status `rolled_back` (episode continues), failure returns status `failed` (episode breaks) — training_system.py:331-336, 344-348.
- `monitoring.steps` only increments inside `update_stats`, so the checkpoint interval counts *successful* steps, not attempted ones.
- Checkpoint retention: entries in `stable_checkpoints` are protected and never pruned; only non-stable ones are deleted oldest-first (training_system.py:140-150). Stability requires `loss_std <= 0.5`, `1e-7 <= grad_ma <= 10.0`, and every component std `<= 0.5`.
- `EarlyStopping.check` returns False until `patience` losses are seen; from 10 losses onward it stops whenever `recent_std > recent_mean * 0.5`, independent of the patience counter (training_system.py:191-196).

## Imports

Third-party: `torch`, `torch.nn`, `numpy`. Standard library: `time`, `collections.deque`, `collections.defaultdict`, `pathlib.Path`. No project-internal imports — every collaborator (child model, memory, emotional regulation, curriculum, mother LLM, metacognition) is injected through `DevelopmentalTrainer.__init__`.

## Defects and gaps

- training_system.py:245-246 — `update_loss_weights` receives the raw loss dict from `_compute_losses` (training_system.py:280), whose values are the `nn.MSELoss()` output tensors (training_system.py:324-329) built from `self.child`'s parameters, which AdamW optimizes (training_system.py:223-227) and therefore require grad. Buffering those tensors and calling `np.mean` on the deque raises (`RuntimeError` for grad-tracking tensors, `TypeError` for CUDA tensors). The call at training_system.py:280 sits outside every `try`, so the failure escapes `training_step` entirely.
- training_system.py:254 — `_compute_stage_weights` ignores `stage_requirements` and returns the same hardcoded dict as `_initialize_loss_weights`; the "stage-adaptive" half of the weighting is a constant.
- training_system.py:246 — the `'std'` entry of `loss_stats` is computed and never read.
- training_system.py:146-147 — if the number of protected (stable) checkpoints alone exceeds `max_checkpoints`, `remaining` empties while the loop condition stays true, and `remaining.pop(0)` raises `IndexError`. `stable_checkpoints` is appended to (training_system.py:125) but never pruned, so this state is reachable and permanent.
- training_system.py:155 — `max(..., key=lambda p: p.stat().st_mtime)` calls `stat()` on every stable path *before* the `exists()` guard on the next line; a missing file raises `FileNotFoundError` instead of returning `None`.
- training_system.py:165 — `load_best_checkpoint` indexes history entries by `metric`; only `'loss'`, `'step'`, `'timestamp'`, `'path'` exist, so any other metric is a `KeyError`. The method is never called anywhere in this file.
- training_system.py:198 — `EarlyStopping.reset` is never called in this file; `train_episode` constructs a fresh instance per episode instead.
- training_system.py:184-190 — on the call that first sets `best_loss`, only the seeding branch runs, so `counter` is neither reset nor incremented; `best_loss` is seeded from whatever loss arrives on the `patience`-th call, not from the minimum of the history already buffered.
- training_system.py:194 — the volatility stop compares `std > mean * 0.5`; for a mean near zero or negative this fires (or never fires) regardless of actual instability.
- training_system.py:63-64 — `check_gradient_issues` treats `grad_norm < 1e-7` as an issue, so a genuinely converged step triggers an error/rollback. The check sits after the `len(grad_buffer) < 2` early return (training_system.py:59-60), so it is silently skipped for the first two steps.
- training_system.py:52-54 — `loss_spike_threshold` (default 2.0) is used both as a multiplicative ratio and as a z-score cutoff depending on whether std is zero; the two meanings are not comparable.
- training_system.py:260-291 — only `RuntimeError` is caught. `stage_requirements['metrics']` (training_system.py:318, 321) raises `KeyError`, and missing model attributes (`self.child.morality`, `.emotional_state`, `.last_attachment_trust`) raise `AttributeError`, both uncaught.
- training_system.py:302 — stage transitions are appended to `monitoring.error_log` with a `transition` key instead of the `type`/`message` schema used by `log_error`, and are surfaced as `error_log` by `summarize_episode`.
- training_system.py:234 — checkpoint directory is hardcoded to the relative path `'checkpoints'` and `max_checkpoints=5`; neither is read from `config`, unlike every other tunable.
- training_system.py:286, 306 — `grad_norm` is the tensor returned by `clip_grad_norm_` and is passed unconverted into `check_gradient_issues` and `update_stats`, so `grad_norm`/`grad_ma` in the stats dict and in checkpoints are tensors, and the `np.isnan`/`np.mean` calls on them (training_system.py:57, 61) raise `TypeError` whenever the parameters live on CUDA — the configured default device.
- training_system.py:296 — `success_rate` is computed as `1.0 - loss`, which is unbounded below and negative for any loss > 1.
- training_system.py:317 — the emotional target defaults to a hardcoded 4-element vector; changing the emotion dimensionality silently changes MSE broadcasting rather than failing.
- training_system.py:157, 167 — `torch.load` is called without `weights_only`. `save_checkpoint` persists only `model_state` (training_system.py:110-116) and the load paths restore only that, so a rollback rewinds weights while the optimizer moments and LR schedule keep advancing.
- training_system.py:298, 322 — `self.metacognition(child_output)` is invoked twice per step (once for the cognitive loss, once for the curriculum metric) on the same input; the second call is a redundant forward pass.
- training_system.py:342-343 — `train_episode` passes the single dict returned by `mother.generate_stimulus` as both the stimulus source (`['embedding']`) and the `mother_response` supplying `reward_score`, `emotional_context.trust`, and `emotional_vector`. The supervision targets are therefore produced before the child's forward pass and cannot be a response to its output.

## Notes

- `self.device` defaults to `'cuda'` and is only used to place freshly created target tensors; the model itself is never moved to the device here.
- `MovingAverageMonitor.component_losses` is a `defaultdict` keyed by whatever `individual_losses` supplies, so component keys are not validated against `loss_weights`.
- This file reads `reward_score`, `emotional_context.trust`, `emotional_vector` and `embedding` off the mother dict, and `stage_requirements['metrics']['emotional_stability' | 'language_complexity']` off the curriculum. Whether the injected collaborators actually supply those keys, and what the child/memory/metacognition modules return, is unverifiable from this file alone.
