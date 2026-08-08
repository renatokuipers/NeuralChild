# replay_system.py

Single-class module implementing a fixed-capacity experience replay buffer with per-slot "importance" scalars held in a CUDA tensor. Experiences are stored in a plain Python list; importance is decayed multiplicatively and incremented by per-sample loss, and the buffer is pruned of its lowest-importance entries when nearly full. Apart from the imports and the class definition the file contains no module-level executable code, no entry point, and never instantiates the class.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `ReplayOptimizer` | class | Replay buffer; constructor takes `memory_capacity=10000`, allocates a CUDA parameter of that length. |
| `ReplayOptimizer.add_experience` | method | Appends while under capacity, otherwise overwrites a uniformly random slot. Returns nothing. |
| `ReplayOptimizer.sample_batch` | method | Uniform random sample of up to `batch_size=32` items; returns `(samples, indices)`, `([], [])` when empty. |
| `ReplayOptimizer.update_weights` | method | Decays all weights, adds per-index losses, and prunes the buffer when it exceeds 95% capacity. |

## Key behaviour

- Constructor (replay_system.py:7-10) creates an empty list, stores `capacity`, allocates `nn.Parameter(torch.ones(capacity, device='cuda'))` and sets `decay_factor = 0.99`. `ReplayOptimizer` is a plain object, not an `nn.Module`, so the `Parameter` is never registered anywhere — it behaves as a leaf tensor with `requires_grad=True` that no optimizer sees.
- Insertion (replay_system.py:13-17) is append-until-full, then uniform random overwrite. It is not a reservoir sample and not importance-weighted.
- Sampling (replay_system.py:21-26) is uniform via `random.sample` over `range(len(self.memory))`. `importance_weights` is never consulted when sampling, so the entire importance machinery has no effect on which experiences are replayed.
- Weight update (replay_system.py:29-31) multiplies the whole weight vector by 0.99, then adds each loss into `importance_weights.data[i]`. All writes go through `.data`, bypassing autograd entirely.
- Pruning (replay_system.py:32-38) fires when `len(memory) > capacity * 0.95`. It argsorts weights ascending, takes the first `len(memory) // 20` (5%) indices, filters those positions out of the list, and reallocates `importance_weights` as a fresh `ones` tensor sized to the surviving list — discarding every accumulated importance value it just used.

```
add_experience ──► memory list (len ≤ capacity)
                        │
sample_batch ──uniform──┤──► (samples, indices)   [weights unused here]
                        │
update_weights(indices, losses)
   decay every weight, then add each loss at its index
   once list length passes 95% of capacity:
        drop lowest-5% by weight   ──► memory shrinks
        weight vector reallocated as all-ones, sized to the
        surviving list  ──► LENGTH NOW < capacity
                            (list regrows to capacity later)
```

## Imports

- Stdlib: `random`.
- Third-party: `torch`, and `nn` via `from torch import nn`.
- Project-internal: none.

## Defects and gaps

- replay_system.py:9 — hard-coded `device='cuda'`. Construction raises on any CPU-only or non-CUDA machine; there is no device argument, no `torch.cuda.is_available()` check, and no fallback.
- replay_system.py:9 — `nn.Parameter` on a non-`nn.Module` class. Nothing registers or optimizes it, and every mutation uses `.data`, so `requires_grad=True` is inert.
- replay_system.py:35-38 — after the first prune, `importance_weights` has length `len(memory)` (~95% of capacity) while `self.capacity` is unchanged. `add_experience` refills the list back to `capacity`, so a later `update_weights` indexes `importance_weights.data[i]` with `i` up to `capacity-1` and raises `IndexError`. The buffer is permanently broken after one prune-then-refill cycle.
- replay_system.py:35-38 — the reallocated tensor is all-ones, wiping the accumulated importance that the prune step was computing. Combined with the 0.99 decay, importance history is destroyed every prune.
- replay_system.py:32 — the prune guard is a bare threshold with no cooldown or counter. Once the list passes 95% of capacity (9501 at the default), *every subsequent* `update_weights` call prunes and reallocates, not one prune per fill cycle.
- replay_system.py:34 — `i not in prune_idx` performs a CUDA elementwise comparison per list element. At default capacity this is ~10,000 device-side scans of a ~500-element tensor per prune; quadratic and device-synchronizing.
- replay_system.py:17 — overwriting `memory[idx]` leaves the old slot's accumulated importance attached to a completely different experience. Importance and content silently desynchronize.
- replay_system.py:29 — decay is applied to all `capacity` entries including slots with no experience yet. While the buffer is filling, an unused slot's weight decays toward zero, so a newly appended experience inherits an artificially low importance and is a prime prune candidate.
- replay_system.py:29-31 — the `losses` argument is annotated but never validated: length, dtype, device and dimensionality are all unchecked, and pairing it with `indices` truncates to the shorter of the two, so a mismatched batch silently drops updates instead of erroring.
- replay_system.py:10, 32, 33 — magic constants `0.99`, `0.95`, and `// 20` are hardcoded and mutually coupled; changing capacity or update frequency silently changes prune cadence and decay half-life.
- replay_system.py:33 — `torch.argsort` is called on the `Parameter` itself rather than `.data`; harmless here, but inconsistent with every other access in the method.
- replay_system.py:33-38 — the prune count is an integer division of the list length by 20, and the list length never exceeds `capacity`. Configure the class with a capacity under 20 and the count is always 0: the prune branch still runs whenever the 95% guard passes, removes nothing, and resets the weights to all-ones anyway, leaving the importance mechanism inert.
- replay_system.py:35 — the attribute is rebound to a *new* `Parameter` rather than resized in place, so any external reference to the previous tensor silently goes stale.
- Nothing in this file calls `add_experience`, `sample_batch`, or `update_weights`; whether any caller exists is unverifiable from this file alone.

## Notes

- Despite the name, this class performs no prioritized replay: sampling is uniform and `importance_weights` influences only pruning. If a caller expects PER-style weighted sampling or importance-sampling correction weights, it does not exist here.
- `update_weights` returns nothing and there is no accessor for importance; a caller would have to reach into the public `importance_weights` attribute directly. Within this file the only observable effect of the weights is which entries get pruned.
- The class holds a CUDA allocation of `capacity` floats for its whole lifetime (40 KB at default capacity) regardless of how many experiences are stored.
