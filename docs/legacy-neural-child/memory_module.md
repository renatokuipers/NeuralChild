# memory_module.py

Torch-based episodic memory store for the neural-child agent. Holds a bounded short-term deque, a tiny working-memory deque, and a list of centroid-based long-term clusters, plus three small MLPs (encoder, importance scorer, consolidation head) and a delegated prioritised replay buffer. It is an `nn.Module` with no `forward`; all use is through explicit methods.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `MemoryCluster` | class | Centroid + list of `(encoded, importance)` tuples, an `importance` float, and a `last_accessed` timestamp. |
| `MemoryCluster.add_memory` | method | Appends an item and recomputes the centroid as the mean of every item's element `[0]`. |
| `MemoryCluster.get_age` | method | Seconds since `last_accessed`. |
| `DifferentiableMemory` | class (nn.Module) | Whole memory subsystem; ctor args `embedding_dim=768`, `short_term_capacity=1000`, `long_term_capacity=50000`. |
| `.compute_memory_importance` | method | Concatenates a memory entry with a 4-d emotion vector, runs `importance_net`, multiplies by the dot product of emotion and `emotional_importance`. Returns a Python float. |
| `.find_similar_cluster` | method | Cosine similarity of a 128-d encoding against every cluster centroid; returns best cluster and its float score, or `(None, 0)`. |
| `.consolidate_memory` | method | Encodes, then either merges into the best cluster or creates a new one, evicting the lowest `importance * exp(-age/86400)` cluster when full. |
| `.record_experience` | method | Builds the concatenated entry, pushes to short-term, scores importance, conditionally pushes to working memory, forwards to the replay optimizer, returns the importance. |
| `.forget_memories` | method | Bernoulli-drops short-term entries by `forgetting_rate`; probabilistically drops long-term clusters by importance x age decay. |
| `.replay_consolidation` | method | Samples a replay batch, computes per-sample MSE between encoder and consolidation outputs, updates replay weights, forgets, returns mean loss. |
| `.retrieve_memories` | method | Encodes a cue, top-k cluster match, flattens their stored encodings, truncates to `top_k`. |

Learnable scalars/vectors: `forgetting_rate` (0.1), `consolidation_threshold` (0.7), `emotional_importance` (ones(4)).

## Key behaviour

- The memory entry is a positional concatenation of input vector, internal state, reward, timestamp and emotion (memory_module.py:114-120). Nothing in the file enforces its width; the 902-float layout (`[0:768]` input, `[768:896]` internal state, `[896]` reward, `[897]` timestamp, `[898:902]` emotion) is only *implied* by the hardcoded `906` importance-net input (memory_module.py:46) and the replay slicing (memory_module.py:156-158).
- `record_experience` calls `.squeeze(0)` on the input vector and internal state (memory_module.py:115-116), so callers are expected to pass batched `(1, N)` tensors.
- Network shapes: encoder `embedding_dim -> 256 -> 128`; `importance_net` `906 -> 64 -> 1` with sigmoid; `consolidation_net` `128 -> 256 -> 128` (memory_module.py:38-57).
- `max_clusters` is `long_term_capacity // 100` = 500 at defaults (memory_module.py:35). Working memory holds at most 10 entries (memory_module.py:33).
- Cluster ageing uses a hardcoded 86400-second decay constant in both eviction and forgetting (memory_module.py:98, memory_module.py:143).
- In `replay_consolidation` only the per-sample loop is under `torch.no_grad()` (memory_module.py:154-164); `update_weights` and `forget_memories` run outside it (memory_module.py:165-166). Forgetting is unconditionally coupled to replay.

```
record_experience ──> short_term_memory (deque, maxlen=1000)
        │                    │
        │  importance>0.8    └──> forget_memories() drops by forgetting_rate
        ├──> working_memory (maxlen=10)   [never read in this file]
        └──> replay_optimizer.add_experience

replay_consolidation ──> sample_batch ──> MSE(encoder(x[:768]), consolidation_net(x[768:896]))
        │                                        │
        │  only if emotional_state is not None   └──> update_weights + forget_memories
        └──> consolidate_memory(full 902-d sample)  <-- shape error, see defects

retrieve_memories ──> encoder(cue) ──> cosine vs centroids ──> topk ──> flatten cluster items
```

## Imports

Third-party: `torch`, `torch.nn`. Standard library: `collections.deque`, `random`, `time`, `math`, `typing` (`Dict`, `List`, `Tuple`). Project-internal: `ReplayOptimizer` from `replay_system`.

## Defects and gaps

- `consolidate_memory` receives the full 902-d sample from `replay_consolidation` (memory_module.py:164) but immediately feeds it to `self.encoder`, whose first Linear expects 768 (memory_module.py:86, memory_module.py:39). Every such consolidation raises a shape error, and because the call is not guarded the exception propagates out of `replay_consolidation` entirely.
- Within this file, `consolidate_memory` is called only from `replay_consolidation` and only when a non-default `emotional_state` is supplied (memory_module.py:161-164). Under the default `None`, `long_term_clusters` is never populated here, so `retrieve_memories` returns `[]` and the long-term half of `forget_memories` is a no-op. Whether other modules call `consolidate_memory` directly is unverifiable from this file alone.
- `forget_memories` rebuilds `self.short_term_memory` as a plain `deque(...)` with no `maxlen` (memory_module.py:137). After the first call the `short_term_capacity` bound is permanently lost and short-term memory grows without limit.
- `retrieve_memories` calls `torch.topk(..., top_k)` without clamping to the number of clusters (memory_module.py:175); with fewer than `top_k` (default 5) clusters this raises.
- `retrieve_memories` returns the *encoded* 128-d cluster items (memory_module.py:178), not the original memory entries, so nothing recovers a stored experience.
- `retrieve_memories` extends the result with every memory of each selected cluster in rank order, then truncates to `top_k` (memory_module.py:177-179). Once the best cluster holds `top_k` items the remaining clusters never contribute, making the top-k cluster search effectively single-cluster.
- Neither `consolidate_memory` (memory_module.py:85-86) nor `retrieve_memories` (memory_module.py:169-172) moves its input to `self.device`, unlike the other entry points; a CPU tensor passed to either fails against the CUDA-resident encoder.
- `record_experience` normalises `emotional_state` dimensionality when building the entry (memory_module.py:119) but passes the un-normalised tensor to `compute_memory_importance` (memory_module.py:123), whose `torch.cat` (memory_module.py:72) fails if the caller supplies a batched `(1,4)` emotion.
- `consolidation_threshold` is used with two incompatible meanings: a cosine-similarity cutoff (memory_module.py:88) and an importance cutoff (memory_module.py:163). One learnable scalar cannot serve both. Both comparisons also pit a Python float against a `nn.Parameter`, yielding a 0-d tensor that only works via implicit truthiness.
- All three `nn.Parameter`s and all three MLPs are untrainable from this file: `compute_memory_importance` returns `.item()` floats (memory_module.py:75), the replay loop is under `no_grad`, and there is no optimizer or `backward` call anywhere in the file.
- `importance_net`'s input width 906 is hardcoded (memory_module.py:46), as is the `sample[:768]` / `sample[768:896]` slicing (memory_module.py:156-158). Constructing with any `embedding_dim` other than 768 silently breaks both.
- `MemoryCluster.add_memory` is annotated as taking a `torch.Tensor` (memory_module.py:18) but every call site passes a `(tensor, float)` tuple (memory_module.py:89, 95, 101) and line 20 indexes `m[0]`. The annotation contradicts the code.
- `MemoryCluster.add_memory` restacks and re-averages the cluster's entire memory list on every insert (memory_module.py:20) instead of updating the mean incrementally.
- `DifferentiableMemory` subclasses `nn.Module` but defines no `forward`; calling the instance raises `NotImplementedError`.
- `Dict` is imported (memory_module.py:7) and never used.
- `working_memory` (memory_module.py:33) is written at memory_module.py:126 and never read in this file.
- `last_accessed` is only refreshed on the cluster-merge path (memory_module.py:91); `retrieve_memories` does not touch it, so recall does not protect a cluster from age-based eviction.
- `MemoryCluster.importance` starts at 1.0 and only ever grows via `max` (memory_module.py:15, memory_module.py:90), while `age_factor` is `exp(-age/86400)`. The drop probability `1 - importance * age_factor` (memory_module.py:144) is therefore near zero for fresh clusters, and exactly zero whenever `importance >= exp(age/86400)` — which is reachable because `compute_memory_importance` is unbounded above (sigmoid output times an unconstrained emotional weight, memory_module.py:74-75).
- `replay_consolidation` does not guard against an empty batch; `torch.tensor([]).mean()` returns NaN (memory_module.py:167).
- `find_similar_cluster` builds `max((s, i) ...)` over 0-d tensors (memory_module.py:82); it works via implicit tensor-to-bool coercion but is fragile, and on exact ties the index comparison silently selects the last cluster.

## Notes

- `ReplayOptimizer`'s constructor kwarg and the `add_experience` / `sample_batch` / `update_weights` contracts are assumed by this file but unverifiable from it alone. Whether `replay_optimizer` is itself moved by `self.to(self.device)` (memory_module.py:64) likewise depends on `replay_system`.
- Device handling is partial: `compute_memory_importance` (memory_module.py:69-70) and `record_experience` (memory_module.py:107-111) re-`.to(self.device)` their inputs; the other public methods do not.
- The magic threshold `0.8` gating working memory (memory_module.py:125) is unrelated to any parameter and cannot be tuned.
