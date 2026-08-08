# emotional_regulation.py

Two independent pieces of affect machinery in one module. `EmotionalState` is a plain (non-`nn.Module`) container holding four scalar `nn.Parameter` emotions plus a fixed table of eight "complex" emotions defined as weighted mixtures of those four. `EmotionalRegulation` is an `nn.Module` that runs an LSTM over a short history of emotion vectors and emits a regulated 4-dim state. The two classes never reference each other inside this file.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `EmotionalState` | class | Holds joy/trust/fear/surprise as scalar parameters; ctor takes `device` (defaults to the literal `'cuda'`). |
| `EmotionalState.update` | method | Moves each named primary toward a target by `learning_rate`, adds noise, appends a volatility sample. Returns None. |
| `EmotionalState.get_complex_emotion` | method | Weighted sum of primaries per the mixture table; returns 0.0 for unknown names. |
| `EmotionalState.get_dominant_emotion` | method | Returns a `(name, intensity)` tuple over the union of 4 primary + 8 complex intensities. |
| `EmotionalState.get_emotional_stability` | method | `1 - min(mean(stability_window), 1.0)`; returns 1.0 when the window is empty. |
| `EmotionalState.to_tensor` / `from_tensor` | methods | Serialize/deserialize the 4 primaries in **alphabetical** key order (fear, joy, surprise, trust). |
| `EmotionalRegulation` | class (`nn.Module`) | Ctor args `emotion_dim=4`, `context_window=5`, `memory_dim=32`; builds LSTM + two MLP heads. |
| `EmotionalRegulation.regulate` | method | Main forward path; returns dict with `emotional_state`, `context_influence`, `memory_influence`. |
| `EmotionalRegulation.update_baseline` | method | EMA (alpha 0.1) of history mean into `self.baseline`. |
| `EmotionalRegulation.detect_trauma` | method | Dict with `is_traumatic`, `duration`, `intensity` from L2 distance to baseline. |
| `EmotionalRegulation.compute_regulation_strength` | method | `sigmoid(|state - baseline| * resilience)`. |

## Key behaviour

- Mixture table (emotional_regulation.py:18-27) is hardcoded and every entry's weights sum to exactly 1.0. Names are psychologically arbitrary relative to their components (`pride` = 0.7 joy + 0.3 fear; `shame` = 0.6 trust + 0.4 surprise).
- `update` (:32-42) writes through `.data`, so no gradient ever flows; the `nn.Parameter` wrapper buys nothing here. Noise amplitude is fixed at 0.05 (:37).
- Volatility sample is `sum(|primary - 0.5|)` over 4 emotions, pushed into a `deque(maxlen=100)` (:29, :41). Range 0..2, so `get_emotional_stability` saturates to 0.0 whenever mean deviation exceeds 0.25 per emotion.
- `EmotionalRegulation` tensor shapes with defaults: history entries `(4,)` → stacked `(T,4)` → unsqueezed `(1,T,4)` → LSTM out `(1,T,8)` → last step `(8,)`. Concatenated with memory `(32,)` → `(40,)` into `stability_net` (Linear 40→256, LayerNorm, GELU, Linear 256→4).
- History is `deque(maxlen=5)`; the LSTM branch requires ≥2 entries, so the first two `regulate` calls fall through to a zero context embedding (:134-139).
- `regulate` appends the **input** `emotional_state.detach()`, not the produced `new_state` (:151).

```
regulate(emotional_state, stimulus, memory_context)
  |
  +- len(history) >= 2 ? LSTM(history) -> ctx(8)  : zeros(8)
  +- memory_context given ? ctx = ctx(8) * memory_gate(...)(4)   <-- shape error
  +- combined = cat[ctx(8), memory or zeros(32)] -> (40,)
  +- stability_net -> clamp(0,1) -> new_state(4)
  +- history.append(emotional_state)        # input, not output
  '- return {emotional_state, context_influence, memory_influence}
```

## Imports

Third-party: `torch`, `torch.nn as nn`. Standard library: `collections.deque`. No project-internal imports.

## Defects and gaps

- **`update` cannot run.** At :35-38 `current`, `delta` and `noise` are all Python floats (each via `.item()`), so `torch.clamp` receives a float as its `input` argument rather than a Tensor and raises `TypeError`. Every call to `EmotionalState.update` fails on the first matching emotion key.
- **`get_dominant_emotion` can never return a complex emotion** (:54-59). Each mixture's weights sum to 1.0, so a complex intensity is a convex combination of primaries and is always ≤ the max primary. Ties go to the primary because primaries are merged first into `all_emotions` and `max` keeps the first maximum.
- **Baseline mismatch in `EmotionalState`** (:12-15 vs :30): primaries start at 0.0 but the baseline dict is 0.5, and that dict is never recomputed anywhere in the class. Since `update` raises before reaching :41 whenever a key matches, the only way a volatility sample is ever recorded is an `update` call whose keys all miss — which appends `sum(|0.0 - 0.5|) = 2.0` and pins `get_emotional_stability` at 0.0.
- **Hardcoded `'cuda'`** at :8 (default arg) and :110 (`self.baseline = torch.zeros(..., device='cuda')`). Line 110 executes *before* the availability check on :111, so `EmotionalRegulation()` raises on a CPU-only or non-CUDA build even though the very next line handles that case. `EmotionalState` has the same problem with no fallback at all: its parameters are built on the ctor `device` (:12-15), defaulting to `'cuda'`.
- **`EmotionalRegulation` defines no `forward`** (:76-156) despite subclassing `nn.Module`, so calling an instance raises `NotImplementedError`; the only entry point is the plain `regulate` method, which bypasses module hooks.
- **`self.baseline` is a plain attribute, not a registered buffer** (:110). `self.to(self.device)` at :112 does not move it, and it is absent from `state_dict`.
- **Memory branch is shape-broken** (:141-142). `memory_gate` outputs `emotion_dim` (4) while `context_embedding` is `emotion_dim * 2` (8); the elementwise multiply is not broadcastable and raises `RuntimeError` whenever `memory_context` is passed.
- **`stimulus` is accepted and never used** (:133).
- **Regulation output ignores the current emotional state.** `new_state` derives only from `context_embedding` and `memory_context` (:145-150); the `emotional_state` argument reaches this call's output only via the (broken) memory-gate multiply — otherwise it just enters the history deque for later calls (:151).
- **Uncalled within this file:** the sole intra-file method call is `get_dominant_emotion` → `get_complex_emotion` (:56). `update_baseline`, `detect_trauma`, `compute_regulation_strength`, `to_tensor`, `from_tensor`, `get_dominant_emotion` and `get_emotional_stability` have no caller here. In particular `regulate` never calls `update_baseline`, so on this file's own code path `self.baseline` stays all-zeros and `detect_trauma`/`compute_regulation_strength` measure raw magnitude rather than deviation.
- **Magic constants:** trauma threshold 1.0 and resilience 1.0 (:84-85, introduced under a comment reading "Define the missing parameters"), the 0.7 duration cutoff (:122), noise 0.05 (:37), EMA alpha 0.1 (:117).
- `detect_trauma` returns `is_traumatic` as a 0-dim bool Tensor, not a Python bool (:124), while the sibling `intensity` is `.item()`-ed to a float.
- `compute_regulation_strength` applies sigmoid to a non-negative absolute deviation (:130-131), so the returned "strength" is bounded to [0.5, 1.0) and can never fall below 0.5.
- `from_tensor` assigns `tensor[i]` straight into `.data` (:74) with no dtype/device/length validation; the parameter then aliases the caller's storage.

## Notes

- `nn.ParameterDict` is constructed inside a non-`nn.Module` class (:11), so the four parameters are not discoverable by any optimizer or `state_dict` traversal from an owning module.
- `to_tensor`/`from_tensor` sort keys alphabetically, which differs from the insertion order used everywhere else (joy, trust, fear, surprise). The pair is self-consistent, but any external consumer indexing the tensor must use alphabetical order.
- The `context_window` ctor arg only sizes the history deque; the LSTM itself is length-agnostic.
- Whether other modules depend on these methods is unverifiable from this file alone.
