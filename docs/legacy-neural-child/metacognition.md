# metacognition.py

A single 66-line PyTorch module defining `MetacognitionSystem`, a small MLP stack that maps a thought embedding to a refined thought plus three self-assessment signals (confidence, uncertainty, complexity). It also exposes a hypothesis-sampling routine that perturbs the input with noise, scores the variants with a critic head, and returns the "best" one. The file is self-contained: no project imports, no instantiation, no training loop, no configuration.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `MetacognitionSystem` | class (`nn.Module`) | Constructed with `base_dim=128`, `num_hypotheses=5`; builds five sub-networks and moves itself to CUDA when available (metacognition.py:4-34). |
| `MetacognitionSystem.forward` | method | Takes one embedding tensor, returns a dict with keys `thought`, `confidence`, `uncertainty`, `complexity` (metacognition.py:36-48). |
| `MetacognitionSystem.self_correct` | method | Takes a thought embedding and `temperature=0.7`, returns one alternative embedding chosen by critic score (metacognition.py:50-65). |

## Key behaviour

Sub-network shapes, all derived from `base_dim` (default 128):

| Attribute | Structure | In → Out |
| --- | --- | --- |
| `base_network` | Linear, GELU, Linear | 128 → 256 → 128 |
| `hypothesis_network` | `ModuleList` of 5 identical Linear/GELU/Linear blocks | 128 → 256 → 128 each |
| `critic` | Linear, GELU, Linear, Sigmoid | 256 → 512 → 1 |
| `bayesian_layer` | `nn.LSTM`, 1 layer, `batch_first` left at default `False` | 128 → 128 |
| `complexity_head` | Linear, GELU, Linear, Sigmoid | 128 → 64 → 1 |

Forward pass (metacognition.py:36-48):

```
input_embedding ──> base_network ──> base_output (…,128) ──> 'thought'
        │                              │
        └── cat(dim=-1) ───────────────┘
                  (…,256) ──> critic ──> 'confidence' (…,1), sigmoid in [0,1]

base_output.unsqueeze(0) ──> LSTM ──> h_n.squeeze(0) ──> 'uncertainty' (…,128), in (-1,1)
base_output ──> complexity_head ──────────────────────> 'complexity' (…,1), sigmoid in [0,1]
```

- `unsqueeze(0)` at metacognition.py:40 injects a length-1 sequence axis, so the LSTM sees `seq_len=1`. A 2-D `(B,128)` input yields hidden `(1,B,128)` → `(B,128)`; a 1-D `(128,)` input is treated as unbatched and yields `(128,)`. A 3-D input would become 4-D and the LSTM rejects it.
- `uncertainty` is the raw LSTM hidden state — a 128-vector of tanh-bounded values in (-1,1), not a scalar and not restricted to [0,1], despite the name and its downstream use as a probability-like weight.
- `self_correct` (metacognition.py:50-65) first runs a full `forward` purely to obtain `uncertainty`; `thought`, `confidence` and `complexity` from that pass are discarded.
- Noise magnitude per hypothesis is `temperature * (i+1) / num_hypotheses` (metacognition.py:55): with defaults that is 0.14, 0.28, 0.42, 0.56, 0.70 standard deviations. Hypothesis index and noise level are coupled — network `i` only ever sees noise level `i`.
- Each alternative is re-scored by the same `critic` against the original embedding (metacognition.py:58-60), so `critic` is shared between the forward confidence estimate and hypothesis ranking.
- Selection is `argmax` over `stack(scores) * (1 - base_uncertainty)` (metacognition.py:63-65).

## Imports

Third-party: `torch`, `torch.nn` (as `nn`).
Project-internal: none.

## Defects and gaps

- **Broadcast blow-up then near-certain `IndexError` (metacognition.py:63-65).** `torch.stack(scores)` has shape `(5,B,1)` (or `(5,1)` unbatched) while `1 - base_uncertainty` has shape `(B,128)` (or `(128,)`). Broadcasting produces `(5,B,128)` / `(5,128)`, not a per-hypothesis score vector. `torch.argmax` without `dim` then returns a flat index over that whole tensor — 640 elements (indices 0–639) in the unbatched case — which is used directly as `alternatives[best_idx]` into a 5-element list. Any flat index ≥ 5 raises `IndexError`; the call only survives when the argmax lands in the first five flattened positions.
- **Uncertainty used as a probability it never is (metacognition.py:63).** `base_uncertainty` is an LSTM hidden state, bounded to (-1,1), so `1 - base_uncertainty` lies in (0,2) — never a probability. It is a 128-element per-dimension multiplier, not a scalar damping factor, so it drives the broadcast above rather than reweighting hypotheses. Weights stay positive, so no sign inversion occurs.
- **Dead guard (metacognition.py:61-62).** `if not alternatives` sits after the loop and can only be true when `num_hypotheses <= 0`; with any positive value it is unreachable.
- **No input device handling (metacognition.py:33-34, 36).** The module moves itself to CUDA in `__init__`, but `forward` never moves `input_embedding` to `self.device`, so a CPU tensor against a CUDA module raises. `self.device` is also a plain attribute — a later `.to('cpu')` leaves it stale, since `nn.Module.to` does not update it.
- **Hardcoded widths (metacognition.py:9-11, 15-17, 21-23).** The 256 and 512 hidden widths do not scale with `base_dim`; a large `base_dim` bottlenecks through fixed 256-unit layers, and `base_dim // 2` at metacognition.py:28 degenerates to 0 for `base_dim == 1`.
- **`self_correct` is defined but never called anywhere in this file**; nothing in the file instantiates `MetacognitionSystem` either. Whether external callers use them is unverifiable from this file alone.
- **`complexity` has no consumer in this file** — it is produced at metacognition.py:42 and only returned; `self_correct` ignores it.
- **`forward` invoked directly rather than through `__call__` (metacognition.py:53).** `self.forward(thought_embedding)` bypasses `nn.Module` hook dispatch, so any registered forward or pre-forward hook is skipped on the `self_correct` path.
- **No training path.** No loss, optimizer, `train()`/`eval()` handling, dropout, or gradient control (`self_correct` runs under grad by default), so the critic and hypothesis heads are effectively random projections unless trained elsewhere.

## Notes

- `bayesian_layer` performs no Bayesian computation: it is a one-layer LSTM run for a single timestep with no state carried between calls, so it is a deterministic feed-forward transform of `base_output` per invocation.
- `num_hypotheses` is stored on the instance (metacognition.py:7); the `ModuleList` is built from the constructor argument (metacognition.py:18) while the noise divisor reads `self.num_hypotheses` (metacognition.py:55). They agree at construction, but reassigning the attribute afterwards silently rescales the noise without changing the number of hypothesis networks.
