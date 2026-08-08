# moral_network.py

A single-class PyTorch module (`MoralPolicyNetwork`) that maps a thought embedding to a scalar in [-1, 1] plus the intermediate gated encoding. It also carries a hinge-loss helper for contrastive training on positive/negative example batches. Everything in the file is 66 lines: one `nn.Module` subclass with `forward` and `reinforce`.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `MoralPolicyNetwork` | class (`nn.Module`) | Constructed with `input_dim=128`, `device=torch.device('cuda')`; builds three MLP stacks plus a 2-entry `ParameterDict`, then calls `self.to(device)`. |
| `MoralPolicyNetwork.forward` | method | Takes `thought_embedding` tensor of last-dim `input_dim`; returns dict with `moral_score` (last dim 1, tanh-bounded) and `constraint_applied` (last dim 256). |
| `MoralPolicyNetwork.reinforce` | method | Runs two forward passes and returns a scalar margin-1 hinge loss tensor. Does not backprop, step, or zero any optimizer. |

## Key behaviour

- Tensor pipeline (moral_network.py:11-31, 44-55), batch dims preserved:

```
thought_embedding (..., input_dim=128)
  -> input_projection: Linear 128->256, LayerNorm(256), GELU        (..., 256)
  -> ethical_encoder: Linear 256->512, LayerNorm(512), GELU,
                      Linear 512->256                                (..., 256)
  -> for each of 2 safety filters: encoded *= sigmoid(param[256])    (..., 256)
  -> value_head: Linear 256->128, GELU, Linear 128->1, Tanh          (..., 1)
```

- `safety_filters` is an `nn.ParameterDict` with keys `self_preservation` and `social_norms`, each a learnable `randn(256)` vector (moral_network.py:34-37). The forward loop iterates `.values()` only — keys are never read, so the two names have no behavioural meaning and the pair is mathematically equivalent to one gate vector whose elementwise value is `sigmoid(a) * sigmoid(b)`.
- The "safety filter" is a plain elementwise multiplicative gate. There is no thresholding, veto, clamping, rejection path, or branch on the score — nothing can refuse or short-circuit an output.
- Each gate is `sigmoid` of `randn`, so every element lies in (0, 1): the gates can only attenuate the encoding, never amplify it or flip a sign. At initialisation the two gates together scale the encoding by ~0.25 on average (~0.5 each), so the value head sees a systematically attenuated signal before any training.
- `reinforce` (moral_network.py:62-65) computes elementwise `relu(1 - pos + neg)` then `.mean()`. Both score terms come from `Tanh`, so the pre-relu term lies in [-1, 3]; loss reaches 0 only when `pos - neg >= 1`, i.e. the two scores must be separated by half the full tanh output range. Loss at identical inputs is exactly 1.0.
- Constructor places `safety_filters` on `self.device` at creation and then calls `self.to(self.device)` for the rest (moral_network.py:35-40).
- Stateless across calls apart from parameters: no buffers, no counters, no `train()`/`eval()`-sensitive layers (no dropout, no batchnorm), so output is deterministic for fixed weights and input.

## Imports

- Third-party: `torch`, `torch.nn` (as `nn`).
- Standard library: `typing.Dict`.
- Project-internal: none.

All three imports are used.

## Defects and gaps

- moral_network.py:6 — `device=torch.device('cuda')` is a default argument evaluated at class-definition (import) time. On a machine without CUDA this constructs fine but every `torch.randn(..., device=...)` and `self.to(...)` at moral_network.py:35-40 raises at instantiation. There is no `cuda.is_available()` check and no CPU fallback.
- moral_network.py:49-50 — comment asserts the parameters "will be on the same device (cuda) as encoded". That is only true when the caller leaves the default; the comment contradicts the configurable `device` parameter.
- moral_network.py:62 — `reinforce` is defined but never referenced anywhere in this file. It has no type hints (unlike `forward`), returns a loss with no optimizer/backward wiring, and its callers are unverifiable from this file alone.
- moral_network.py:65 — the hinge margin `1` is hardcoded and coupled to the `Tanh` range at moral_network.py:30. Removing or replacing the `Tanh` silently changes the loss semantics; nothing links the two.
- moral_network.py:19-37 — widths 256 and 512 are hardcoded while only `input_dim` is configurable. The 256 in `safety_filters` must match the `ethical_encoder` output; there is no assertion tying them together.
- moral_network.py:59 — the returned key `constraint_applied` holds the 256-dim gated encoding, not any indication of whether a constraint was applied. The name promises a flag/decision; the value is a feature tensor.
- moral_network.py:34-37 — two independently-parameterised gates are redundant: their product is reparameterisable as a single vector, so the pair adds 256 parameters with no representational gain and no way to attribute a score to either named concept.
- No input validation: a `thought_embedding` whose last dim is not `input_dim` fails inside `nn.Linear` with a shape error rather than a domain-level message.
- `reinforce` performs no shape or batch-size agreement check between `positive_examples` and `negative_examples`. Scores carry a trailing dim of 1, so `1 - pos_scores + neg_scores` (moral_network.py:65) broadcasts silently when one side has batch size 1 (pairing every negative against a single positive), and otherwise raises a raw tensor shape error. Neither path reports a domain-level mismatch.

## Notes

- The module is self-contained: no config objects, no logging, no persistence, no external project imports. It can be exercised standalone given a tensor of the right last dimension.
- Nothing in this file names or enumerates a moral/ethical taxonomy beyond the two unused `ParameterDict` keys; the "moral" semantics are entirely whatever the training signal supplies through `reinforce`.
