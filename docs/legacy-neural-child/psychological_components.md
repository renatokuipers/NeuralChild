# psychological_components.py

Three standalone `nn.Module` subclasses modelling social/affective faculties: theory of mind, caregiver attachment, and Freudian-style defense mechanisms. Each owns its own MLPs, a bounded history deque, and one or more `nn.Parameter` tensors overwritten via `.data` outside autograd. No module instantiates or calls another; the file defines components only and has no entry point, no `__main__`, and no exports list.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `TheoryOfMind` | class (nn.Module) | Maps a 398-dim social context vector to four perspective-taking heads. |
| `TheoryOfMind.forward` | method | Returns dict with `emotional` (4), `belief` (64), `intention` (32), `attention` (16). |
| `TheoryOfMind.update_relationship_model` | method | Appends `(interaction, outcome)`; rewrites `social_bias` once history ≥100. |
| `AttachmentSystem` | class (nn.Module) | Maps a 4-dim caregiver vector to trust/attachment-style/bonding outputs. |
| `AttachmentSystem.forward` | method | Returns dict `trust_level`, `attachment_style` (4), `bonding_features` (64). |
| `AttachmentSystem.update_attachment` | method | Appends a quality float; nudges attachment style once history ≥100. |
| `DefenseMechanisms` | class (nn.Module) | Gates seven defense heads behind a learnable anxiety threshold. |
| `DefenseMechanisms.forward` | method | Returns `active_defense` (name or None), `defense_strength`, `all_mechanisms`. |
| `DefenseMechanisms.update_threshold` | method | Scales `anxiety_threshold` by stress, clamped to [0.3, 0.9]. |

## Key behaviour

- All three constructors default `device='cuda'` and call `.to(device)` on submodules; parameters are created with `device=device` directly.
- TheoryOfMind trunk: 398 → 512 → LayerNorm → GELU → Dropout(0.1) → 256 → LayerNorm → GELU → 128 (psychological_components.py:14-23). The 398 is hardcoded, justified by a comment claiming 128 base + 256 sensory + 10 drives + 4 emotional.
- Heads apply different squashes: sigmoid on `emotional` and `attention`, tanh on `belief`, softmax(dim=-1) on `intention` (psychological_components.py:42-45). `emotional` is then multiplied by `social_bias` (psychological_components.py:47), which is initialised to `ones(4)` and so is a no-op until `update_relationship_model` first rewrites it; thereafter the rewrite rule confines it to (0,1), making the multiplication a pure attenuation of the emotional head.
- 1-D input is unsqueezed to batch-of-1 (psychological_components.py:37-38); no other shape validation anywhere in the file.
- `social_bias` recomputes as `sigmoid(mean(last 100 outcomes) * social_bias)` on every `update_relationship_model` call once the deque reaches 100 (psychological_components.py:52-54). Since the deque is only ever appended to, the gate stays open once passed, and the repeatedly applied sigmoid contracts toward a fixed point rather than accumulating a signal.
- AttachmentSystem: two parallel MLPs from 4 dims — trust (4→256→1→Sigmoid) and bonding (4→256→64). `trust_level` is an EMA with fixed coefficients 0.95/0.05 written to `.data` each forward (psychological_components.py:84). `attachment_styles` (init `[0.7, 0.1, 0.1, 0.1]`) is scaled by trust then softmaxed again at output.
- `update_attachment` bumps index 0 (secure) by 1.01 when mean quality > 0.8, index 3 by 1.01 when < 0.3, then re-softmaxes the whole vector.
- DefenseMechanisms trunk: 398 → 256 → LayerNorm → GELU → Dropout(0.1) → 128 → LayerNorm → GELU, feeding seven independent `Linear(128, 1)` heads (repression, projection, denial, sublimation, rationalization, displacement, regression). Only the arg-max head is reported as active.

```
anxiety_level > anxiety_threshold (init 0.7, clamped 0.3–0.9)
        │ true                                  │ false
        ▼                                       ▼
 mechanism_strength(emotional_input:398)   active_defense = None
        ▼                                  strength = tensor(0.0)
 7 × sigmoid heads → argmax by .item()     all_mechanisms = 0-d zeros
```

## Imports

Third-party: `torch`, `torch.nn` as `nn`. Standard library: `collections.deque`. No project-internal imports.

## Defects and gaps

- psychological_components.py:47 — `predictions['emotional'] *= self.social_bias` mutates the output of `torch.sigmoid` in place. Autograd needs that output for sigmoid's backward, so a backward pass through this branch raises a modified-by-inplace-operation error.
- psychological_components.py:84 — `self.trust_level.data = 0.95 * ... + 0.05 * trust_prediction` assigns a shape `(B, 1)` tensor onto a 0-dim `nn.Parameter`, silently resizing it. With B > 1, line 85 then broadcasts to `(B, 4)` and `softmax(dim=0)` normalizes across the batch instead of across the four attachment styles. The resize is permanent, so a later forward with a different batch size B' hits a broadcast failure at line 84 itself.
- psychological_components.py:86-90 — returns the live `self.trust_level` Parameter object, so any caller holding it observes later mutations rather than a snapshot.
- psychological_components.py:95 — variable named `recent_quality` uses the entire deque (up to 1000 entries), not a recent window; contradicts the name and the `>= 100` gate's intent.
- psychological_components.py:101 — softmax applied to an already-softmaxed vector on every call, flattening the distribution toward uniform and washing out the 1.01 nudges from lines 98/100.
- psychological_components.py:129 — `anxiety_level > self.anxiety_threshold` used as a Python bool; a multi-element `anxiety_level` tensor raises an ambiguous-truth-value error.
- psychological_components.py:132 — `.item()` in the argmax key requires single-element head outputs, so batch size > 1 raises.
- psychological_components.py:133-142 — the two return branches disagree in type and shape: active returns `(B, 1)` tensors and a string name; inactive returns 0-dim tensors and `None`.
- psychological_components.py:53 — `recent_outcomes` is built with `torch.tensor(...)` and no `device`, landing on CPU while `social_bias` is on `device`. It survives only because PyTorch permits 0-dim CPU scalars in ops with CUDA tensors. `self.device` is stored at line 8 but never read anywhere in `TheoryOfMind`, including here where it was needed.
- psychological_components.py:51 — `interaction` tensors are appended undetached, so up to 1000 autograd graphs can be pinned alive by the deque.
- psychological_components.py:12, 109 — input dimension 398 is hardcoded in two places; line 109's comment ("Changed from 393 to 398") documents an edit rather than a constraint, and nothing validates the incoming width.
- psychological_components.py:145-148 — `update_threshold` multiplies the threshold by `1 + (stress - 0.5) * 0.1`, so stress held on one side of 0.5 walks it monotonically into a clamp bound (0.3 or 0.9), after which further same-direction calls are no-ops. Nothing restores the 0.7 initial value.
- psychological_components.py:6, 57, 104 — `device='cuda'` default; construction fails on a CPU-only machine unless `device` is passed explicitly at every instantiation site.
- `update_relationship_model` (:50), `update_attachment` (:92) and `update_threshold` (:144) are never referenced anywhere in this file. Whether external callers invoke them is unverifiable from this file alone.
- `relationship_memory` and `caregiving_history` are plain deques, not buffers, so they are absent from `state_dict()` and lost across save/load.

## Notes

- `social_bias`, `trust_level`, `attachment_styles` and `anxiety_threshold` are `nn.Parameter`s (so `requires_grad`) but are only ever written through `.data`, which bypasses autograd. Any gradient-based update to them would be overwritten by the next manual write. Whether an optimizer is attached is unverifiable from this file alone.
- `anxiety_threshold` is a Parameter used solely as a Python-level branch condition (psychological_components.py:129), so no gradient can flow to it.
- Dropout(0.1) sits in both trunks, so inference is nondeterministic unless the caller sets `.eval()`.
- The `bonding_features` output (64-dim) of `AttachmentSystem` and TheoryOfMind's `belief`/`intention`/`attention` heads are computed and returned but consumed nowhere within this file.
