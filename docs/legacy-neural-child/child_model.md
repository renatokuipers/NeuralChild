# child_model.py

Defines the legacy "child" agent: a `nn.Module` (`DynamicNeuralChild`) that fuses a projected 768-d input embedding with a synthetic sensory vector, a drive/personality vector, and a 4-d emotional state, then pushes the concatenation through a decision network and a growable stack of core layers. Three plain (non-`nn.Module`) helper classes supply the sensory channels, the drive/personality state, and a confirmation-bias multiplier. Emotional updates, trauma handling, and a textual feeling readout live on the same class.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `SensoryExperience` | class (plain) | Holds four sensory `nn.Parameter` vectors + a 960→512→256 integration MLP; `process_input` returns 256-d, `update_sensitivity` nudges 4 attention weights. |
| `CoreDrives` | class (plain) | Five drives + five personality traits as scalar `nn.Parameter`s, plus a 10→64→10 regulation MLP; `get_motivation_vector` returns sigmoid 10-d. |
| `CognitiveBiases` | class (plain) | Single float `confirmation_bias_strength = 0.3`; `apply_confirmation_bias` scales evidence by `1 + 0.3 * beliefs`. |
| `DynamicNeuralChild` | class (`nn.Module`) | The agent. `forward` maps (B, 768) → (B, base_dim). |
| `DynamicNeuralChild.update_emotions` | method | Runs attachment → regulation → defense, writes `self.emotional_state`, records to memory, returns the three sub-results. |
| `DynamicNeuralChild._process_trauma` | method | Decays defense anxiety threshold ×0.95, forces attachment update 0.3, triggers memory replay (batch 64). |
| `DynamicNeuralChild.express_feeling` | method | Returns a bracketed uppercase mood string from the 4 emotion scalars. |
| `DynamicNeuralChild.grow_layer` | method | Appends 4 layers widening `current_dim` by ×1.2. |
| `DynamicNeuralChild._reparametrize_weights` | method | Intended L0 parametrization pass. |
| `DynamicNeuralChild.update_drives_and_senses` | method | Fans feedback/satisfaction out to drives, sensory, ToM, attachment, defenses. |

## Key behaviour

- Dimensions in `forward` (child_model.py:268): input (B, 768) → `input_projection` → (B, 128); sensory → (B, 256); drives → (B, 10); emotion → (B, 4). Concatenation is 398, which is exactly the `decision_network` input width (child_model.py:148). Output of `decision_network` is `base_dim` = 128.
- `core_layers` (child_model.py:139) is a 4-element `ModuleList` iterated manually — Linear/LayerNorm/GELU/Dropout at 128 wide. It runs *after* `decision_network`, not before.
- Training-only stochastic masking (child_model.py:316-319): after every `nn.Linear` in `core_layers`, `curiosity_level = curiosity * trust_level` and elements survive where `rand > 0.1 / curiosity_level`. No rescaling compensates for the dropped mass.
- Training-only defense gating (child_model.py:321-325) multiplies the output by `1 - defense_strength` when a defense is active; anxiety is taken from emotion index 2 (fear), consistent with `update_emotions` (child_model.py:177).
- Growth: `growth_rate = 1.2`, so `grow_layer` goes 128 → 153 → 183 …, appending Linear/LayerNorm/GELU/Linear each call. Widths chain correctly across repeated calls, but the module's output width changes with them.
- `express_feeling` short-circuits to `[CALM]` when `‖emotional_state‖ < 0.2`; otherwise thresholds are joy 0.8/0.6, trust 0.8/0.6, fear 0.7/0.5, surprise 0.7/0.5, with a `tired` fallback when all components are below 0.3.

```
 x(768) ─input_projection─► x(128) ─┐
                                    ├─cat(398)─► decision_network ─► (128)
 sensory.process_input(x) ─►(256) ──┤                                 │
 drives.get_motivation_vector()►(10)┤                    confirmation_bias (no-op)
 emotional_state ─────────────►(4) ─┘                                 │
                                                          core_layers loop ─► out
```

## Imports

- Third-party: `torch`, `torch.nn`, `torch.nn.utils.parametrize`.
- Standard library: `time`.
- Project-internal: `EmotionalRegulation` from `emotional_regulation`; `DifferentiableMemory` from `memory_module`; `TheoryOfMind`, `AttachmentSystem`, `DefenseMechanisms` from `psychological_components`.

## Defects and gaps

- **Sensory input is ignored.** `process_input` (child_model.py:27) reshapes `stimulus` only to read its batch size, then builds `combined` purely from the four learnable parameter vectors (child_model.py:40-45). The 256-d "sensory" output is identical for every input of a given batch size.
- **Helper classes are not `nn.Module`s.** `SensoryExperience` (child_model.py:9) and `CoreDrives` (child_model.py:54) are plain classes assigned onto an `nn.Module`. Their `nn.Parameter`s, `integration_net`, and `regulation` are therefore outside `parameters()`, `state_dict()`, and `self.to(self.device)` (child_model.py:167) — no gradients through an optimizer built from `model.parameters()`, and no checkpointing.
- **`grow_layer` parametrization loop always fails silently** (child_model.py:252-259): `register_parametrization` takes a module, not a `nn.Parameter`, and the bare `except Exception: pass` swallows every raise. The whole loop is dead work.
- **New layers stay on CPU.** `grow_layer` (child_model.py:245-250) never calls `.to(self.device)`, so when `self.device` resolved to CUDA (child_model.py:107) the appended layers run on CPU against CUDA activations.
- **Growth breaks the output contract.** After one `grow_layer` call `forward` returns `new_dim` (153) instead of `base_dim` (128), while `decision_network`'s final layer stays at 128. Nothing in this file calls `grow_layer` (defined at child_model.py:242), so whether it is ever exercised is unverifiable from this file alone.
- **Hallucinated API in `_reparametrize_weights`** (child_model.py:265): `nn.utils.parametrization.L0Parametrization` does not exist in PyTorch (the module is `torch.nn.utils.parametrize`, and it has no such class). The method is also never called anywhere in this file.
- **`psychological_projection` (child_model.py:132) and `drive_projection` (child_model.py:131) are constructed and never used.** `psychological_projection` is sized 393 (`base_dim + 256 + 5 + 4`), contradicting the actual 398-wide concatenation, which uses all 10 drive+trait dims rather than the 5 drives.
- **Confirmation bias is a permanent no-op.** `current_beliefs` is zeroed at init (child_model.py:157) and never written in this file, so `apply_confirmation_bias` always multiplies by 1.0.
- **`theory_of_mind_output` (child_model.py:305) is computed and discarded** — the Theory-of-Mind forward pass has no effect on the returned output.
- **Masking can zero the whole activation.** With `curiosity` at its clamp floor 0.1 (child_model.py:92) and `trust_level` ≤ 0.5, `0.1 / curiosity_level` ≥ 2.0, so `rand > threshold` is False everywhere and `output` becomes all zeros (child_model.py:318). Both operands are Python floats via `.item()` (child_model.py:317), so a `trust_level` of exactly 0 raises `ZeroDivisionError`.
- **Emotional state is a plain tensor attribute, not a buffer** (child_model.py:127): it is absent from `state_dict()`, and `update_emotions` assigns the regulator's output without `detach` (child_model.py:181), so the autograd graph built in one step is still attached when `forward` re-reads it (child_model.py:286).
- **`experience_feedback` is an unused parameter** of `CoreDrives.update_drives` (child_model.py:88); `personality_traits` are never updated by any method in this file.
- **Dead arithmetic** in `express_feeling`: `baseline` is `zeros_like`, so `deviation` equals `emotional_state` (child_model.py:208-209).
- **String multiplication instead of intensity:** `"HAPPY" * min(int(joy * 3), 3)` (child_model.py:220) emits `HAPPYHAPPY` for joy in [0.8, 1.0) — repeated words, not a scalar.
- **Inconsistent call inputs within this file.** `attachment` is called with `mother_vector` (child_model.py:170) and with the 4-d `emotional_state` (child_model.py:306). `defense_mechanisms` is called with `mother_vector` (child_model.py:178), with the 398-wide `combined_input` (child_model.py:323), and with a (1, 4) `emotional_state` (child_model.py:215) — at most one of the three widths can be the intended one.
- **`express_feeling` reaches past the module call** and reads `self.attachment.attachment_styles` directly (child_model.py:214). Because the joy test is `joy >= 0.8 and attachment_state[0] > 0.5` (child_model.py:219), high joy with a low first attachment style falls through to the `elif joy >= 0.6` branch and reports `content`.
- **Hardcoded constants that break under changed parameters:** 768 input width (child_model.py:130), 398 decision-network width (child_model.py:148), 960 sensory concat width (child_model.py:19), replay batch 64 (child_model.py:203), and the fixed emotion width 4 assumed by `emotional_state`, index-2 fear, and the 4-way unpack in `express_feeling`. Only `base_dim` is a constructor parameter, and changing it breaks the 398 concatenation immediately. The `device='cuda'` defaults on both helper classes (child_model.py:10, child_model.py:55) are dead — both are always constructed with an explicit device (child_model.py:110-111).
- **`update_sensitivity` assumes a broadcastable feedback shape:** the in-place `self.attention.data += torch.tanh(feedback) * 0.1` (child_model.py:51) requires `feedback` to broadcast to shape (4,) or it raises; nothing in this file validates it.
- **Comment/code contradiction:** the comment at child_model.py:138 claims the core-layer input was changed to `base_dim` "from 393", but `core_layers` consumes `decision_network`'s output, never a 393-wide tensor.
- **Batch handling is wrong for rank-3 inputs** (child_model.py:270-274): anything that is not 2-D gets `unsqueeze(0)` and `batch_size = 1`, so a 3-D input silently becomes 4-D and the asserts at child_model.py:295-298 still pass.

## Notes

- The four asserts at child_model.py:295-298 are tautological — every operand's batch dimension is derived from `batch_size` earlier in the same function.
- `express_feeling` and `update_emotions` both assume `emotional_state` is a 1-D 4-vector; `update_emotions` assigns whatever `regulate` returns (child_model.py:181), so a (1, 4) return would break the 4-way unpack and the `unsqueeze(0)` in `forward`. Whether that happens is unverifiable from this file alone.
- `last_attachment_trust` is initialised at child_model.py:164 and overwritten at child_model.py:307; nothing in this file reads it. The comment above it (child_model.py:163) says it is "for loss computation", which would have to live outside this file — unverifiable from this file alone.
- The only `print` is the trauma warning at child_model.py:191, driven by real `trauma_info` values.
