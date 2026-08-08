# thoughts.py

Defines ThoughtsNetwork, a subclass of the project's NeuralNetwork base that converts inbound vectors into "thoughts" and occasional Belief objects. A GRU plus an MLP produce a thought vector; the human-readable thought text is not derived from that vector at all — it is drawn from hardcoded template strings and a random vocabulary, selected by developmental stage. Two of the four declared submodules are never invoked.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| ThoughtsNetwork | class | Stage-aware thought/belief generator; ctor takes input_dim=64, hidden_dim=128, output_dim=64 |
| .forward | method | (batch, input_dim) or (batch, seq, input_dim) → (batch, output_dim) in [0,1] via Sigmoid, plus noise/creativity perturbation |
| .process_message | method | Dispatches on message_type in {perception, emotion, pattern, query}; returns VectorOutput or None |
| .autonomous_step | method | Probabilistic spontaneous thought + random jitter of one existing belief's confidence |
| .update_developmental_stage | method | Calls super, then overwrites the three capacity scalars, extends vocabulary, clears hidden state |
| .generate_text_output | method | TextOutput with stage-specific phrasing and a hardcoded confidence (0.4–0.8) |
| .clone_with_growth | method | Builds a *new, randomly initialised* ThoughtsNetwork with scaled dims; copies only Python attributes, not weights — and raises NameError before returning (see Defects) |
| ._generate_thought / ._generate_thought_text / ._remember_thought / ._form_belief | methods | Internal; text is template-based, beliefs are random vocabulary triples |

## Key behaviour

- Submodules built in `__init__` (thoughts.py:41-73): thought_generator (hidden_dim→hidden_dim→hidden_dim→output_dim, Dropout 0.3, Sigmoid), association_network (input_dim*2→hidden_dim→1), belief_network (input_dim→hidden_dim//2→3, Softmax), thought_rnn (GRU input_dim→hidden_dim, 1 layer, batch_first).
- Forward path: input is unsqueezed to a length-1 sequence if 2-D (thoughts.py:128-131), pushed through the GRU, last timestep taken, then thought_generator. Hidden state is persisted on the instance across calls and across message sources; the only reset of *this* instance's state is in update_developmental_stage (thoughts.py:587). The assignment in clone_with_growth (thoughts.py:701) clears the new network's already-None state, not the original's.
- Developmental modulation, thoughts.py:144-156: if abstract_thinking < 0.5, Gaussian noise scaled by (0.5 − abstract_thinking) is added and clamped to [0,1]; with probability equal to creativity, 20% of output dimensions get a random additive boost scaled by creativity.
- Capacity scalars by stage (thoughts.py:528-554): INFANT 0.1/0.1/0.4, TODDLER 0.3/0.2/0.6, CHILD 0.5/0.5/0.7, ADOLESCENT 0.7/0.7/0.5, MATURE 0.9/0.9/0.8 for abstract_thinking / logical_reasoning / creativity. Init defaults are 0.1/0.1/0.3 (thoughts.py:79-85), so creativity's initial value matches no stage.
- Belief formation only fires when stage.value ≥ TODDLER.value AND pattern_strength > 0.4 AND random() < logical_reasoning (thoughts.py:235-239). Confidence is scaled by (0.7 + 0.3·logical_reasoning) (thoughts.py:467).
- Thought memory is capped at `2 + stage.value * 2` entries (thoughts.py:430), keeping the newest.
- Spontaneous-thought probability is `0.1 + stage.value * 0.05` (thoughts.py:487); belief-confidence drift requires stage.value ≥ CHILD.value and random() < logical_reasoning, adjusting one randomly chosen belief by ±0.05 (thoughts.py:507-513).

```
process_message(msg)
 ├─ "perception" → pad/truncate vector_data to input_dim → forward → thought → VectorOutput
 ├─ "emotion"    → zero vector, [0]=intensity        → forward → thought → VectorOutput
 ├─ "pattern"    → (stage≥TODDLER only) maybe _form_belief → queue NetworkMessage in
 │                  state["pending_messages"]; zero vector [0]=strength → forward → VectorOutput
 ├─ "query"      → "current_thought": one-hot placeholder vector
 │                 "belief": vector with [0]=max confidence of matching subject
 └─ anything else, or missing keys → None
```

## Imports

- Third-party: torch, torch.nn, numpy, plus stdlib typing, random, datetime, logging.
- Project-internal: core.neural_network (NeuralNetwork); core.schemas (NetworkMessage, VectorOutput, TextOutput, DevelopmentalStage, Belief).

## Defects and gaps

- `copy` is used at thoughts.py:679-682, 685, 689 but never imported — clone_with_growth raises NameError on first call.
- `NeuralGrowthRecord` is used at thoughts.py:690 but never imported — a second NameError on the same path.
- Hidden state is hardcoded to width 128 at thoughts.py:125, ignoring the hidden_dim constructor argument. Any hidden_dim ≠ 128 makes the first forward fail on a GRU size mismatch. The scaled hidden dim computed in clone_with_growth (thoughts.py:665) would hit this too, though that path dies earlier on the NameError above.
- thoughts.py:155 fills the creative mask via torch.rand with no device argument, while creative_mask is zeros_like(thought) and so follows the model's device; the indexed assignment of a CPU tensor into a non-CPU mask fails.
- Dropout(0.3) in thought_generator (thoughts.py:44) stays active during the inference paths in this file: process_message and autonomous_step wrap forward in torch.no_grad(), which does not disable dropout, and nothing in this file ever calls eval(). Repeated forwards on identical input therefore return different thought vectors.
- self.hidden_state is reassigned from the GRU output without detach (thoughts.py:133), so a training-mode second backward pass would traverse a freed graph. It is also fixed to the first batch size seen, so a changed batch size errors.
- association_network (thoughts.py:52) and belief_network (thoughts.py:60) are constructed but never referenced anywhere in this file — dead parameters, still counted in state_dict and optimizers.
- concept_network (thoughts.py:94) is initialised empty and only ever deep-copied; nothing writes to it, despite the "stores associations between concepts" comment.
- The INFANT early-return in _form_belief (thoughts.py:451-452) is unreachable from this file: its only call site (thoughts.py:242) is already inside a stage ≥ TODDLER guard.
- numpy (thoughts.py:12) and the typing names List and Tuple (thoughts.py:9) are imported but unused.
- The perception branch pads short input by list concatenation, `vector_data + [0.0] * n` (thoughts.py:179), silently assuming a Python list; a NumPy array or tensor would broadcast-add instead of pad, or raise. The truncation branch (thoughts.py:177) tolerates both.
- update_developmental_stage extends vocabulary lists unconditionally (thoughts.py:568-584); calling it twice with the same stage duplicates every word. MATURE adds no vocabulary at all.
- Even setting the NameErrors aside, clone_with_growth transfers no learned weights — the network built at thoughts.py:669-673 keeps its four freshly random-initialised submodules, contradicting the "clone" name and the growth-record framing.
- The query branch returns hand-built placeholder vectors (thoughts.py:296-298, 314-316) whose only informative element is index 0; the comments themselves say "In a real implementation, this would encode the thought/belief".
- _form_belief ignores its source_info parameter entirely (thoughts.py:440-478); the subject/predicate/object triple is uniformly random vocabulary, so the docstring claim of forming beliefs "based on observed patterns" is false.
- hidden_dim is never stored as an attribute; clone_with_growth recovers it from thought_rnn.hidden_size (thoughts.py:665) — a mismatch with the hardcoded 128 above.
- pending_messages accumulates in state without any visible drain in this file (thoughts.py:263-265); whether anything consumes it is unverifiable from this file alone.

## Notes

- The code requires DevelopmentalStage.value to be numeric (arithmetic at thoughts.py:430 and 487), not merely orderable. Whether the enum satisfies that is unverifiable from this file alone.
- Base-class members relied on but not defined here: input_dim, output_dim, name, developmental_stage, state.parameters, update_state, growth_metrics, experience_count, growth_history. Belief.update_confidence and Belief.to_natural_language, and NetworkMessage.to_dict, are likewise assumed.
- Output confidence values in generate_text_output are fixed constants per stage, not computed from any network signal.
