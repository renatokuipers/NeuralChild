# main.py

Entry point of the legacy neural-child program (358 lines). Defines two classes — `MotherLLM`, an LLM-backed caregiver that emits stage-conditioned stimuli, and `DigitalChild`, a facade that wires together the brain, memory, morality, metacognition, curriculum and trainer subsystems — plus a `main()` that runs an unbounded interaction loop until Ctrl-C. The bulk of the file (main.py:18-198) is a literal dict of 18 hand-written system prompts, one per developmental stage.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `MotherLLM` | class | Holds `stage_prompts` (18 entries), plus `emotional_history`, `feedback_history`, `conversation_history` lists (main.py:14-200). |
| `MotherLLM.generate_stimulus(stage, child_response="[EMPTY]")` | method | Appends child turn, calls `chat_completion` with structured output, returns `{'text', 'emotional_vector'}` (main.py:202-226). |
| `DigitalChild` | class | Constructs all six subsystems, records `birth_date`, zero-inits a 4-element emotional state on `cuda` (main.py:228-238). |
| `DigitalChild.update_emotions(mother_vector)` | method | Moves state 30% toward the mother vector, adds Gaussian noise, clamps to [0,1] (main.py:240-243). |
| `DigitalChild.express_feeling()` | method | Maps the 4 scalars to an uppercase bracketed label such as `[HAPPY & WARM]`; `[NEUTRAL]` when no threshold is met (main.py:245-282). |
| `DigitalChild.perceive(stimulus)` | method | Embeds `stimulus['text']`, returns a `(1, D)` cuda tensor (main.py:284-288). |
| `DigitalChild.respond(perception)` | method | Forward pass through `self.brain` under `torch.amp.autocast("cuda")` (main.py:290-292). |
| `DigitalChild.learn(mother_feedback)` | method | Records the experience in memory, returns `trainer.training_step(...)` (main.py:294-302). |
| `DigitalChild.age()` | method | Whole months since birth, computed as `days // 30` (main.py:304-305). |
| `main()` | function | Infinite perceive/respond/learn loop; on `KeyboardInterrupt` prints a farewell and saves `brain.state_dict()` (main.py:307-355). |

## Key behaviour

- Emotion vector is fixed at 4 dims in the order joy, trust, fear, surprise — written at main.py:211-216, read by index at main.py:246-249.
- Mother emotion carry-over: `decay = 0.9 ** len(emotional_history)` and only the *previous* vector is added (main.py:217-219). The raw (pre-sigmoid) sum is what gets stored in history; the caller receives `torch.sigmoid(...)` (main.py:220-225).
- `express_feeling` thresholds: joy/trust ≥0.8 then ≥0.6; fear/surprise ≥0.7 then ≥0.5 (main.py:254-275).
- Child emotion update is `state += 0.3*(mother - state) + 0.1*N(0,1)`, then clamped (main.py:242-243).
- Loop consolidation triggers: wall-clock `time.time() % 86400 < 3600` (main.py:348), and `memory_allocated / max_memory_allocated > 0.9` → `replay_consolidation(batch_size=16)` (main.py:350-351).
- Curriculum advance is driven entirely by the second LLM call's fields `success_metric`, `complexity_rating`, `self_critique_score` (main.py:341-345).

```
generate_stimulus(stage, feeling)      [LLM call #1]
        |  {'text', 'emotional_vector'}
        v
update_emotions  ->  perceive (embed -> (1,D) cuda)
        |
        v
     respond (brain forward, autocast)
        |  response tensor
        v
chat_completion(f"...{response}")      [LLM call #2] -> feedback dict
        |
        +--> learn(): memory.record_experience -> trainer.training_step -> loss
        +--> telemetry['loss' | 'memory_usage' | 'moral_scores']
        +--> curriculum.update_stage(...)
        +--> conditional memory.replay_consolidation()
        |
        \--(loops forever; only KeyboardInterrupt exits -> torch.save)
```

## Imports

- Third-party / stdlib: `torch`, `time`, `datetime.datetime`.
- Project-internal: `llm_module.chat_completion`, `child_model.DynamicNeuralChild`, `curriculum_manager.DevelopmentalStage`, `curriculum_manager.DevelopmentalSystem`, `memory_module.DifferentiableMemory`, `moral_network.MoralPolicyNetwork`, `metacognition.MetacognitionSystem`, `self_supervised_trainer.AutonomousTrainer`, `text_embed.get_embeddings`.

## Defects and gaps

- Device is hardcoded `'cuda'` at main.py:216, main.py:238, main.py:287 and in the autocast at main.py:291, while `MoralPolicyNetwork` alone receives `self.brain.device` (main.py:232). No CPU path exists; the program cannot run without CUDA.
- Mixed key-access discipline in the same response dict: `.get(..., default)` for the four emotion scalars (main.py:212-215) but bare `response['content']` at main.py:221 and main.py:224 — a missing key raises `KeyError`. Same for `feedback['reward_score']` (main.py:334, 339) and `success_metric` / `complexity_rating` / `self_critique_score` (main.py:342-344).
- `0.9 ** len(self.emotional_history)` (main.py:218) decays with total history length, not elapsed steps, so the carry-over term shrinks toward nothing as the run proceeds (0.9^100 ≈ 3e-5) — the "emotional history" stops influencing anything.
- `self.conversation_history` is stringified whole into every prompt (main.py:205) and never truncated; `emotional_history` and `feedback_history` grow unbounded too. Prompt size grows linearly with runtime.
- `self.feedback_history` is appended at main.py:222 but never read anywhere in this file.
- `self.metacognition` is constructed at main.py:233 and never used again in this file.
- `telemetry` (main.py:309-314, 337-339) is never written to disk, plotted, or printed. Its `psychological` block is snapshotted once before the loop and never refreshed.
- `telemetry['moral_scores']` (main.py:339) stores `feedback['reward_score']` from the LLM, not any output of `MoralPolicyNetwork` — the key name contradicts the value.
- `time.time() % 86400 < 3600` (main.py:348) is a UTC-midnight-hour test, not a once-per-day gate; during that hour it fires on *every* loop iteration.
- `torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()` (main.py:350) compares current usage against the all-time peak, not device capacity, so it fires whenever memory is near its own high-water mark — including immediately after a fresh peak, when consolidation cannot help.
- The `response` tensor is interpolated into a prompt string via f-string at main.py:325, sending a tensor repr to the LLM.
- The second `chat_completion` prompt (main.py:324-328) never names the fields the code then requires from it — `reward_score`, `success_metric`, `complexity_rating`, `self_critique_score` (main.py:334-344). Those names appear only inside the *stage* prompts used for call #1, so the second call is asked for free-form "nurturing feedback" and indexed as if it returned the schema.
- The child's actual brain output never reaches the mother. `generate_stimulus` is called with `child.express_feeling()` (main.py:319), so the `"role": "child"` turn recorded at main.py:203 is only an emotion label; `response` (main.py:322) is used solely in the call-#2 f-string and as `internal_state` (main.py:333). The conversation loop is not closed.
- `torch.sigmoid` at main.py:225 is applied to a sum of non-negative terms, so every returned component sits in [0.5, 1) and can never signal a low emotion; with the history carry-over (main.py:219) the components drift toward 1.0, pulling the child's state up in all four dimensions via main.py:242.
- `stage_prompts.get(stage, ...NEWBORN)` (main.py:204) silently falls back to the newborn prompt for any unmapped stage — a curriculum/enum mismatch degrades quietly instead of failing.
- The comment at main.py:330/337 asserts the trainer's return is "already a float"; nothing in this file verifies that, and a returned tensor would keep autograd graphs alive in `telemetry['loss']`.
- No error handling around either `chat_completion` call or the embedding call — only `KeyboardInterrupt` is caught (main.py:353). Any LLM/network failure terminates the run without saving weights.
- `get_embeddings(...)[0]['embedding']` (main.py:286) assumes a list-of-dicts shape with no validation.
- `age()` uses `days // 30` (main.py:305), so a fresh run always saves `digital_child_0mo.pth` into the current working directory (main.py:355), overwriting prior checkpoints.

## Notes

- Weights are persisted only on Ctrl-C; there is no periodic checkpointing and no resume path.
- The 18 prompt strings instruct the LLM to emit a "MotherResponse schema" with fields like `cognitive_labels` and `emotional_context`; this file never reads those fields — it only reads `joy`/`trust`/`fear`/`surprise`, `content`, `reward_score`, `success_metric`, `complexity_rating`, `self_critique_score`.
- Whether `DynamicNeuralChild` actually exposes `attachment.attachment_styles`, `defense_mechanisms.anxiety_threshold`, `theory_of_mind.social_bias` (main.py:311-313) is unverifiable from this file alone.
