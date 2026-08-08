# Network design

Reference for the neural networks in **this** repo (`core/neural_network.py`, `mind/networks/*.py`, `mind/language.py`), with the predecessor repo as an external comparison. Every claim is tagged **CURRENT** (true of the code today) or **PLANNED / NOT IMPLEMENTED**. Per-file detail: [neural_network](modules/core/neural_network.md) · [consciousness](modules/mind/networks/consciousness.md) · [emotions](modules/mind/networks/emotions.md) · [perception](modules/mind/networks/perception.md) · [thoughts](modules/mind/networks/thoughts.md) · [language](modules/mind/language.md) · [mind_core](modules/mind/mind_core.md) · [message_bus](modules/communication/message_bus.md) · [config](modules/config.md) · [ideas](ideas.md)

## 1. The `NeuralNetwork` base class — CURRENT

`core/neural_network.py`, subclass of `nn.Module` **and** `ABC`. Builds no layers itself: `__init__` (:139-176) records `name`, `input_dim`, `output_dim` and bookkeeping only.

| Member | Kind | Status |
|---|---|---|
| `forward(x) -> Tensor` | `@abstractmethod` (:178) | CURRENT — implemented in all 5 subclasses |
| `process_message(msg) -> Optional[VectorOutput]` | `@abstractmethod` (:190) | Implemented in all 5, **called from nowhere in the repo** — the whole inbound half is unreachable |
| `generate_text_output() -> TextOutput` | `@abstractmethod` (:202) | CURRENT — `mind_core.py:1061`, `neural-child-dashboard.py:278` |
| `autonomous_step()` | **not on the base class** | Defined per subclass; `mind_core.py:1064` calls it behind `hasattr` |
| `clone_with_growth(factor, min_dim)` | concrete, raises `NotImplementedError` (:970) | Overridden by all 5; every override crashes — §5 |

Other concrete base methods: `experiential_learning`, `batch_learning`, `evaluate`, `update_developmental_stage`, `_check_for_network_growth`, `_grow_layer`, `_prune_layer`, `_grow_network_for_new_stage`, `save_model`/`load_model` (`pytorch`|`torchscript`|`onnx`), `merge_with`, `apply_gaussian_noise`, `set_plasticity`, `get_complexity_metrics`.

**Data model.** `NetworkState` (`core/schemas.py:47`) — `name`, `active`, `last_update`, free-form `parameters`, `developmental_weights: Dict[DevelopmentalStage, float]`; `parameters["pending_messages"]` doubles as the outbound mailbox (§7). `GrowthMetrics` — six floats in [0,1] (`connection_density`, `plasticity`, `pruning_rate`, `specialization`, `integration`, `adaptability`), refreshed per stage. `NeuralGrowthRecord` — one grow/prune/merge/noise event with layer, old/new shape, `growth_factor`, `trigger`, stage.

**Developmental weighting.** `learning_rate = 0.01 * (0.5 + plasticity)` (:251, :1055) and `effective_lr = learning_rate * developmental_weights[stage]` (:290, :852). `__init__` seeds those weights to **0.0 for every stage** (:153-155) and the update branch is gated on `if self.training and effective_lr > 0` (:293), so learning is a no-op until `update_developmental_stage` runs — in practice at registration (`mind_core.py:973`).

**Hand-rolled gradient descent.** `experiential_learning` (:260-337) **bypasses `torch.optim` entirely**; no optimizer is constructed anywhere in the repo.

```
output = forward(input)
target ??= where(output > 0.8*rowmax, output*1.1, output*0.9)   # not detached (:283)
loss = MSELoss(output, target); zero_grad(); loss.backward()
with no_grad():
    for name, param in named_parameters():
        param -= effective_lr * param.grad                      # manual SGD step
        activity_tracker[name].append(param.grad.abs().mean())  # last 50 samples
if experiences_since_last_growth >= 100: _check_for_network_growth(...)
```

The "Hebbian" label in the source comment describes a plain SGD step — no co-activation term exists; the `activity_tracker` samples are the only input to the growth/prune decision (§5). Callers: `mind_core.py:1015` (perception) and `:1039` (language). CURRENT: **only perception is ever trained** — no entry point registers a language network, and `cli.py` never calls `process_input` at all.

## 2. Per-network table — CURRENT

| Network | File | Architecture | Defaults in/hid/out | `config.py` `MindConfig.networks` | Constructed at | Stage-modulated attributes |
|---|---|---|---|---|---|---|
| consciousness | `mind/networks/consciousness.py` | 2-layer `nn.RNN` + self-model MLP + Linear head | 64 / 128 / 64 | 64 / 128 / 64 | `cli.py:180`, dashboard:103 | `self_awareness`, `integration_capacity`, awareness cap |
| emotions | `mind/networks/emotions.py` | 3× Linear MLP, Dropout 0.2, Sigmoid | 32 / 64 / 32 | 32 / 64 / 32 | `cli.py:189`, dashboard:104 | `reactivity`, `regulation`, memory cap, emotion unlocks |
| perception | `mind/networks/perception.py` | two stream MLPs + attention + integration MLP → Sigmoid | 128 / 256 / 64 | 128 / 256 / 64 | `cli.py:198`, dashboard:105 | `object_recognition`, `pattern_recognition`, memory cap |
| thoughts | `mind/networks/thoughts.py` | `nn.GRU` + thought-generator MLP (+2 unused submodules) | 64 / 128 / 64 | 64 / 128 / 64 | `cli.py:207`, dashboard:106 | `abstract_thinking`, `logical_reasoning`, `creativity`, memory cap |
| language | `mind/language.py` | embedding MLP + 2-layer `nn.LSTM` + grammar MLP + Linear head | 96 / 192 / 48 | 96 / 192 / 48 | **never** | `sentence_complexity`, `understanding_level`, `expression_level` |

Config and defaults agree on every dimension; the repo's `config.yaml` sets only `model.llm_model` and `model.embedding_model`, so `config.py`'s defaults are what runs. `cli.py` reads `config.mind.networks` with those same values as fallbacks; the dashboard constructs all four with **no arguments**. `hidden_dim` is a constructor argument in all five and is **never stored on the instance** in any of them.

## 3. Per-network detail

### consciousness — CURRENT

`forward` gives 2-D input a length-1 sequence axis, keeps `self.hidden` across calls without detaching, and feeds the last RNN timestep to `output_layer`. `awareness_level` is written as `sigmoid(result.mean())` *before* the self-model blend, so it describes a different vector than the one returned. Blending is gated on `self_awareness > 0.1`; INFANT is exactly 0.1, so it is off at the starting stage.

| | INFANT | TODDLER | CHILD | ADOLESCENT | MATURE |
|---|---|---|---|---|---|
| `self_awareness` | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| `integration_capacity` | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
| awareness cap | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |

`autonomous_step`: hidden ×0.95, awareness ± uniform 0.025, clamp [0.1, 1.0], append a `consciousness` message.

### emotions — CURRENT

`forward` scales the MLP output by `reactivity`, then applies a dampening term toward 0.5 when `regulation > 0.5` (emotions.py:91-92) which, given the pairings below, raises the mean rather than suppressing it. Emotion values are **not** produced by the network: `_update_emotional_state` picks the emotion from the message's valence sign and scales by `response.mean()`. Nothing trains `emotion_processor`, so it stays a fixed random projection.

| | INFANT | TODDLER | CHILD | ADOLESCENT | MATURE |
|---|---|---|---|---|---|
| `reactivity` | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 |
| `regulation` | 0.2 | 0.4 | 0.6 | 0.7 | 0.9 |
| memory cap (`5+5·value`) | 10 | 15 | 20 | 25 | 30 |
| unlocks | — | INTEREST | ANTICIPATION | — | — |

`autonomous_step`: fixed per-call decay (SURPRISE 0.03, FEAR 0.015, JOY 0.005, others 0.01), independent of wall time; emotions already at exactly 0.0 are never removed.

### perception — CURRENT

`forward` splits input into visual `x[:, :in//2]` and auditory `x[:, in//2:]`, runs each through Linear→ReLU→Dropout(0.2)→Linear→ReLU to `hidden//4`, concatenates, gates the halves with a `Linear(hidden//2, 2)`+softmax attention at TODDLER and above, then integrates to `output_dim` via Sigmoid. Gaussian noise scaled by `0.5 − object_recognition` is added while that value is below 0.5. `_extract_perception` derives `emotional_valence = 2 × (mean first output half − mean second half)`, clamped to [-1, 1]; `salience` is arithmetically identical to `output_summary`.

| | INFANT | TODDLER | CHILD | ADOLESCENT | MATURE |
|---|---|---|---|---|---|
| `object_recognition` | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 |
| `pattern_recognition` | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| perception cap (`3+3·value`) | 6 | 9 | 12 | 15 | 18 |

`autonomous_step`: needs >2 stored perceptions and `pattern_recognition > 0.3` (CHILD+). The "detection" is `random.random() < pattern_recognition` — stored perceptions are counted, never inspected.

### thoughts — CURRENT

`forward` takes the last GRU timestep into `thought_generator`, adds Gaussian noise scaled by `0.5 − abstract_thinking` while below 0.5, and with probability `creativity` boosts 20% of output dims. Thought **text** is not derived from that vector — hardcoded templates plus a stage-gated random vocabulary. `_form_belief` ignores its `source_info`; the subject/predicate/object triple is uniformly random vocabulary. `association_network` and `belief_network` are constructed and never referenced.

| | INFANT | TODDLER | CHILD | ADOLESCENT | MATURE |
|---|---|---|---|---|---|
| `abstract_thinking` | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| `logical_reasoning` | 0.1 | 0.2 | 0.5 | 0.7 | 0.9 |
| `creativity` | 0.4 | 0.6 | 0.7 | 0.5 | 0.8 |
| thought cap (`2+2·value`) | 4 | 6 | 8 | 10 | 12 |

`autonomous_step`: spontaneous-thought probability `0.1 + 0.05 × stage.value`; at CHILD+ one random belief's confidence drifts by ±0.05.

### language — PLANNED / NOT IMPLEMENTED (code complete, never instantiated)

~1170 lines. Vocabulary acquisition, POS guessing and sentence generation are rule-based Python; the torch stack only produces the vector `forward` returns. At TODDLER+ the 48-wide `grammar_network` output is zero-padded to 192 and blended into the LSTM activation with weight `min(1.0, 0.2 × (stage.value − 1))`; the result is multiplied by `expression_level`, which the constructor initialises to 0.0 (language.py:147), so `forward` returns all zeros until `update_developmental_stage` runs. `autonomous_step` is a probabilistic spontaneous utterance plus association consolidation.

| | INFANT | TODDLER | CHILD | ADOLESCENT | MATURE |
|---|---|---|---|---|---|
| `sentence_complexity` | 0.0 | 0.2 | 0.5 | 0.8 | 1.0 |
| `understanding_level` | 0.1 | 0.3 | 0.6 | 0.9 | 1.0 |
| `expression_level` | 0.1 | 0.3 | 0.6 | 0.8 | 1.0 |

CURRENT: `LanguageNetwork` is never constructed by `cli.py` or the dashboard. `mind_core.py:1653` gates a `vocabulary_learned` truncation on a registered `language` network; when one is registered the Mind's vocabulary set is sliced to that network's reported `vocabulary_size` — measured, 15 distinct words fed, 7 retained.

## 4. Developmental modulation — CURRENT

```
update_developmental_stage(stage)                                   # base, :220-258
  ├─ developmental_weights: 0.2/0.4/0.6/0.8/1.0 for stages <= stage, 0.1 above  → effective_lr
  ├─ growth_metrics.update_for_developmental_stage(stage)                       → table below
  └─ if stage.value > 1: _grow_network_for_new_stage(stage)         # base no-op, never overridden
     then each subclass override applies its own table (§3)
```

| | INFANT | TODDLER | CHILD | ADOLESCENT | MATURE |
|---|---|---|---|---|---|
| connection_density | 0.1 | 0.3 | 0.5 | 0.7 | 0.8 |
| plasticity | 0.9 | 0.8 | 0.6 | 0.4 | 0.3 |
| pruning_rate | 0.1 | 0.3 | 0.4 | 0.3 | 0.2 |
| specialization | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| integration | 0.1 | 0.2 | 0.4 | 0.6 | 0.8 |
| adaptability | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 |

`plasticity` feeds `learning_rate` and the growth probability; `pruning_rate` feeds the prune probability. Callers: `mind_core.py:973` (registration), `:1751` (stage advance), `cli.py:222` (optional `starting_stage`). CURRENT: the INFANT→TODDLER gate requires 3 distinct emotions experienced and that counter can never leave 0, so **only the INFANT column is ever applied** in a shipped run.

## 5. The growth mechanism

**PLANNED design**, two independent paths:

```
A. intra-network (core/neural_network.py)
   experiences_since_last_growth >= 100 → _check_for_network_growth(trigger)
        mean(activity_tracker[param]) > 0.7 → growth candidate
        mean(activity_tracker[param]) < 0.1 → prune candidate
        random() < plasticity   → _grow_layer()   ─┐ base impl: logs, records a
        random() < pruning_rate → _prune_layer()  ─┘ NeuralGrowthRecord, returns False

B. whole-network swap (mind_core.py:1594-1626, on a timer)
   random() < growth_schedule[stage]        # INFANT 0.0, TODDLER .001, CHILD .002,
     → self.networks[name] = network.clone_with_growth(1.2)   # ADOLESCENT .003, MATURE .001
```

**CURRENT: neither path has ever executed.** Two bug classes.

**Class A — `random` is never imported in the base class.** Used at `core/neural_network.py` :382, :386, :400, :402. The first call reaching a growth or prune candidate raises `NameError`, uncaught inside `process_input`. No subclass overrides `_grow_layer`, `_prune_layer` or `_grow_network_for_new_stage`, so even past that the base implementations only log and return `False`.

**Class B — `clone_with_growth` crashes in all five subclasses.**

| Network | missing `copy` import | missing `NeuralGrowthRecord` import | wrong `nn.Sequential` index (index 1 is `ReLU`, no `.out_features`) |
|---|---|---|---|
| consciousness | consciousness.py:392 | consciousness.py:397 | — |
| emotions | emotions.py:429 | emotions.py:440 | **emotions.py:418** — first failure, `AttributeError` |
| perception | perception.py:495 | perception.py:503 | **perception.py:481** — first failure, `AttributeError` |
| thoughts | thoughts.py:679 | thoughts.py:690 | — |
| language | language.py:1139 | language.py:1151 | — |

Even fixed, **none of the five transfers weights** — the "clone" would be a freshly random-initialised network with only Python-side scalars and history copied. `mind_core.py:1616-1626` wraps the call in a broad `try/except`, so failure surfaces as a log line. Path B is additionally unreachable in a shipped run: `growth_schedule[INFANT]` is 0.0 and the simulation cannot leave INFANT.

## 6. Dimension hazards — CURRENT

| Hazard | Site | Effect |
|---|---|---|
| Hidden state hardcoded to width **128** | `consciousness.py:101` — `torch.zeros(2, x.size(0), 128)` | any `hidden_dim != 128` fails inside the RNN on the first `forward`; the leading `2` also hardcodes `num_layers=2` |
| Hidden state hardcoded to width **128** | `thoughts.py:125` — `torch.zeros(1, batch_size, 128, device=x.device)` | same for the GRU |
| `hidden_dim` never stored | all 5 networks | clone paths reverse-engineer it from `rnn.hidden_size` / `thought_rnn.hidden_size` / `lstm.hidden_size`, or from the wrong `Sequential` index |
| `hidden_dim % 4 != 0` | `perception.py:59-67` | concat gives `2*(hidden//4)` while attention/integration expect `hidden//2` |
| odd `input_dim` | `perception.py:105` vs `:51` | auditory slice is one element wider than its `Linear` accepts |
| `output_dim > hidden_dim` | `language.py:304` | `torch.zeros(batch, negative)` |
| recurrent state persists undetached | `consciousness.py:104`, `thoughts.py:133`, `language.py:287-292` | batch size pinned to the first call; a different batch size raises inside the recurrent module |

Both 128 literals are correct only for the default `hidden_dim`. `clone_with_growth(1.2)` produces 153, which would trip them immediately if the import bugs above were fixed first.

## 7. Message contracts

Each network emits by appending `NetworkMessage.to_dict()` to `state.parameters["pending_messages"]`. `Mind._retrieve_network_messages` (`mind_core.py:1094`) drains that list each `step()`: `receiver == "mind"` goes to `_process_mind_message`, everything else is published to the bus.

| Sender | `message_type` | `receiver` | priority | emitted from | CURRENT status |
|---|---|---|---|---|---|
| consciousness | `consciousness` | mind | 0.7 | `autonomous_step` | reaches `_process_mind_message` |
| emotions | `emotion` | mind | 0.8 | every `_update_emotional_state` | only reachable via `process_message` → never fires |
| perception | `perception` | emotions | 0.7 | `process_message` (sensory_input) | never fires |
| perception | `pattern` | thoughts | 0.6 | `autonomous_step`, CHILD+ | never fires (stage stuck at INFANT) |
| thoughts | `belief` | mind | 0.7 | `_form_belief`, TODDLER+ | never fires |
| language | `language_output` | **consciousness** | 0.7 | `_send_language_output` | network never instantiated; `_process_mind_message` has a `language_output` branch this receiver bypasses |

**The consuming half is unreachable.** `process_message` is the only inbound entry point on a network and nothing in the repo calls it. Messages addressed to `emotions`, `thoughts` or `consciousness` are published to the bus, but the only subscriber is `Mind` with `MessageFilter(receiver="mind", min_priority=0.3)` (`mind_core.py:876-880`), so no subscriber matches and they are dropped. Measured over 400 interactions with rich input: the bus's message history stayed empty; 0 emotions experienced, 0 beliefs, 0 consolidated memories, stage never left INFANT.

## 8. Comparison with the predecessor — EXTERNAL REFERENCE

`github.com/renatokuipers/neural-child` (2025, flat layout, archived) is **not part of this repository**. Summarised from [docs/legacy-neural-child/](legacy-neural-child/).

| Legacy component | File | Shape | Contrast with this repo |
|---|---|---|---|
| `DynamicNeuralChild` | `child_model.py` | 768→128 projection ⊕ sensory 256 ⊕ drives 10 ⊕ emotion 4 = 398 → decision net → 4 core layers @128 | One monolithic agent; here the mind is five `NeuralNetwork` subclasses coordinated by `Mind` |
| `EmotionalRegulation` / `EmotionalState` | `emotional_regulation.py` | LSTM over a 5-entry history of 4-d emotion vectors → 40→256→4 | Here emotions are a Python dict driven by valence sign; the MLP is untrained |
| `TheoryOfMind`, `AttachmentSystem`, `DefenseMechanisms` | `psychological_components.py` | 398→512→256→128 trunk + 4 heads; 4→256 trust/bonding; 398→256→128 + 7 defense heads | No analogue here — no attachment, ToM or defense modelling |
| `MetacognitionSystem` | `metacognition.py` | 128→256→128 base + 5 hypothesis nets + critic 256→512→1 + LSTM + complexity head | Nearest analogue is `Mind._self_reflection`, inert below `self_awareness` 0.3 |
| `MoralPolicyNetwork` | `moral_network.py` | 128→256→512→256 + two sigmoid gate vectors → value head → tanh | No moral/value network here |
| `SymbolGrounding` | `symbol_grounding.py` | 768-d embedding matrix, dot-product argmax lookup | Here grounding is `hash(token) % 1000` digits (`mind_core.py:1034-1036`) |

Patterns that carried over as regressions: legacy `grow_layer` widened `current_dim` ×1.2 and chained correctly across calls (though it broke the output contract and left new layers on CPU), whereas this repo's `clone_with_growth` crashes before constructing anything usable (§5); legacy hardcoded 768 / 398 / 960 widths, this repo hardcodes 128 in two `forward` implementations (§6); legacy computed `theory_of_mind_output` and discarded it, this repo builds `association_network`, `belief_network` and `concept_network` in thoughts and never references them. Legacy also had a self-supervised trainer and a curriculum manager ([self_supervised_trainer](legacy-neural-child/self_supervised_trainer.md), [curriculum_manager](legacy-neural-child/curriculum_manager.md)); this repo has neither — only the manual SGD step in `experiential_learning`, reached by perception alone.
