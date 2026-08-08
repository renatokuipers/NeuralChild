# Merge workflow — order of operations

The route from two half-working codebases to one that develops. Steps are ordered; the
boundary action after each one is part of the step, not an afterthought.

**Scope.** Merging the psychological models of the predecessor
([`docs/legacy-neural-child/`](../legacy-neural-child/)) into this repo's architecture
([`docs/architecture.md`](../architecture.md)), plus the Mother design in
[`docs/ideas.md`](../ideas.md). Three sources, not two.

## How to use this file

Work top to bottom. Tick a row only when its **exit criterion** is demonstrated, not when the
work feels done. Where a row says `/clear` or `/handoff`, that is a required step — skipping it
is how the last two attempts at this project accumulated context debt.

The slash commands in the Skill column are agent skills installed at the user level, not files in
this repo — `/setup-matt-pocock-skills` installs the set, and `/ask-matt` routes a situation to
the right one. A row whose skill is unavailable still names what has to happen; run it by hand.

## Ground rules

**Context hygiene.** Grilling → spec → tickets stay in **one unbroken window**; do not clear or
compact until after `/to-tickets`, so the thinking compounds. Every `/implement` then starts
fresh from its ticket. If a window approaches the smart zone (~150k tokens) before tickets are
cut, `/compact` at the nearest boundary rather than pushing on degraded.

**Boundary choice.** At each phase boundary, in order of preference: Continue → `/clear` →
`/handoff` → subagent → `/compact`. `/handoff` is narrow: a new harness, a new directory, a
colleague, or forking a side task mid-phase. Prototypes qualify (own directory); the wayfinder
map does not (its state lives on the tracker).

**Standing constraint.** No capability is ported until the code that consumes it exists and
changes behaviour. Both codebases failed by accumulating producers with no consumers — this repo
has `pending_messages` nothing drains and `process_message` with zero call sites; the predecessor
computes `theory_of_mind_output` and discards it on the next line. If you cannot name the
consumer, the port is premature.

---

## Checklist

| # | Stage | Step | Skill / action | Boundary after | Done |
|---|---|---|---|---|---|
| 0.1 | Precondition | Tracker and doc layout configured, labels named | `/setup-matt-pocock-skills` | — | [x] |
| 1.1 | Resurrect | File blocking defects as tickets | `/to-tickets` | Continue | [x] |
| 1.2 | Resurrect | Make the test harness go red | `/implement` → `/tdd` | `/clear` | [x] |
| 1.3 | Resurrect | Fix each defect, one ticket per window | `/implement` → `/tdd` → `/code-review` | `/clear` **between every ticket** | [ ] |
| 1.4 | Resurrect | Any fix that misbehaves against prediction | `/diagnosing-bugs` | `/clear` | [ ] |
| 1.5 | Resurrect | **Gate: a run reaches TODDLER** | run it, capture output | `/clear` | [ ] |
| 2.1 | Ground | Survey seams before grafting new networks | `/improve-codebase-architecture` | Continue | [ ] |
| 2.2 | Ground | Resolve the vocabulary collisions | `/domain-modeling` | `/clear` | [ ] |
| 3.1 | Chart | Chart the merge as decision tickets | `/wayfinder` | `/compact` at each decision | [ ] |
| 3.2 | Chart | Agent SDK facts for the Mother | `/research` (background) | Continue | [ ] |
| 3.3 | Chart | Questions needing a runnable answer | `/handoff` → `/prototype` → `/handoff` | `/handoff` both ways | [ ] |
| 3.4 | Chart | **Gate: every map decision closed** | — | `/clear` | [ ] |
| 4.1 | Plan | Collapse the map into a buildable plan | `/to-spec` | Continue — same window | [ ] |
| 4.2 | Plan | Split into tracer-bullet tickets with blocking edges | `/to-tickets` | `/clear` | [ ] |
| 5.1 | Build | Attachment (first capability) | `/implement` → `/tdd` → `/code-review` | `/clear` | [ ] |
| 5.2 | Build | Remaining capabilities, blockers-first | `/implement` per ticket | `/clear` **between every ticket** | [ ] |
| 5.3 | Build | **Gate: each capability observably changes behaviour** | run it | `/clear` | [ ] |
| 6.1 | Experiment | Vary the Mother, measure the trajectory | main flow from `/grill-with-docs` | — | [ ] |

---

## Stage detail

### Stage 0 — Precondition (complete)

`CLAUDE.md` and [`docs/agents/`](../agents/) exist: GitHub Issues via `gh`, external PRs enabled
as a triage surface, the five canonical triage labels named, single-context doc layout.
`CONTEXT.md` and `docs/adr/` do **not** exist yet — by design, they are created lazily at step 2.2.

**The labels are named in `docs/agents/`, not all created on the tracker.** Only `ready-for-agent`
exists, created when step 1.1 needed it to file tickets. `needs-triage`, `needs-info` and
`ready-for-human` are still absent — create them before a step reaches for one.

### Stage 1 — Resurrect

**Why not `/grill-with-docs` first.** Nothing to sharpen. The work is already specified to issue
quality in [`docs/architecture.md`](../architecture.md) §8 — twelve defects with file:line and
consequence — so it goes straight to `/to-tickets`. Grilling an enumerated bug list wastes a
window.

**Why not `/triage`.** Triage is for issues you did not create. These are ours.

**Ticket source** — §8's twelve, plus three cross-file defects found by execution that the
per-file reports could not see (each report was restricted to its own file and forbidden from
following imports):

- `Belief` declares no `id` field, but `BeliefNetwork.add_belief` assigns `belief.id` —
  raises `ValueError` under Pydantic v2, so every belief formed via `_process_mind_message`
  dies and is swallowed into a log line.
- `Belief` has no `to_dict()`, but `BeliefNetwork.to_dict()` calls it — raises `AttributeError`
  on the `save_state` path. Currently masked by the earlier `memories.json` failure; fixing that
  one exposes this one.
- `GrowthMetrics(plasticity=0.8)` raises `KeyError: 'field'` — the Pydantic v1 validator
  signature is broken under the installed v2. Quiet only because the sole construction site
  passes no arguments, which skips validators for defaulted fields.

One correction to fold in: §8 defect 11 overstates. `developmental_weights` is seeded to 0.0, but
`register_network` calls `update_developmental_stage` immediately — measured `effective_lr` after
registration is 0.00280, not zero. The seeding only bites a network that is never registered.

**Step 1.2 comes before every fix.** All five test files are 2-byte stubs, which is why every
defect above shipped silently. `/tdd` needs something that can go red. Both `pytest` and
`hypothesis` are already importable in the project environment, but `pyproject.toml` declares only
`pytest` — adding `hypothesis` to the `dev` extra is part of this step. It suits a simulation full
of invariants.

**Exit criterion for 1.5.** A run that reaches TODDLER — which this project has never done. One
synthetic emotion message carrying three emotions moves `emotions_experienced` 0 → 3, exactly the
INFANT gate, so the gate is close. But wiring `process_message` alone is **not sufficient**: of
the inter-network edges, perception→emotions delivers, perception→thoughts delivers only at
TODDLER and above, language→consciousness has no handler branch on the receiver, and
thoughts→mind raises on the `Belief.id` defect above.

### Stage 2 — Ground

`/clear` first: the bug-fix context is spent and irrelevant to design.

**2.1** — run the survey while the codebase is small. Five new networks are about to be grafted
in; find the seams before, not after. Anything it surfaces becomes an idea for the main flow.

**2.2** — `/domain-modeling` is normally pulled in underneath other skills, but here the *words*
are genuinely the problem, so invoke it directly. Collisions to settle, each of which currently
means two different things across the two codebases:

| Term | Collision |
|---|---|
| stage | 5 stages here, 18 in the predecessor's curriculum, 12+1 in `ideas.md` |
| network | a peer with a `NetworkState` here; a submodule of one monolith there |
| memory cluster | ID list with an unassigned centroid here; centroid + tensor list there |
| emotion | dict keyed by `EmotionType` here; 4-vector plus 8 named mixtures there |
| attachment | absent here; 4 softmax styles + trust EMA there |
| growth | `clone_with_growth` here; `grow_layer` widening in place there |

Output lands in `CONTEXT.md` and `docs/adr/`.

### Stage 3 — Chart

**Why `/wayfinder` and not `/grill-with-docs`.** The resurrection was an idea that fits in one
session. The merge is not: it spans two codebases, ~4,000 lines of predecessor source, and a set
of decisions that have to be made in a particular order. Wayfinder produces **decisions, not
deliverables**, on a shared map — which is exactly the shape of what is missing.

Decisions the map has to close, roughly in dependency order:

1. **Mechanism or instrument?** If attachment style is a softmax over four learnable outputs,
   the four styles were assumed, not discovered, and any result restates the prior. The
   alternative is a thinner substrate with the predecessor's four-way classifier attached as a
   *measurement lens*. Some things must be mechanisms — defense activation has to gate output to
   matter. This decision constrains every port below it, so it goes first.
2. **How do psychological peers see global state?** The predecessor's 398-d concatenation is not
   incidental — it is how theory-of-mind and the defense heads got global context. In a peer
   model each network sees a narrow message instead. Needs a composite mind-state message type.
3. **Which stage ladder**, and what replaces cosine scoring — scale-invariance is precisely why
   the predecessor's curriculum could never advance (measured ceiling 0.2905 against a 0.85
   threshold).
4. **Which Mother.** Current template bank with no memory, or the Agent SDK design in
   `ideas.md` — persistent sessions, four subagent personalities, session forking for
   counterfactual runs.
5. **Port order and admission criteria** — what must be true before a capability is allowed in.
6. **What "emergence" means operationally** here, and how it would be measured.

**3.3 prototypes.** Reach for one whenever a decision needs a runnable answer rather than an
argument — most likely for decision 1 (does attachment-style-like clustering appear on its own
when the Mother's consistency is varied?) and decision 2 (does the composite message feel right).
A prototype lives in its own directory, so `/handoff` out and `/handoff` back, and keep it on a
`prototype/<name>` branch as a primary source pointed at from the issue.

**3.4** — wayfinder **hands off, it does not build**. Merge onto the main flow at `/to-spec`.
Going straight to `/implement` throws away the linked detail the map spent its effort producing.

### Stage 4 — Plan

`/to-spec` collapses the map's decisions into a buildable plan; `/to-tickets` splits it into
tracer bullets with blocking edges. **One window for both** — that is the context-hygiene rule,
and the spec is worthless if the tickets are cut from a compacted memory of it.

### Stage 5 — Build

**Attachment first**, and not arbitrarily: smallest surface (4 styles, a trust EMA, two small
MLPs); its input is the Mother's behaviour, the one leg already verified working end to end; its
consumer already exists in the emotions network; it is observable from outside, so it respects
the `ObservableState` boundary; and it is the substrate the conditions this project exists to
study are built on.

`/clear` between every ticket. Each ticket is self-contained by construction, so the previous
one's context is disposable — carrying it forward buys nothing and costs the smart zone.

**Do not port** as-is: `child_model.py` (monolith, hardcoded 398, unregistered parameters),
`replay_system.py` (breaks permanently after one prune-then-refill), `symbol_grounding.py`
(float-tuple dict keys, CUDA-hardcoded — and `utils/llm_module.get_embeddings` is already a
better substrate for it than `hash(token) % 1000`), `training_system.py` (never imported by
anything, `update_loss_weights` raises on grad-tracking tensors outside every `try`). Take
`training_system.py`'s *shape* — four weighted losses, stability-filtered checkpoints, early
stopping — and reimplement.

**Protect the `ObservableState` boundary.** The Mother sees mood, energy, expressed needs,
vocalization and stage — never internals. That constraint is what makes this a developmental
simulation rather than supervised learning, and it is what makes stage 6 mean anything. Check
every merge decision against it.

### Stage 6 — Experiment

Vary the Mother's personality traits, response interval and technique selection; measure the
effect on stage trajectory, belief consistency and emotional stability. This is the point of the
project ([`docs/ideas.md`](../ideas.md)) and is unreachable until stages 1–5 land. Re-enters the
main flow at `/grill-with-docs`, because by then it is an idea that fits in one session again.

---

## Skills deliberately not on this route

| Skill | Why not |
|---|---|
| `/triage` | Only for issues we did not create. Revisit if external reports arrive. |
| `/grill-me` | Stateless variant. We have a working directory, so `/grill-with-docs` is strictly better. |
| `/to-questionnaire` | Nothing is blocked on another person's knowledge. |
| `/wizard` | No infrastructure, credentials or third-party dashboards in scope. |
| `/resolving-merge-conflicts` | Reach for it if a rebase conflicts — not a planned step. |
