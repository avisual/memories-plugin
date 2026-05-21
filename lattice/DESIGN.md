# LATTICE — Design

A compositional, self-improving reasoning substrate for small coding LLMs.

> An organism, not a prompt. Senses, muscles, memory, dreams — and small
> typed primitives that grow into bigger forms with use. Built so a 7B
> model with this body beats a frontier model in a chat window, gets
> meaningfully better every week, and lets the human step in at any
> layer of the composition.

## Context

Conventional coding harnesses give a small LLM a long text prompt with
files, history, and tool definitions, then ask it to generate code as
free-form text. Two failure modes follow: small models degrade with
long context, and free-form text generation has no structural
guarantees of correctness.

LATTICE rejects both defaults. It builds the LLM a body — typed
senses, typed actions, a world model it can imagine in, a Hebbian
memory that learns from outcomes, and a population of specialist
critics it can consult without rendering text. Then it lets that whole
system evolve: the action vocabulary grows from observed behavior, a
fast distilled policy is trained continuously from successful traces,
and a personalized model of the user's taste conditions every
decision.

The [`memories-plugin`](../) codebase is the seed: a Hebbian graph with
spreading activation, embeddings, decay/consolidation, and hookable
lifecycle. LATTICE is the organism that grows around it.

## The spine: compositional emergence

> **Everything in LATTICE is a small typed primitive that composes
> into bigger primitives, by the same rules, at every level.**

The same Hebbian co-activation rule that links two memory atoms also
extracts a macro from two recurring actions, also fuses two related
plan branches, also pulls two specialist critics into a debate, also
turns two macros into a project recipe. *One rule, recursively
applied.*

Why this matters in practice:

- The system has **no fixed ceiling**. Today it knows 50 actions.
  Next month it knows 80, because it composed 30 macros from
  observed patterns. The month after, recipes emerge from macros.
  A year from now, project idioms emerge from recipes. Same machine,
  same rule.
- The user can **plug in at any rung**. They can accept a single
  diff, approve a proposed macro, edit a recipe, or tell the system
  what idiom this project follows. Their interaction at any level
  feeds back via the same Hebbian update.
- The Hebbian force is the **gradient of composition**. Things that
  co-fire wire together. Things that wire together get promoted.
  Things that get promoted become the new primitives for the next
  layer.

## The three commitments

1. **WORKS** — runs end-to-end on day 1 with a small model on real
   coding tasks, with *deterministic* guarantees about output
   validity.
2. **LEARNS** — every interaction (model action *or* human touch)
   produces a usable Hebbian update, stored in the graph, applied
   without retraining the LLM.
3. **IMPROVES** — the system extends itself: vocabulary, fast
   apprentice, user-taste model, world model all grow with use.
   The LLM stays fixed; the organism evolves.

## Architecture: ten organs on one substrate

```
                  ┌──────────────────────────────────────────┐
                  │       PROJECT LATTICE STORE              │
                  │ symbols ↔ atoms ↔ outcomes ↔ tastes ↔    │
                  │   macros ↔ recipes ↔ idioms (one graph)  │
                  └──────────────────────┬───────────────────┘
                                         │
       ┌──────────┬──────────────────────┼──────────────────────┐
       │          │                      │                      │
    SENSE      STEER                  ACT (LLM)             WORLD MODEL
  (typed obs) (atom-vectors)        (typed actions)        (imagine outcomes)
       │          │                      │                      │
       └────────┬─┴──────────────────────┴──────────────────────┘
                ▼
            POPULATION  ──►  VERIFY (real)
        (evolutionary plans) (types/tests/sandbox)
                │                      │
                └──────────► LEARN ◄───┘
                          (Hebbian: atoms, vectors,
                           value, counterfactuals)
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
               DISTILL       EVOLVE         TASTE
              (apprentice) (DSL growth)   (user model)
                  │             │             │
                  └─────────────┼─────────────┘
                                ▼
                      HUMAN INTERFACE
                  (every layer of zoom)
```

### Organ 1 — The lattice store

One graph, seven node kinds:

| Node     | What it represents              | Where it composes from   |
|----------|---------------------------------|--------------------------|
| Symbol   | code element (fn, class, var)   | (extracted by tree-sitter) |
| Atom     | episodic memory item            | (LLM output / hook capture) |
| Outcome  | result of an action             | (verify gate) |
| Taste    | user style preference           | (mined from accept/revert) |
| Macro    | a verb composed from primitives | **co-firing actions**  |
| Recipe   | a workflow composed of macros   | **co-firing macros**   |
| Idiom    | a project-wide pattern          | **co-firing recipes**  |

Built as an extension of memories' SQLite schema (`storage.py`).
Spreading activation (`retrieval.py:380`) traverses every edge kind
natively. Hebbian rules in `learning.py` apply to every edge kind.
**One rule, every level.**

### Organ 2 — SENSE (typed observations)

The LLM never sees a file. Per turn it receives:
- **Active subgraph** of the lattice (≤30 nodes) selected by
  spreading activation,
- **State snapshot**: position in plan-DAG, last outcome (typed),
- **Memory salience**: top-K atom IDs (content lives in Organ 4),
- **Scratchpad vector** (continuous, persists across turns).

A few hundred structured tokens replace tens of thousands of file
tokens. Multi-language via tree-sitter.

### Organ 3 — ACT (typed action DSL)

Output vocabulary is a typed DSL — currently ~50 verbs, grows over
time via Organ 9. Each verb is typed; each action carries a
`confidence ∈ [0,1]`. Sampled via constrained decoding (Outlines for
open weights; structured-output APIs for hosted). Invalid syntax,
undefined symbols, unresolved imports — deleted by construction.

A deterministic *action compiler* turns each typed action into the
AST mutation + a textual diff for the user.

### Organ 4 — STEER (atoms as activation bias)

Atom embeddings aren't injected as text. The top-K activated atoms
are summed (weighted by activation) into a steering vector applied
to the LLM's residual stream at selected layers (open-weight) or
rendered as a compact priming preamble (hosted fallback).

Steering vectors themselves get Hebbian-EMA updates: atoms active in
successful turns nudge their vector toward the "direction of
success." Over months, the bank of steering vectors becomes a
codebase-personalized prior.

### Organ 5 — WORLD MODEL (imagination)

Before any action is materialized:
- **Type-shape simulator**: applies the action to a typed abstract
  codebase; computes type-lattice delta in μs.
- **Caller-impact simulator**: walks call graph; reports affected
  sites with categorical confidence.
- **Test-impact simulator**: from a coverage map, picks the minimal
  test subset affected. Run 7, not 7000.
- **Expected-free-energy signal**: high when the action's outcome is
  ill-predicted. The system is biased to act in ways that *reduce
  uncertainty over time* — active inference.

Imagination is cheap (graph ops). Real verify (Organ 7) is ground
truth.

### Organ 6 — POPULATION (evolutionary plan search)

Multi-step tasks are searches:
- LLM emits N=4 typed candidates at each choice point.
- Each seeds a branch in a beam of ≤8 active plans.
- Branches scored by world-model + recalled-atom value + heuristic.
- Bad branches die; winners continue; macro-recombination splices
  prefix/suffix across siblings when the world model type-checks
  the splice.

Beam width = compute knob. The user sees the surviving branch.

### Organ 7 — VERIFY (silent ground truth)

Surviving branches face real verify: type checker, lint, narrow test
subset, in a sandbox, ≤2s budget. Failures don't reach the user —
they become atoms (and counterfactuals).

### Organ 8 — DISTILL (the apprentice)

Every successful trace is logged. A small distilled policy net
(single transformer block + action head, <100M params, CPU is fine)
trains continuously in the background.

Within weeks, the apprentice handles 40–80% of decisions without
invoking the LLM. Per-user, overfit-on-purpose to *this* codebase.

### Organ 9 — EVOLVE (the DSL grows itself)

- Cluster traces by action-sequence + intent-embedding.
- Clusters with ≥10 successful instances and ≥0.8 sequence
  similarity → propose `Macro{Verb, schema, body}`.
- Macros that fire together with high success → propose `Recipe`.
- Recipes that recur across files → propose `Idiom`.

### Organ 10 — HUMAN INTERFACE (zoom-in at every level)

Six surfaces, one shared store:

| Surface         | What the human does                       | Update fed back        |
|-----------------|-------------------------------------------|------------------------|
| Diff lens       | Accept / edit / revert each code change   | Taste atoms, atom value  |
| Plan view       | See the plan-DAG; drag, prune, comment    | Plan-DAG atoms          |
| Atom inspector  | Read / edit / soft-delete memory atoms    | Direct atom edit         |
| Promotion queue | Approve proposed macros, recipes, idioms  | DSL growth               |
| Intent bar      | One-line NL ("add rate-limit")            | Initial task atom        |
| Why trace       | "Why did you do this?" → graph trace      | Tagged as inspected      |

Each surface is a different zoom level on the *same* lattice.

## The reasoning loop, in 12 lines

```
loop until task done:
    obs       = sense(active_subgraph, last_outcome, scratchpad)
    seeds     = spread_activate(obs, lattice)           # over all 7 node kinds
    steer     = weighted_sum(a.steer_vec for a in seeds)
    if apprentice.confident(obs, seeds):
        action = apprentice(obs, steer)                 # the cheap path
    else:
        cands  = llm_propose(obs, steer, N=4)           # constrained decode
        imag   = [world_model.simulate(c) for c in cands]
        action = population.select(cands, imag, seeds)
    outcome   = verify(action_compiler(action))         # silent ground truth
    lattice.learn(obs, action, outcome, seeds)          # Hebbian on every edge
    apprentice.train_step(obs, action, outcome)
    evolver.maybe_promote()                              # macro → recipe → idiom
```

## What's genuinely new

1. **Hebbian-updated activation steering from an episodic graph.**
   The bias the model receives learns from this codebase's outcomes,
   with no LLM fine-tuning.
2. **One graph for code, episodic memory, outcomes, taste, and
   composed verbs.** Spreading activation routes across all of them.
3. **A typed action DSL as the model's only code-modifying channel,
   with a world model for cheap imagination.**
4. **An always-on distilled apprentice** trained continuously from
   real outcomes, personalized per user.
5. **A DSL that grows itself**: macros → recipes → idioms by the
   same Hebbian co-firing rule that links memory atoms.
6. **A human interface that is the same graph at different zooms**,
   so every human touch is a learning signal.

## Build sketch — incremental, every milestone shippable

**Months 1–2 — Substrate + v0 loop (WORKS)**
- Lattice schema (extend memories SQLite, all 7 node kinds).
- Tree-sitter symbol graph (Python + TS).
- World model v0 (type-shape + call graph + test-impact).
- Action DSL v0 (20 verbs + constrained decoding).
- v0 reasoning loop: argmax over world-model heuristic.
- Diff lens + intent bar in the human interface.
- **Milestone**: SWE-bench-Lite subset with Qwen-Coder 7B beats a
  prompt-based baseline with the same model.

**Months 3–4 — Population + steering + plan view (LEARNS)**
- Population-based plan search (beam ≤8, crossover, selection).
- Activation patching pipeline; steering vectors per atom.
- Outcome-conditioned Hebbian + value + counterfactual updates.
- Plan view + atom inspector in the interface.
- **Milestone**: 5pp pass@1 lift from steering on tasks with prior
  related atoms.

**Months 5–6 — Apprentice + verify + taste + promotion queue (IMPROVES, v1)**
- Distilled policy net + continuous training pipeline.
- Real verify path (mypy, tsc, narrow tests, sandbox).
- Taste mining from git history (accept/revert).
- Promotion queue UI for macros.
- **Milestone**: ≥40% of decisions handled by apprentice with ≥95%
  match-to-LLM-choice on held-out trace.

**Months 7–12 — Evolution + multi-specialist debate + multi-language**

**Months 13–18 — Federation + benchmarks + ecosystem**

## Verification gates

1. **Action DSL soundness**: property-based — every random valid
   action produces parsable code, every time, zero exceptions.
2. **World-model calibration**: break-prediction precision ≥90%,
   recall ≥80% on 1000-sample real-vs-imagined.
3. **Steering ablation**: steering ON vs OFF ≥5pp on tasks with
   prior related atoms.
4. **Curriculum effect**: pass@1 at session 0/50/200/1000 — monotonic
   upward, ≥10pp by 200 sessions.
5. **Apprentice quality**: ≥95% LLM-choice agreement at confidence
   ≥0.8 on held-out trace.
6. **Macro acceptance rate**: ≥70% of auto-proposed verbs approved
   after 3 months on a real codebase.
7. **Human-touch impact**: each interface action measurably shifts
   later recall and action selection.

## Reuse from `memories-plugin`

- `src/memories/retrieval.py:296` — `Brain.recall` and
  `_spread_activation`; extended to traverse all node kinds.
- `src/memories/storage.py` — extend schema; SQLite + vec + FTS5 is
  exactly the right substrate.
- `src/memories/learning.py` — Hebbian + auto-link + `rate_recall`
  EMA already shapes the right update; we add counterfactual,
  steering-vector, taste, and macro-promotion branches.
- `src/memories/embeddings.py` — embed symbols, atoms, intents,
  tastes, macros — everything that lives in the graph.
- `src/memories/consolidation.py` — decay/prune/promote cycle works
  on every new node kind by the same rules.

## Risks

- **Activation steering is per-arch work.** Ship the open-weight path
  for one model family first; hosted-model fallback (text priming)
  still benefits from every other organ.
- **World-model fidelity ceiling.** Imagination is for search; real
  verify is ground truth.
- **Distillation could overfit to bad behavior.** Train only on
  traces whose final outcome was good, recency-weighted.
- **Macro/recipe promotion could explode the DSL.** Promotion queue
  requires a human click.
- **Interface complexity.** Diff lens + intent bar are mandatory at
  v1; others ship as the corresponding organ matures.
