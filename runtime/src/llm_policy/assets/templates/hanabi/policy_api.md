# Hanabi policy API

Import `from llm_policy import hanabi_primitives as p`. All helpers are pure
JAX with fixed shapes. No runtime I/O, subprocesses, environment imports, mutable
globals, or host callbacks are allowed. Development commands are permitted
within the task bundle under AGENTS.md.

## Policy inputs and returns

{{POLICY_STRUCTURE_INTERFACE}}

Each act function receives
`HanabiObservation(features=float[658], legal_actions=bool[21])`, `reset`, private
memory and a PRNG key. Return `(integer_scalar_action, next_memory, PolicyTrace)`.
Memory must preserve its initializer's PyTree, leaf shapes and dtypes. The runner
resets it between episodes. Roles never share memory. Both roles are called on
every global step; only the active role's action is executed. Illegal active
actions invalidate evaluation. An inactive action is replaced by native no-op.

## Facts and controller memory

`p.init_knowledge()` initializes `HanabiKnowledge`: `possible[2,5,5,5]`,
`previous_deck_remaining`, and `episode_step`. Player axis is observer-relative:
self=0, partner=1. This optional controller tracks literal public knowledge;
it is not a strategy. Additional policy-specific arrays can live alongside it.

```python
facts, knowledge = p.observe(obs, reset, knowledge)
```

Call once per observation/transition, including inactive turns. This combines
`decode_observation(obs)` and `update_knowledge(facts, reset, knowledge)`.
Do not apply the same last-action event twice. At reset there is no previous event.
`episode_step` counts observations processed (1 on the initial observation).

Facts include `partner_hand[5,5,5]` one-hot identities, `occupied[2,5]`,
`fireworks[5]` heights 0..5, `discards[5,5]`, `deck_remaining`, `info_tokens`,
`life_tokens`, `is_my_turn`, and `legal_actions[21]`.

Public event fields are `has_last_action`, `last_actor`, `last_kind`,
`last_target`, `last_color`, `last_rank`, `last_matches[5]`, `last_slot`,
`last_card[5,5]`, `last_play_scored`, and `last_play_added_token`.
Only interpret fields applicable to that event; an argmax field can be zero
when absent. Kinds are `PLAY`, `DISCARD`, `HINT_COLOR`, `HINT_RANK`, `NOOP`.

`upstream_v0[2,5,5,5]`, `positive_colors[2,5,5]` and `positive_ranks[2,5,5]`
expose native inputs unchanged for inspection. V0 is NOT the exact posterior.
The positive flags do NOT encode negative hints. Empty-slot detection uses the
native zero-row representation; literal constraints are reconstructed from events.

## Literal hints and shifting slots

`apply_hint_constraints(possible[5,5,5], matches[5], kind, value)` intersects
the candidate mask: matching cards keep the named attribute; unmatched cards
exclude it. Values are zero-based color/rank IDs. Only hint kinds are accepted
by this contract. Existing exclusions remain excluded.

`hint_matches(visible_hand, kind, value)` returns the matching slots.
`literal_hint_effect(possible, visible_hand, kind, value)` returns
`(matches, updated_possible)`. It does not evaluate the hint's strategic meaning.

`remap_after_card_removal(per_slot[5,...], slot, replacement[...])` shifts later
entries left and appends the supplied replacement. Use this for private tags as
well as beliefs. Slots follow card order, not permanent card identities.
`update_knowledge` handles this for both players' literal masks automatically:
a drawn card starts unconstrained; an empty replacement has no possibilities.
The removed card's negative and positive hints never transfer to a fresh card.

## Card counting and exact literal beliefs

`public_remaining_counts(facts)[5,5]` subtracts discards and successful plays
from initial multiplicities. `unseen_counts(facts)[5,5]` additionally subtracts
the visible partner hand: it describes own hidden hand PLUS undealt deck.
Neither helper reads deck order, own identities, or the partner's private data.

```python
belief = p.own_hand_belief(facts, knowledge)
```

This is shorthand for `literal_hand_marginals(knowledge.possible[0],
unseen_counts(facts), facts.occupied[0])`; the lower-level function also accepts
explicit literal masks, counts and occupancy for hypothetical calculations.

The result has `probabilities[5,5,5]`, `feasible[5,5,5]`, scalar `valid` and
`normalizer`. `feasible_card_mask(belief)` returns the same feasibility array.
Occupied-slot marginals sum to one for valid evidence; empty slots are zero.
Impossible evidence gives `valid=False` and zero arrays, NOT uniform beliefs.
Counts must be finite nonnegative integers and masks must match the five-slot
contract. Check validity before interpreting probabilities.

Inference is exact under uniformly dealt physical cards, conditioned on supplied
literal masks and counts, including sampling without replacement. It accounts
for competition among own slots for limited copies. A 32-subset dynamic program
avoids enumerating all hands. It does NOT infer intent from which hint was
chosen, learn partner conventions, or model a partner's policy. It is therefore
not a full posterior conditioned on strategic action selection.

`knowledge.possible[1]` is the partner's PUBLIC literal constraints, not its
private posterior. The conditional partner helper below does not expose hidden
cards or recover the partner's actual private belief.

## Joint own-hand samples

```python
sample = p.sample_own_hand(facts, knowledge, rng)
```

Returns `HanabiHandSample` with integer one-hot `hand[5,5,5]`, scalar `valid`,
and scalar `log_probability` of that whole identity assignment. Empty slots are
zero. Invalid evidence gives `valid=False`, an all-zero hand and log probability
`-inf`; never use an invalid sample as a hypothesis.

Sampling uses the same exact literal distribution as `own_hand_belief`, preserving
limited-copy dependencies and all slots' positive/negative constraints. It does
not independently sample the marginals or reveal your actual cards. The lower-level
`sample_literal_hand(possible[5,5,5], counts[5,5], occupied[5], rng)` accepts the
same evidence as `literal_hand_marginals`. Both use subset dynamic programming;
the sampler backtracks the partition table rather than rejecting random hands.

For multiple samples, split the PRNG key and `jax.vmap` the sampling call. Reusing
a key reproduces a draw. The sample count must be static under JIT; larger batches
cost more computation. Sampling itself neither searches actions nor infers intent.

## Partner's belief about their own hand

```python
sample = p.sample_own_hand(facts, knowledge, rng)
partner_belief = p.partner_hand_belief(facts, knowledge, sample.hand)
# Use only when sample.valid and partner_belief.valid are true.
```

This returns a `HanabiBelief` answering: **if my hand were this hypothetical hand,
what would the partner's literal own-hand belief be?** The partner can see your
hand; you cannot. The helper subtracts the hypothetical hand from PUBLIC remaining
counts and applies `knowledge.possible[1]` with `facts.occupied[1]`. It does not
subtract the partner's actual hand from their belief counts, since they cannot
see it. Candidate own hands must be one-hot in occupied slots, zero in empty
slots, and compatible with your literal masks and available counts; otherwise
the result is invalid with zero arrays and normalizer.

The input must be a sample or a constructed hypothesis from permitted evidence,
never your actual hidden identities obtained from runtime state. Your visible
partner hand is not a substitute for the hypothetical own hand. You can first
apply `literal_hint_effect` to the partner's masks and pass an updated knowledge
object to examine the literal effect of a candidate clue.

Repeat over own-hand samples to represent uncertainty about the partner's belief.
An average of those conditional marginals is a Monte Carlo estimate of your
expected partner belief, not their actual belief. For nonlinear decisions, evaluate
your partner model separately on each conditional belief before averaging outcomes.
These beliefs use current literal masks and counts, not an inferred partner policy,
full private observation history, or a model of why an action was chosen. Modeling
intent still requires your own hypotheses and action likelihoods; no helper supplies
a play/save convention or assumes the partner uses your code.

## Optional belief context and partner-model template

`build_belief_context(facts, knowledge, rng, num_samples=N)` bundles:

- `own_literal_belief`: the usual `HanabiBelief` for your hand.
- `own_hand_samples`: a batched `HanabiHandSample`, with `hand[N,5,5,5]`.
- `partner_literal_beliefs`: a batched `HanabiBelief`, with
  `probabilities[N,5,5,5]`, conditional on the corresponding sampled own hand.
- `valid[N]`: usable sample/partner-belief pairs, including own-belief validity.

`N` is a positive static Python integer chosen by you; when JIT-compiling the
builder directly, mark `num_samples` static. The helper is optional, has no default
sample count, and does not run automatically. Valid independent draws have equal
Monte Carlo weight: do not weight them again by their sampling probabilities.
These are freshly sampled hypotheses, not persistent particle identities.

The primitives module also contains two optional copy-and-edit functions:
`init_partner_belief_model_template()` and
`update_partner_belief_model_template(facts, reset, belief_context, model_state)`.
Copy them into your editable `custom_primitives.py` if useful. Keep immutable
`context/hanabi_primitives.py` unchanged. The provided initializer returns empty
state; the update returns `(belief_context.partner_literal_beliefs, model_state)`
unchanged. **This is a literal-only reference, not an implemented intent model.**

You may design your own representation, hypotheses, parameters and updates, or
skip this interface entirely. Literal partner beliefs are an optional reference,
not an assumption that the partner reasons this way. A custom model may extend,
reweight or replace that reference using permitted evidence. Model what the partner
believes separately from how they act; neither their actual private belief nor
their strategy is directly supplied. There are no predefined conventions or priors.

Store custom model state alongside literal knowledge in private policy memory.
Keep its PyTree/shapes/dtypes fixed, reset between episodes, and process each
transition once, including inactive turns. If interpreting an observed action,
retain the relevant pre-action context rather than accidentally using only the
post-action belief. Sample indices do not track the same hypothetical hand across
calls. Custom model outputs remain hypotheses, not observed facts or hidden truth.

## Card properties

`card_properties(facts)` returns bool[5,5] identity masks:
`playable` (next rank), `already_played`, `unreachable` (a necessary unplayed
rank is exhausted), and `critical` (last remaining copy of a still-useful card).
`card_property_probabilities(belief, facts)` sums the supplied marginals over
these masks, returning float[5] per property plus `valid`. These quantities
are not action advice, risk thresholds or a recommended discard order.

## Actions, legality, and traces

`play(slot)`, `discard(slot)`, `hint_color(color)`, and `hint_rank(rank)` encode
native actions. Arguments must be 0..4; helpers do not repair invalid indices.
`noop()` is legal only while inactive. `decode_action(action)` returns
`(kind, detail)` for valid action IDs; no-op's detail is unused.

`legal_actions(facts)` returns bool[21]. `is_legal(mask, action)` checks bounds
and membership. `masked_argmax(values[21], mask)` selects your highest-valued
legal action (lowest-index tie break). `sample_legal(mask, rng)` samples uniformly.
Both selectors return -1 for an empty mask; they do not supply a strategy.

`make_trace(action, phase_id=0, decision_id=0)` constructs PolicyTrace. Kind and
detail describe the emitted action; status=1 means emission, not successful
environmental effect. Confirm outcomes from the NEXT observation. The generic
trace field `movement_blocked` is always false for this non-spatial task.
`select_pytree(condition, yes, no)` selects matching fixed-shape leaves.

## Evaluation

Use `policy-eval validate`, `score`, and `trace` as described in AGENTS.md.
`python rollouts.py` uses the same observation-safe evaluator for exploratory
measurements. Only official scoring produces selectable candidate evidence.
Incomplete episodes and illegal active actions are errors, not ordinary returns.
Search wall time is distinct from game turns and evaluation safety limits.
