# Yokai policy API

Import `from llm_policy import yokai_primitives as p`.

{{POLICY_STRUCTURE_INTERFACE}}

`YokaiObservation` is a PyTree containing `node_features: int8[13,13]`,
`adjacency_matrix: uint8[13,13]`, `positions: int32[9,2]` and
`legal_actions: bool[1068]`. Positions are integer grid coordinates.
Card IDs remain stable across moves.

`p.decode_observation(obs)` returns `YokaiFacts`: visible color IDs `[9]` (-1
unknown), public hint-color membership `[4,3]`, card positions `[9,2]`, hint
positions `[4,2]`, locked cards `[9]`, placed hints `[4]`, partner-seen indicators
`[9]`, scalar active/substep, adjacency `[13,13]` and legal actions `[1068]`.
Unknown hints have no visible color membership. A hint may be incorrect; its
colors do not establish the underlying card's identity.

`p.init_memory()` returns `YokaiMemory(known_colors=int32[9])`, initially -1.
`facts, memory = p.observe(obs, reset, memory)` resets at episode boundaries and
records only colors personally observed, indexed by stable card ID. This works
in both memory modes. It does not infer colors from hints or partner actions.
You may implement additional private fixed-shape memory using pure JAX.

## Optional graph and movement tools

These are queries and encoders, not a planner or move-ranking policy. All
operations are pure JAX at runtime, with fixed-shape outputs and JIT/vmap support.
They target this adapter's nine cards and three colors.

- `card_adjacency(positions)` returns `bool[9,9]` orthogonal adjacency, without
  self-loops or hint nodes. Do not use the full 13-node observation graph to
  compute color clusters: that graph also includes hint edges.
- `connected_components(adjacency, mask)` takes a square adjacency and a
  `bool[N]` selection. Returns `YokaiComponents(labels=int32[N], sizes=int32[N],
  count=int32 scalar)`. A component's label is its smallest card ID; `sizes[i]`
  is the size of card i's component. Excluded cards have label -1 and size 0.
  Components use the undirected induced graph; labels are bookkeeping IDs, not
  strategic targets or a suggested card ordering.
- `is_connected(adjacency, mask)` is false for an empty mask and true for a
  singleton. It checks only the selected cards.
- `color_components(positions, colors)` returns batched component fields of
  shape `[3,9]`, `[3,9]`, and `[3]`. Unknown color -1 is excluded, not treated
  as another color. Connected known cards do not prove a whole color is solved.
- `is_solved(positions, colors)` requires a complete assignment with three
  cards of each color and checks that every color forms one connected component.
  A hypothetical assignment is not an observation of hidden truth.
- `legal_move_mask(facts)` reshapes the native current-substep mask to
  `bool[card=9,x=9,y=9]`. It is not a prediction of future-substep legality.
- `hypothetical_move(positions, card, x, y)` returns `(new_positions,
  new_adjacency)`. It only changes geometry; it does not validate a move, advance
  the turn, or simulate hidden information. Inputs require card 0..8 and x/y
  0..8. Consult native legality for locks, occupancy, connectivity and substep.
  Moves relocate a card directly; there is no walking path.
- `legal_hint_placement_mask(facts)` returns native current-substep legality
  as `bool[hint=4,card=9]`. Convert the target card to its current coordinates
  when calling `place_hint`.

## Optional joint card-identity beliefs

`YokaiBelief` is a distribution over complete color assignments to stable card
IDs: `assignments: int32[1680,9]`, `probabilities: float32[1680]`, and scalar
`valid: bool`. There are 9!/(3!3!3!) = 1680 assignments with three copies of each
color. Rows are hypotheses, not hidden observations. Their order has no strategic
meaning. Helpers preserve this support; modifying the assignments array directly
is outside the supported contract.

- `literal_card_belief(known_colors)` starts with the uniform shuffled-deck
  prior and conditions on personally observed colors (`int32[9]`, -1 unknown).
  It respects the exact color counts and does not condition on hints, locks,
  locations, or partner actions. Passing `memory.known_colors` retains personal
  evidence in either observation-memory mode.
- `condition_on_colors(belief, observed_colors)` adds hard identity evidence,
  with -1 meaning no new evidence. Reapplying the same observations is idempotent.
  Contradictory identities, impossible counts, or values outside -1/0/1/2 yield
  `valid=False` and zero probabilities, never a silent uniform fallback.
- `card_marginals(belief)` returns `float32[9,3]`. Joint dependence is retained:
  observing one card affects the remaining cards. Invalid beliefs yield zeros,
  which must not be interpreted as a valid distribution.
- `reweight_belief(belief, log_likelihood)` multiplies weights by caller-defined
  likelihoods and normalizes stably. Supply `[1680]` log likelihoods (a scalar
  is broadcast); zero is neutral and -infinity excludes a hypothesis. All mass
  excluded, NaN/+infinity likelihoods, or negative/nonfinite prior weights
  produce an invalid, zero-weight belief. No hint-trust likelihood is supplied.
- `sample_assignment(belief, rng)` optionally draws one count-consistent
  `int32[9]` assignment. Invalid beliefs return all -1. Sampling is not required
  to use the representation; marginals and exact weighted queries also work.
- `solved_probability(positions, belief)` evaluates `is_solved` for every
  hypothesis and sums its probability. This is exact under the **supplied
  belief**, not a guarantee about actual hidden colors. Invalid beliefs yield
  NaN. Evaluate candidates selectively if this exact query is expensive.

The representation stores beliefs about all cards, not just unobserved ones.
Observed cards have point-mass marginals. Movement and locking do not change
card identity; the belief remains indexed by stable ID.

## Hints, locks and observation events

- `hint_card_links(facts)` returns `bool[4,9]`: public placed-hint/card
  associations reconstructed from positions. Unplaced hints have no links.
- `hint_correct_probability(belief, hint_colors, card)` computes the probability
  that a card belongs to the listed colors. Pass one hint's `bool[3]` membership
  vector, not its ID. An unknown all-false hint, invalid card index or invalid
  belief returns NaN. This query does not itself update the belief.
- `init_observation_snapshot()` creates empty fixed-shape `YokaiFacts` for
  initializing optional event-tracking memory.
- `observation_changes(previous, current, reset=False)` returns `YokaiEvents`:
  `newly_revealed_hints[4]`, `newly_placed_hints[4]`, `newly_locked_cards[9]`,
  `moved_cards[9]`, `visible_color_changes[9]`, `newly_seen_by_partner[9]` (bool).
  On reset only currently visible colors are reported; no previous episode's
  events carry over. Save `current` as the next snapshot on every global
  transition, including inactive turns. Repeated identical observations produce
  no events. Reappearance after masking can trigger `visible_color_changes`;
  it does not prove a new observation action. Partner-seen indicators have the
  configured native memory semantics, not access to the partner's private state.

A placed hint locks its card against observation and movement, even if the hint
is false. This is a hard mechanical fact. Treating its listed colors as evidence
about identity instead requires a behavioral assumption; the literal belief
does not make one. Event deltas allow a custom model to process a placement once
instead of counting the same visible token as repeated independent evidence.

## Optional custom belief-model template

`init_belief_model_template()` returns `()`. The supplied
`update_belief_model(facts, events, literal_belief, model_state)` returns
`(literal_belief, model_state)` unchanged. Neither is called by the starter.
They impose no hint convention, trust level, intent interpretation or strategy.
If useful, copy/edit these functions in `custom_primitives.py`, retaining a
fixed-shape JAX state. Both literal and custom outputs use `YokaiBelief`.

For example, optional event bookkeeping has this shape:

```python
# In your own initializer, alongside other private memory:
previous = p.init_observation_snapshot()

# On each global transition:
facts, knowledge = p.observe(obs, reset, knowledge)
events = p.observation_changes(previous, facts, reset)
literal = p.literal_card_belief(knowledge.known_colors)
# Your optional custom update goes here. Save facts as next previous.
```

For a persistent custom posterior, condition it on new direct observations and
apply each behavioral likelihood once. If rebuilding from `literal` every turn,
retain the required behavioral history in custom state instead. Reset any custom
posterior and history at episode boundaries. Do not combine both approaches in
a way that counts the same evidence twice. Inferred intent cannot override
personally observed identities or reveal partner-private information.

## Actions and runtime contract

Action constructors (zero-based card/hint/grid indices): `observe_card(card)`,
`move_card(card,x,y)`, `reveal_hint(hint)`, `place_hint(hint,x,y)`, `end_game()`,
`noop()`. They encode actions only; check the native mask for legality.
`is_legal(mask, action)` checks bounds and mask membership.
`sample_legal(mask, rng)` samples uniformly over legal actions.
`make_trace(action, phase_id=0, decision_id=0)` returns a `PolicyTrace`.

Both seats process each global transition with independent private memory.
Only the active seat acts; inactive actions are forced to zero. An illegal
active action invalidates evaluation. Return an integer scalar and preserve
the initializer's memory structure, shapes and dtypes. Host I/O, callbacks,
environment imports, hidden state, and partner memory are unavailable.

The task specification and runtime contract identify memory and scoring modes.
The configured `mean_return` is the selection objective. Additional rollout
metrics include both scoring definitions, success, early ending and length;
they describe completed episodes and are not policy observations.
