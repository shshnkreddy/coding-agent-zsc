# Sixteen-card Yokai policy API

Import `from llm_policy import yokai16_primitives as p`.

{{POLICY_STRUCTURE_INTERFACE}}

Keep the starter's exact metadata: adapter `yokai`, version 1, two agents, four
colors, sixteen cards, a 10×10 grid, seven hints, 23 nodes, 15 node features,
and 2325 actions. Use `init_agent()` and `agent_act(obs, reset, memory, rng)`.

## Observation and private memory

`YokaiObservation` contains `node_features[23,15]`, `adjacency_matrix[23,23]`,
`positions[16,2]`, and `legal_actions[2325]`. Card IDs are stable across moves.
Node rows 0–15 are cards; 16–22 are hints. Feature columns are own visible
colors 0:4, public hint colors 4:8, position 8:10, lock/placement 10,
partner-seen indicator 11, active player 12, substep 13, and stable ID 14.
Zero color vectors mean unknown.

`decode_observation(obs)` returns `YokaiFacts` with `visible_colors[16]`
(-1 unknown), `hint_colors[7,4]`, `positions[16,2]`, `hint_positions[7,2]`,
`locked[16]`, `hints_placed[7]`, `seen_by_partner[16]`, scalar `active` and
`substep`, adjacency and legal actions. Hint membership never proves identity.

`init_memory()` returns `YokaiMemory(known_colors=int32[16])`, initially -1.
`facts, memory = observe(obs, reset, memory)` records only personally visible
colors and clears previous-episode memory on reset. Both seats process every
transition, including inactive turns, with separate private memory. Consult the
task specification for perfect versus standard observation-memory semantics.

## Geometry, actions and events

All helpers are pure JAX and support JIT/vmap with fixed-shape inputs/outputs.

- `card_adjacency(positions)` returns orthogonal `bool[16,16]` adjacency,
  excluding self-loops and hint edges.
- `connected_components(adjacency, mask)` returns component `labels[N]`,
  `sizes[N]` and scalar `count`. Labels are smallest member IDs, -1 outside
  the mask. `is_connected` is false for an empty selection.
- `color_components(positions, colors)` returns labels/sizes `[4,16]` and
  counts `[4]`; unknown colors are excluded.
- `is_solved(positions, colors)` requires a complete assignment, exactly four
  cards per color, and one connected component per color.
- `hypothetical_move(positions, card, x, y)` returns new positions and adjacency.
  Supply card 0–15 and coordinates 0–9. It changes geometry only and does not
  check legality, locks, connectivity, hidden identities, or turn progression.
- `legal_move_mask(facts)` reshapes current native legality to `[16,10,10]`.
- `legal_hint_placement_mask(facts)` returns `[7,16]` legality by target card.
- `hint_card_links(facts)` returns public hint/card attachment links `[7,16]`.
- `init_observation_snapshot()` creates empty fixed-shape facts.
  `observation_changes(previous,current,reset=False)` returns boolean changes
  in hint reveals/placements, locks, moves, visible colors and partner-seen
  indicators. Save current facts after every transition. Reappearing visible
  colors do not necessarily imply a new peek. On reset only current visible
  colors are reported; prior-episode events are discarded.

Action encoders: `observe_card(i)=1+i`; `move_card(i,x,y)=17+100*i+10*x+y`;
`reveal_hint(h)=1617+h`; `place_hint(h,x,y)=1624+100*h+10*x+y`;
`end_game()=2324`; `noop()=0`. Encoders do not check legality. Use
`is_legal(mask,action)`, `sample_legal(mask,rng)` and the native action mask.
`make_trace(action,phase_id=0,decision_id=0)` creates a `PolicyTrace`.

## Optional literal card beliefs

`literal_card_belief(known_colors)` represents the uniform shuffled-deck prior
conditioned only on direct card identities. `YokaiBelief` contains
`known_colors: int32[16]` and scalar `valid`. It never interprets hints or
partner actions. A complete joint distribution has 63,063,000 assignments;
this compact representation supports exact literal marginals and exact draws
without materializing that space.

- `condition_on_colors(belief,observed_colors)` merges direct evidence (-1 means
  no new evidence). Contradictions, invalid colors and impossible counts mark
  the belief invalid. Repeated identical evidence has no extra effect.
- `card_marginals(belief)` returns exact `[16,4]` probabilities. Known cards
  are point masses; each unknown card has the remaining color proportions.
  Invalid beliefs yield zero arrays, which are not valid distributions.
- `sample_assignment(belief,rng)` draws an exact joint `int32[16]` assignment
  conditional on direct evidence, respecting all four color counts. Invalid
  beliefs return all -1. Draws are hypotheses, never hidden observations.
- `sampled_solved_probability(positions,belief,rng,num_samples=128)` returns a
  Monte Carlo estimate under that literal prior; it is not an exact query.
  `num_samples` must be a positive static integer. Invalid beliefs yield NaN.
- `hint_correct_probability(belief,hint_colors,card)` is an exact marginal
  query for whether the given card is in a hint's `bool[4]` color set. Unknown
  hints, invalid card indices and invalid beliefs yield NaN. It does not update
  the belief or presume the hint is truthful.

The compact literal belief does not represent arbitrary behavioral likelihoods.
For richer models, use your own fixed-shape JAX memory/particles and reset it
between episodes. `init_belief_model_template()` returns `()` and
`update_belief_model(facts,events,literal_belief,model_state)` returns its belief
and state unchanged. These optional templates prescribe no signaling convention,
hint-trust level, risk threshold, planning algorithm or strategic action order.
Direct observations cannot be overridden by inferred intent.

## Runtime boundary

Return an integer scalar action, same-structure/shape/dtype private memory, and
`PolicyTrace`. Only the active seat acts. Illegal active actions and incomplete
episodes invalidate evaluation. Host I/O, callbacks, environment state and
partner memory are unavailable during policy execution. Native observations,
legal actions, reset and private randomness are the available information.
The task specification defines reward and observation-memory modes. Both sparse
and terminal-penalty returns, success, early ending and length are recorded as
evaluation metrics; they are not additional policy observations.
