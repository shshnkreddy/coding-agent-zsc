# Two-player Hanabi: environment specification

## Goal, objects and initial state

Both players work together to build the five public fireworks, one for each color. Within a firework, ranks must be 1, 2, 3, 4, 5 in sequence, with neither repeated nor skipped ranks. Both players share the team score as episode return, with a maximum of 25. These rules specify mechanics and data formats, without assigning additional meanings to which action a player chooses.

The deck has 50 cards in five colors. Each color has 3 copies of rank 1, 2 copies each of ranks 2, 3 and 4, and 1 copy of rank 5. Colors use IDs 0..4; ranks in arrays also use IDs 0..4, representing human ranks 1..5. No color has special status.
Each episode starts with a uniform shuffle of this fixed multiset, five cards dealt to player 0 and five to player 1, leaving 40 in the draw pile. Player 0 has the first turn. Fireworks and discards begin empty; eight information tokens and three lives are available. The initial hands carry no hints.

A player's observation reveals the other player's card identities and their order, but hides their own card identities and the draw-pile order. Firework heights, discard counts, information tokens, lives, deck size, and the latest action with its public outcome are visible to both. Hints and identities of played/discarded cards are public too. The partner's private memory is never observed; a player's own private memory can persist during an episode.

## Turn sequence and action space

On each transition the environment executes the active player's action and then switches the active player. Every transition calls both policy instances, giving both the previous turn's public result. A single seat is active at a time.
The action is a scalar integer in 0..20; the action mask has 21 booleans.
A true entry identifies an action permitted by the rules, without promising points. In particular, a card's legal play status gives no indication of whether playing it succeeds.

| IDs | Action | Exact legality and effect |
|---|---|---|
| 0..4 | Discard own slot s: action = s | Requires an occupied slot and fewer than 8 information tokens. Reveal and discard its card, regain one information token, then replace it by the procedure below. Lives are unchanged. |
| 5..9 | Play own slot s: action = 5+s | Play is allowed from any occupied slot. Reveal its card; success occurs exactly when the rank ID matches the color's firework height. Success raises that height by one; failure discards the card and loses one life. Replace the removed card. |
| 10..14 | Hint color c to the partner: action = 10+c | Requires an available information token and a partner card of color c. Use one token to mark all matching slots. |
| 15..19 | Hint rank r to the partner: action = 15+r | Requires an available information token and a partner card with rank ID r. Use one token to mark all matching slots. |
| 20 | Inactive no-op | The inactive seat can legally choose only this action; passing is unavailable to the active seat. |

Completing a successful rank-5 play restores one information token, up to eight; a failed play restores none. Repeating a hint is allowed even when its literal information is already fully known, but an empty hint is illegal. Every matching card is included, rather than a sender-chosen subset. Matched slots have the named attribute value and unmatched occupied slots lack it. No cards move or are drawn when hinting.

Playing or discarding removes the chosen slot and shifts each later slot one place left. Append the next drawn card at slot 4 if the deck contained cards; otherwise append an empty slot. This procedure also applies to unsuccessful plays. Public hint records shift with their cards, and new draws begin without previous hints. Slots identify current positions rather than stable card IDs.
A newly drawn card is hidden from its holder but visible to the partner. Neither discarded cards nor failed plays can be recovered.

## Observation arrays

HanabiObservation supplies the following two input fields:

- features: shape [658], dtype float32.
- legal_actions: shape [21], dtype bool.

Features form a length-658 flat vector. Every interval below is half-open.
A one-hot encoding places a single 1 in its indexed category; where specified, an all-zero vector indicates absence. A thermometer representing k has k leading ones followed by zeros. Sum that thermometer to recover its count.
Indices for players are relative to the observer: 0 means self and 1 partner.
Neither absolute seat nor turn count has a separate feature. Inactivity is indicated by the no-op mask entry, which is false at the active seat.

| Feature interval | Shape / meaning |
|---|---|
| [0:125] | Partner cards arranged as [5 slots, 5 colors, 5 ranks]. Entry 25*s+5*c+r is 1 for the card in partner slot s. An empty slot has 25 zeros. |
| [125:127] | Two missing-card flags in relative-player order. The flag is 1 if the hand holds fewer than five cards, not the number of missing cards. |
| [127:167] | The remaining draw-pile count uses a 40-entry thermometer, initialized with 40 entries equal to 1. |
| [167:192] | Fireworks, reshaped [5 colors, 5 ranks]. Each color marks its highest successful rank ID with a one-hot encoding. An all-zero vector indicates height 0; a one at rank ID r means height r+1. These rows are not thermometers. |
| [192:200] | Information tokens, an 8-entry thermometer. |
| [200:203] | Lives, a 3-entry thermometer. |
| [203:253] | Discard counts, reshaped [5 colors, 10 entries]. Within each color the five rank subblocks are [0:3], [3:5], [5:7], [7:9], [9:10]. Each subblock thermometrically counts discarded copies of that rank, failed plays included. Cards successfully placed in fireworks are excluded. |
| [253:308] | The latest public action is described next; its block is entirely zero at reset. |
| [308:658] | Native knowledge for each card, shaped [2 relative players, 5 slots, 35 entries]; see the details below. |

### Most recent action: offsets relative to 253

| Local interval | Meaning |
|---|---|
| [0:2] | The acting player is one-hot encoded in observer-relative order. |
| [2:6] | One-hot action type, ordered play, discard, color hint, rank hint. |
| [6:8] | Observer-relative one-hot hint recipient, set to zero for play/discard. |
| [8:13] | The named color is one-hot for color hints and zero otherwise. |
| [13:18] | The named rank ID is one-hot for rank hints, zero otherwise. |
| [18:23] | Five hint-match flags using recipient-slot indices. Unmatched slots are 0. Zero for play/discard. |
| [23:28] | The removed slot is one-hot for play/discard in the actor's pre-removal hand; otherwise zero. |
| [28:53] | The played/discarded card's revealed identity as a [5 colors, 5 ranks] one-hot vector; zero otherwise. |
| [53] | Boolean indicating whether this action added a card to a firework. |
| [54] | Boolean indicating that a play completed a color and actually increased information tokens. Discards give zero despite restoring a token. |

Hand, board, discard, and knowledge blocks reflect the post-action state, whereas the removed-slot field identifies a slot in the pre-action state. Each transition overwrites the last-action block instead of accumulating history. The acting player alternates, and both seats can read each result.

### Native knowledge block: 35 entries per current slot

Entries [0:25] are the environment's native V0 weights, flattened color-major
as 5*c+r. Entries [25:30] are positive color-hint flags and [30:35] positive
rank-hint flags. Matching cards accumulate positive flags, which shift when cards are removed. A missing positive flag alone does not express a negative hint.

For an occupied slot, V0 takes the native literal-possibility mask times the remaining public counts by card type, normalizing the slot's 25 entries to sum to one. Remaining public counts subtract discarded copies and the copy already in the corresponding color/rank firework position from the original multiplicity. They do not subtract visible partner-hand cards or impose joint card-count consistency across slots. V0 is therefore an observation feature, neither an exact hidden-hand joint distribution nor a model of another policy's intent. The native possibility mask begins unrestricted, narrows with matching and nonmatching hints, shifts on card removal, and resets on a new draw. All V0 entries and positive hint flags are zero in empty slots. Both players' knowledge records are in relative order and derive from public hints rather than private policy memory.

## Rewards and episode termination

A successful play yields shared reward +1. Hints, discards, and failed plays with lives still remaining yield 0. Losing the final life subtracts the current firework score, leaving episode return 0 under bomb-zero scoring.
The configured correction ends the episode immediately on final-life loss:
True. Completing all five fireworks ends the episode
with total return 25.

Taking the final draw starts the native final-round counter. Unless another ending occurs sooner, two active-player turns remain: first the other player's turn, then the final drawer's turn. No more cards are drawn, and hints use up turns as usual. When the final round finishes, return equals the summed firework heights unless all lives were lost. An empty deck by itself does not immediately terminate play.
Each evaluation episode is limited to 128 transitions for safety. Incomplete episodes are rejected rather than assigned ordinary success or failure scores.

A terminal transition triggers automatic reset by the adapter. On the next policy call reset is true, observations come from the next shuffled deal, and private memory is reset to the value returned by its initializer. Hidden environment state and reward are never policy inputs. Evaluation tools separately report returns and public observations. An illegal active action invalidates evaluation; the adapter forces the inactive seat to no-op.


See specification.json for numeric dimensions, resolved flags, and array schemas.
