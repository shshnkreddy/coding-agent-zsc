# Two-player Hanabi: environment specification

## Goal, objects and initial state

The players cooperate on five color-specific public fireworks. Every firework follows ranks 1, 2, 3, 4, 5 in order, never duplicating or skipping a rank. The team score is the episode return for both players, up to 25. Only game mechanics and data formats are specified here; no additional meaning is attached to a player's selection of an action.

Across five colors the deck totals 50 cards, with each color contributing 3 rank 1 copies, 2 each of ranks 2, 3 and 4, and 1 rank 5 copy. The color ID range is 0..4. For array ranks, IDs 0..4 map to human ranks 1..5. No special color is included.
A new episode shuffles the fixed multiset uniformly, giving five cards to player 0 and five to player 1 and retaining 40 in the draw pile. The first action belongs to player 0. The initial state has empty fireworks and discards, eight information tokens, three lives, and no hints on either hand.

The public information available to both seats includes firework heights, discard counts, information tokens, lives, deck size, hints, played/discarded card identities, and the latest action and its public outcome. Each player additionally sees the identities and ordering of the other player's cards. Own card identities and draw-pile order remain hidden. A player's private memory may persist within an episode, but a partner's private memory is never an observation.

## Turn sequence and action space

Each environment transition handles the action of the sole active player and then changes who is active. Calling both policy instances at every transition lets each receive the public result of the prior turn.
The action is a scalar integer in 0..20; the action mask has 21 booleans.
Being permitted under the rules, indicated by a true mask entry, does not imply scoring points. Legal card plays are not thereby revealed to be successful.

| IDs | Action | Exact legality and effect |
|---|---|---|
| 0..4 | Discard own slot s: action = s | Allowed for an occupied slot when fewer than 8 information tokens remain. The card is revealed and discarded, one token is restored, and replacement follows the description below. Life is not lost. |
| 5..9 | Play own slot s: action = 5+s | Any occupied slot can be selected to play. Reveal the card; exactly matching its rank ID to that color's firework height succeeds and increases the height by one. Otherwise discard it and lose one life. The removed card is replaced. |
| 10..14 | Hint color c to the partner: action = 10+c | There must be an information token left and a card with color c held by the partner. Identify every matching slot at a cost of one token. |
| 15..19 | Hint rank r to the partner: action = 15+r | There must be an information token left and a card with rank ID r held by the partner. Identify every matching slot at a cost of one token. |
| 20 | Inactive no-op | This alone is legal for the inactive seat. There is no pass for the active seat. |

An empty hint is not legal, but repeating a hint whose literal information is entirely known is legal. The sender does not choose a subset: all matching cards are identified. The named attribute value is present in matched slots and absent in unmatched occupied slots. Hinting moves no cards and draws none. A successful rank-5 play also restores one information token, capped at eight; a failed play does not.

The hand is compacted after every play or discard: remove that slot and shift all later slots left one position. Append a draw at slot 4 when the deck was nonempty, otherwise append an empty slot. This is independent of play success. Public hint records shift with the cards they describe, and a new draw starts with no previous hints. Slot numbers denote current positions, not stable card IDs.
New own-hand draws stay concealed from their owner and are exposed to the partner. Discards and cards from failed plays cannot be retrieved.

## Observation arrays

The input has type HanabiObservation and the following two fields:

- features: shape [658], dtype float32.
- legal_actions: shape [21], dtype bool.

The feature representation is a flat vector of length 658, with half-open intervals below.
A one-hot vector marks its category with a single 1 and uses all zeros for absence where specified. A value-k thermometer begins with k ones and then zeros. Obtain counts by summing the thermometer.
Indices for players are relative to the observer: 0 means self and 1 partner.
Absolute seat and turn counter are not supplied as separate features. Use the no-op mask entry as the inactivity indicator; it is false for the active seat.

| Feature interval | Shape / meaning |
|---|---|
| [0:125] | Partner-hand encoding with shape [5 slots, 5 colors, 5 ranks]. Entry 25*s+5*c+r is 1 for the card in partner slot s. An empty slot has 25 zeros. |
| [125:127] | Two missing-card flags in relative-player order. A hand with fewer than five cards sets the flag to 1; the flag is not a count of missing cards. |
| [127:167] | Remaining cards in the draw pile use a 40-entry thermometer whose 40 entries are initially 1. |
| [167:192] | Fireworks, reshaped [5 colors, 5 ranks]. The highest successfully played rank ID is one-hot encoded for each color. An all-zero vector indicates height 0; a one at rank ID r means height r+1. These rows are not thermometers. |
| [192:200] | Information tokens, an 8-entry thermometer. |
| [200:203] | Lives, a 3-entry thermometer. |
| [203:253] | Discard counts, reshaped [5 colors, 10 entries]. Within each color the five rank subblocks are [0:3], [3:5], [5:7], [7:9], [9:10]. The rank's discarded copies, including failed plays, form each subblock's thermometer count. Successful firework cards are not discarded into this pile. |
| [253:308] | Details of the latest public action follow. Reset makes this block all zero. |
| [308:658] | Native card knowledge with shape [2 relative players, 5 slots, 35 entries]; the following section gives its encoding. |

### Most recent action: offsets relative to 253

| Local interval | Meaning |
|---|---|
| [0:2] | The actor's identity is one-hot, in observer-relative order. |
| [2:6] | Action-type one-hot entries follow play, discard, color hint, rank hint. |
| [6:8] | The hint recipient is one-hot in relative order, or zero for play/discard. |
| [8:13] | For a color hint, encode its named color one-hot; otherwise use zero. |
| [13:18] | For a rank hint this is one-hot at the named rank ID; otherwise it is zero. |
| [18:23] | The five hint-match indicators are indexed by the recipient's slots. Unmatched slots are 0. Zero for play/discard. |
| [23:28] | On play/discard, encode the removed slot one-hot relative to the actor's pre-removal hand. Otherwise use zero. |
| [28:53] | A [5 colors, 5 ranks] one-hot encoding identifies the revealed played/discarded card. Other actions give zero. |
| [53] | Boolean marking whether a firework gained a card from the action. |
| [54] | Boolean marking an actual information-token increase caused by a color-completing play. It is zero on discards even when a token is restored. |

Both seats can read every result even though the active actor alternates. The last-action block is overwritten on each transition rather than retaining a history. Hand, board, discard, and knowledge describe the state after the action; the removed-slot field describes its pre-action slot.

### Native knowledge block: 35 entries per current slot

Entries [0:25] are the environment's native V0 weights, flattened color-major
as 5*c+r. Entries [25:30] are positive color-hint flags and [30:35] positive
rank-hint flags. Matched cards accumulate these positive flags, and removal shifts them. Absence of a positive flag does not independently encode a negative hint.

To form V0 for an occupied slot, multiply native literal possibilities by remaining public counts for each card type; normalize these 25 values so they sum to one. Compute remaining public count by removing discarded copies and the copy in that color/rank firework position from original multiplicity, without subtracting visible partner cards or enforcing joint card-count consistency across slots. Accordingly, V0 is an observation feature and does not give an exact joint hidden-hand distribution or predict another policy's intentions. Native possibilities start unrestricted. Matching and nonmatching hint evidence reduces the mask, card removal shifts it, and a new draw resets it. For empty slots, every V0 entry and positive hint flag is zero. Relative-player order is used for both native knowledge records, whose source is public hints, not private policy memory.

## Rewards and episode termination

Success on a play gives shared +1 reward; hints, discards, and failures leaving lives give 0. The final life loss subtracts the current firework score and sets total episode return to 0 (bomb-zero scoring).
The configured correction ends the episode immediately on final-life loss:
True. Completing all five fireworks ends the episode
with total return 25.

Two active-player turns follow the final draw, unless another termination condition intervenes. The native final-round counter starts with that draw; the other player acts next, followed by the player who drew the final card. There are no additional draws, and hints count as turns. Return at the end of the final round is the sum of firework heights unless all lives were lost. Merely reaching the deck's end does not immediately end the episode.
The evaluation safety limit is 128 transitions for an episode, with incomplete episodes rejected instead of scored as usual successes or failures.

The adapter automatically resets following a terminal transition, so reset is true at the next policy call. That call observes the next shuffled deal and resets private memory to the value returned by its initializer. Policy inputs exclude hidden environment state and reward. Returns and public observations are separately reported by evaluation tools. Illegal active actions invalidate evaluation; inactive seats are forced to no-op.


Consult specification.json for resolved flags, numeric dimensions, and array schemas.
