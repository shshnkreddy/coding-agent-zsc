# Two-player Hanabi: environment specification

## Goal, objects and initial state

The shared episode return equals the team score and cannot exceed 25. The task for the two cooperating players is to construct five public fireworks, one per color. Each must progress through ranks 1, 2, 3, 4, 5 without repetition or omission. This description covers mechanics and data formats, not additional meanings for players' action choices.

There are 50 cards across five colors: per color, 3 rank 1 cards, 2 cards of each rank 2, 3 and 4, and 1 rank 5 card. Color IDs range over 0..4, and array rank IDs over 0..4 for human ranks 1..5. The game has no special color.
Uniformly shuffle the fixed multiset at episode start. Deal five cards to player 0 and five to player 1, leaving 40 for drawing. Player 0 goes first. The fireworks and discard pile are initially empty, information tokens number eight, lives number three, and neither hand starts with hints.

Neither a player's own card identities nor the draw-pile order is visible to that player. The partner's card identities and ordering are visible. Both players see firework heights, discard counts, information tokens, lives, deck size, and the latest action's public outcome together with the action. Hints and played/discarded card identities are public information. Private memory can persist within the episode for its owner, but the partner's private memory is never part of an observation.

## Turn sequence and action space

Only one seat acts as the active player. A transition executes that player's action before transferring activity to the other player. Both policy instances are called at every transition and therefore both obtain the public result of the preceding turn.
The action is a scalar integer in 0..20; the action mask has 21 booleans.
The mask indicates rule legality rather than scoring potential. A true entry allows the action; for card plays, legality does not disclose whether the play would succeed.

| IDs | Action | Exact legality and effect |
|---|---|---|
| 0..4 | Discard own slot s: action = s | The chosen slot must contain a card and information tokens must be below 8. Reveal and discard the card, restore a token, and replace the card as specified below without losing a life. |
| 5..9 | Play own slot s: action = 5+s | An occupied slot always permits play. Reveal the card and compare its rank ID with the current firework height of its color: equality means success and a one-step increase. Otherwise discard it and lose a life. Replace the removed card. |
| 10..14 | Hint color c to the partner: action = 10+c | Legal only with an information token available and at least one color c card in the partner's hand. Consume one token and indicate all matching slots. |
| 15..19 | Hint rank r to the partner: action = 15+r | Legal only with an information token available and at least one card of rank ID r in the partner's hand. Consume one token and indicate all matching slots. |
| 20 | Inactive no-op | Only this action is legal for the inactive seat, while the active seat has no pass. |

Hinting neither repositions cards nor draws cards. A hint covers all cards that match, not a subset chosen by the sender: the named attribute value holds in matching slots and not in unmatched occupied slots. Empty hints are illegal; repeats remain legal even if all the literal information is known already. A successful rank-5 play additionally restores one information token with a cap of eight, whereas an unsuccessful play restores none.

Slots represent present hand positions, not permanent card IDs. After a play or discard, remove its slot and move all subsequent slots one place to the left. If the deck was nonempty, put the next draw at slot 4; if not, append an empty slot. Success of the play does not affect this procedure. Hint records travel with shifted cards, while newly drawn cards have no prior hints.
After drawing, the card's owner cannot see its identity, but the partner can. Cards discarded or lost on failed plays are unrecoverable.

## Observation arrays

The two fields of the HanabiObservation input are:

- features: shape [658], dtype float32.
- legal_actions: shape [21], dtype bool.

The features vector is flat and has length 658; the intervals below are all half-open.
One-hot means a single 1 at the indexed category, with all zeros indicating absence where noted. Value k in a thermometer is k leading ones and then zeros; its sum gives the count.
Indices for players are relative to the observer: 0 means self and 1 partner.
Separate features for absolute seat and turn count are absent. The active seat has a false no-op mask entry; that entry identifies inactivity.

| Feature interval | Shape / meaning |
|---|---|
| [0:125] | The partner hand uses shape [5 slots, 5 colors, 5 ranks]. Entry 25*s+5*c+r is 1 for the card in partner slot s. An empty slot has 25 zeros. |
| [125:127] | Two missing-card flags in relative-player order. Value 1 means fewer than five cards remain in that hand; it does not count missing cards. |
| [127:167] | Draw-pile size remaining is encoded by a 40-entry thermometer; all 40 entries start at 1. |
| [167:192] | Fireworks, reshaped [5 colors, 5 ranks]. For every color, a one-hot entry identifies the highest successfully played rank ID. An all-zero vector indicates height 0; a one at rank ID r means height r+1. These rows are not thermometers. |
| [192:200] | Information tokens, an 8-entry thermometer. |
| [200:203] | Lives, a 3-entry thermometer. |
| [203:253] | Discard counts, reshaped [5 colors, 10 entries]. Within each color the five rank subblocks are [0:3], [3:5], [5:7], [7:9], [9:10]. A subblock uses a thermometer for that rank's discarded copies, including failed plays but not successfully played firework cards. |
| [253:308] | Latest public-action information, expanded next, with all entries zero at reset. |
| [308:658] | Per-card native knowledge arranged as [2 relative players, 5 slots, 35 entries], explained below. |

### Most recent action: offsets relative to 253

| Local interval | Meaning |
|---|---|
| [0:2] | Observer-relative one-hot encoding of the acting player. |
| [2:6] | The action type uses one-hot categories play, discard, color hint, rank hint, in that order. |
| [6:8] | Hint-target identity encoded one-hot relative to the observer; zero for play/discard. |
| [8:13] | One-hot named-color encoding for a color hint, with zero for other actions. |
| [13:18] | Rank hints encode the named rank ID one-hot; other actions leave zero. |
| [18:23] | Five hint-match flags, one for each recipient slot. Unmatched slots are 0. Zero for play/discard. |
| [23:28] | For play/discard, a one-hot identifies the slot in the acting player's hand before removal; other actions give zero. |
| [28:53] | One-hot revealed identity of the played/discarded card in [5 colors, 5 ranks], or zero otherwise. |
| [53] | Boolean reporting that the action placed a card on a firework. |
| [54] | Boolean for an actual information-token increase from completing a color on a play. A discard yields zero even though it restores a token. |

The removed-slot field uses the state before the action; hand, board, discard, and knowledge blocks use the state after it. Last-action information is replaced each transition, not appended into a history. Activity alternates between actors while both seats can observe every result.

### Native knowledge block: 35 entries per current slot

Entries [0:25] are the environment's native V0 weights, flattened color-major
as 5*c+r. Entries [25:30] are positive color-hint flags and [30:35] positive
rank-hint flags. Positive flags build up on matched cards and move with removals; not having a positive flag is not itself a negative-hint encoding.

V0 for an occupied slot multiplies each card type's remaining public count by the native literal-possibility mask and normalizes the 25 slot values to total one. Start with original multiplicity, subtract discarded copies, and subtract the copy occupying that color/rank firework position to obtain remaining public count. Visible partner-hand cards are not subtracted, and no cross-slot joint card-count consistency is enforced. Thus V0 represents an observation feature rather than an exact joint distribution of hidden hands or a prediction of another policy's intentions. Initially unrestricted native possibilities are reduced by both matching and nonmatching hint evidence; the mask shifts with removed cards and resets for new draws. In an empty slot, the V0 entries and positive hint flags all have value zero. The native knowledge records of both players are ordered relatively and reflect public hints, not private policy memory.

## Rewards and episode termination

Each successful play earns the team +1. Hints, discards, and unsuccessful plays that do not exhaust lives give 0. When the final life is lost, subtract the accumulated firework score so the episode return becomes 0 (bomb-zero scoring).
The configured correction ends the episode immediately on final-life loss:
True. Completing all five fireworks ends the episode
with total return 25.

The deck running out is not an immediate episode ending. Drawing its last card starts the native final-round counter, allowing two more active-player turns unless another termination condition comes first: the other player acts, then the player who drew last. These turns have no draws, and giving a hint consumes a turn. The final-round return is the sum of firework heights except when all lives have been lost.
Evaluation allows at most 128 transitions per episode; reaching the cap without completion causes rejection, not normal success or failure scoring.

The adapter resets automatically after terminal transitions. The next policy call has reset true, a fresh shuffled-deal observation, and private memory reset to the value returned by its initializer. No policy input contains hidden environment state or reward. Tools report returns separately from public observations. Illegal active actions invalidate evaluation, while the adapter imposes no-op on the inactive seat.


Numeric sizes, resolved flags, and array schemas are recorded in specification.json.
