# Two-player Hanabi: environment specification

## Goal, objects and initial state

The team score, shared by both players as episode return, has a maximum of 25. To score, the two players cooperate in building five public fireworks, one per color, containing ranks 1, 2, 3, 4, 5 in order without duplication or skipping. This specification describes the mechanics and data formats, leaving action choices without any additional assigned meaning.

The game uses 50 cards in five colors; each color has 3 cards of rank 1, 2 of each of ranks 2, 3 and 4, and 1 of rank 5. The colors are indexed 0..4. Array rank IDs also run from 0..4, equivalent to human ranks 1..5, with no special color present.
Start each episode by uniformly shuffling this multiset, then dealing five cards to player 0 and five to player 1; the other 40 form the draw pile. Player 0 begins. Fireworks and the discard pile are empty at reset, with eight information tokens, three lives, and unhinted initial hands.

The other hand's card identities and ordering are visible to each player, whereas their own card identities and the draw pile's order are hidden. Both observe public firework heights, discard counts, information tokens, lives, deck size, and the most recent action with its public outcome. Hints and played/discarded card identities are public. Private memory belonging to a player can persist within the episode and is never observable by the partner.

## Turn sequence and action space

Every transition calls both policies, so each receives the public result of the previous turn. Just one seat is active: the environment executes its action and then changes the active player.
The action is a scalar integer in 0..20; the action mask has 21 booleans.
The meaning of a true mask entry is permission under the game rules, not a guarantee of points. In particular, legal status for playing a card does not reveal its success.

| IDs | Action | Exact legality and effect |
|---|---|---|
| 0..4 | Discard own slot s: action = s | With an occupied slot and fewer than 8 information tokens, reveal and discard the card, replenish one information token, and replace it as described below. This costs no life. |
| 5..9 | Play own slot s: action = 5+s | Playing any occupied slot reveals its card. A rank ID equal to the current firework height for that color succeeds and raises the height by one. Otherwise discard the card and lose one life; replace the removed card in either case. |
| 10..14 | Hint color c to the partner: action = 10+c | At least one partner card must have color c, and at least one information token must remain. One token is spent to identify all matching slots. |
| 15..19 | Hint rank r to the partner: action = 15+r | At least one partner card must have rank ID r, and at least one information token must remain. One token is spent to identify all matching slots. |
| 20 | Inactive no-op | The inactive seat is restricted to this action, and the active seat is not allowed to pass. |

A successful rank-5 play returns one information token, subject to the eight-token cap; unsuccessful plays return no token. Hints may repeat known literal information but must not be empty. All matching cards are addressed, never only a subset chosen by the sender. Matched slots possess the named attribute value and unmatched occupied slots do not. Neither card positions nor draws change as a result of hinting.

After playing or discarding, the removed slot is closed by shifting every later slot one position left. The next draw is appended at slot 4 if the deck was nonempty; otherwise an empty slot is appended. The same rule holds if a play fails. Public hint records follow shifting cards and new draws carry no prior hints. Slots track current positions instead of stable card IDs.
The partner sees each newly drawn own card, but its owner does not. Discarded cards and unsuccessful plays are not recoverable.

## Observation arrays

A HanabiObservation input consists of these two fields:

- features: shape [658], dtype float32.
- legal_actions: shape [21], dtype bool.

Features occupy a flat vector with length 658. The following intervals use half-open bounds.
The one-hot category is marked by one 1; an all-zero vector indicates absence wherever specified. A thermometer of value k has k leading ones followed by zeros. Summing the indicated thermometer yields the count.
Indices for players are relative to the observer: 0 means self and 1 partner.
No dedicated absolute-seat or turn-count feature is provided. The no-op mask entry denotes inactivity, being false for the active seat.

| Feature interval | Shape / meaning |
|---|---|
| [0:125] | The partner's hand is represented as [5 slots, 5 colors, 5 ranks]. Entry 25*s+5*c+r is 1 for the card in partner slot s. An empty slot has 25 zeros. |
| [125:127] | Two missing-card flags in relative-player order. The flag equals 1 for a hand containing fewer than five cards and does not report how many cards are missing. |
| [127:167] | Draw-pile remainder, represented by a 40-entry thermometer. Its 40 entries begin at 1. |
| [167:192] | Fireworks, reshaped [5 colors, 5 ranks]. For each color the one-hot position is its highest successfully played rank ID. An all-zero vector indicates height 0; a one at rank ID r means height r+1. These rows are not thermometers. |
| [192:200] | Information tokens, an 8-entry thermometer. |
| [200:203] | Lives, a 3-entry thermometer. |
| [203:253] | Discard counts, reshaped [5 colors, 10 entries]. Within each color the five rank subblocks are [0:3], [3:5], [5:7], [7:9], [9:10]. Each thermometer subblock counts copies of its rank in discards, including failed plays and excluding cards successfully added to fireworks. |
| [253:308] | The last public action is represented here and detailed next; reset clears it to zero. |
| [308:658] | Per-card knowledge from the native environment, reshaped [2 relative players, 5 slots, 35 entries], with details below. |

### Most recent action: offsets relative to 253

| Local interval | Meaning |
|---|---|
| [0:2] | Acting-player identity in an observer-relative one-hot vector. |
| [2:6] | The one-hot type categories are ordered as play, discard, color hint, rank hint. |
| [6:8] | Encode the hint recipient as observer-relative one-hot, using zero for play/discard. |
| [8:13] | This names the hinted color one-hot on color hints and is zero otherwise. |
| [13:18] | The named rank ID appears one-hot when hinting rank and zero otherwise. |
| [18:23] | Hint matches are recorded by five recipient-slot-indexed flags. Unmatched slots are 0. Zero for play/discard. |
| [23:28] | The actor's slot before removal is one-hot encoded for play/discard; in other cases the field is zero. |
| [28:53] | Revealed identity for a played/discarded card, represented one-hot over [5 colors, 5 ranks]; otherwise all zero. |
| [53] | Boolean that is set when the action adds a card to a firework. |
| [54] | Boolean denoting that a play's color completion actually increased information tokens. A discard restores a token but leaves this flag zero. |

The last-action block records only the latest transition and is replaced rather than accumulated. Its removed-slot field refers to the state before the action, unlike the hand, board, discard, and knowledge blocks, which describe the state afterward. Both seats can read each result as the active actor alternates.

### Native knowledge block: 35 entries per current slot

Entries [0:25] are the environment's native V0 weights, flattened color-major
as 5*c+r. Entries [25:30] are positive color-hint flags and [30:35] positive
rank-hint flags. On matching cards, positive flags accumulate and shift when removal occurs; the lack of a positive flag alone conveys no encoded negative hint.

For each occupied slot, V0 uses the product of the native literal-possibility mask and the remaining public card-type counts, with its 25 values normalized to sum to one. Original multiplicity is reduced by discards and by the copy already in that color/rank firework position. This remaining public count neither subtracts visible partner-hand cards nor enforces joint card-count consistency across slots. V0 consequently supplies an observation feature, not an exact joint distribution over hidden hands and not an inference about another policy's intent. The native possibility mask initially permits everything, narrows with matching and nonmatching hints, moves with removal, and resets on drawing a new card. For an empty slot, all V0 entries and positive hint flags are zero. Knowledge records for both players are observer-relative and based on public hints, not private policy memory.

## Rewards and episode termination

The team receives +1 for every successful play. Hints, discards, and unsuccessful plays while lives remain receive 0. Final-life loss subtracts the current firework score, making the episode's total return 0 under bomb-zero scoring.
The configured correction ends the episode immediately on final-life loss:
True. Completing all five fireworks ends the episode
with total return 25.

An episode does not stop immediately just because the deck is exhausted. The last draw begins the native final-round counter: the other player and then the last drawer get one further active turn each, unless play terminates sooner for another reason. No draws remain, but hints still take turns. Once the final round ends, return is the sum of firework heights unless all lives have been lost.
Evaluation episodes have a 128-transition safety cap. Failure to complete makes the episode invalid rather than an ordinarily scored success or failure.

Terminal transitions are followed by an automatic adapter reset. At the next policy call, reset is true and observations refer to the new shuffled deal, with private memory reset through its initializer. Reward and hidden environment state never appear in policy inputs. Evaluation tools report public observations and returns separately. Illegal active actions invalidate evaluation, and the adapter requires no-op from the inactive seat.


Find numeric dimensions, resolved flags, and array schemas in specification.json.
