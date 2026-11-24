# Two-player Hanabi: environment specification

## Goal, objects and initial state

Five public fireworks, one for each color, are built cooperatively by the two players. A firework accepts ranks 1, 2, 3, 4, 5 in precisely that sequence; duplicates and skipped ranks are not permitted. Episode return is the shared team score, whose maximum is 25. The specification supplies mechanics and data formats but attaches no extra interpretation to an action choice.

The 50-card deck spans five colors, each containing 3 copies of rank 1, 2 of each rank 2, 3 and 4, and 1 of rank 5. IDs 0..4 label colors. Array ranks likewise use IDs 0..4 for ranks 1..5 as named by humans. There are no special colors.
At a new episode, uniformly shuffle this fixed card multiset and deal five cards each to player 0 and player 1; 40 remain in the draw pile. Player 0 acts first. Initially fireworks and discards are empty, eight information tokens and three lives remain, and the hands have no hints.

Players can see the identity and position of each card in the other hand, not the identities in their own hand or the ordering of the draw pile. Shared visible information includes firework heights, discard counts, information tokens, lives, deck size, and the most recent action and public result. Hints and the identities of played/discarded cards are also public. Each player may keep private memory across the episode; the other player's private memory is never observed.

## Turn sequence and action space

The active player changes after their action is executed in an environment transition. Although only one seat is active, both policies are called on every transition and both receive the previous turn's public outcome.
The action is a scalar integer in 0..20; the action mask has 21 booleans.
An allowed action has a true mask entry, but this is not evidence that it earns points. Whether a card play succeeds cannot be inferred from the legality of that play.

| IDs | Action | Exact legality and effect |
|---|---|---|
| 0..4 | Discard own slot s: action = s | An occupied slot is required, with fewer than 8 information tokens available. Reveal and discard the card, restore one token, and follow the replacement rule below. No life is consumed. |
| 5..9 | Play own slot s: action = 5+s | Reveal a card from any occupied slot. It succeeds iff its rank ID equals its color's current firework height, raising the height by one. Otherwise it is discarded and one life is lost. Replacement follows the removal. |
| 10..14 | Hint color c to the partner: action = 10+c | An information token must be available, and the partner must hold at least one card of color c. Pay one token, identifying each matching slot. |
| 15..19 | Hint rank r to the partner: action = 15+r | An information token must be available, and the partner must hold at least one card with rank ID r. Pay one token, identifying each matching slot. |
| 20 | Inactive no-op | Inactive seats have this as their sole legal action; active seats cannot pass. |

One information token is restored by a successful rank-5 play, without exceeding eight. Unsuccessful plays restore no token. Hints are repeatable even when they add no new literal information, but they cannot be empty. The sender must include every match. Slots that match have the named attribute value; unmatched occupied slots do not. Hints leave card positions unchanged and do not cause draws.

A played or discarded card leaves a slot that is removed; all later slots shift left by one. Slot 4 receives the next draw when the deck was nonempty, or an empty slot is appended otherwise. Successful and failed plays use the same procedure. Public hints remain attached to cards as they shift. New cards have no earlier hints, and slots are current positions rather than stable IDs.
The owner does not see a new draw's identity, while the partner does. There is no recovery of discarded cards or failed plays.

## Observation arrays

Input arrives as HanabiObservation containing two fields:

- features: shape [658], dtype float32.
- legal_actions: shape [21], dtype bool.

A flat vector of length 658 holds the features. All listed intervals are half-open.
The indexed category of a one-hot vector contains its sole 1; all-zero vectors indicate absence where stated. Thermometer value k is encoded by k leading ones followed by zeros. Counts equal sums of the specified thermometers.
Indices for players are relative to the observer: 0 means self and 1 partner.
The observation includes no separate absolute-seat or turn-counter feature. Its no-op mask entry signals inactivity and is false for the active seat.

| Feature interval | Shape / meaning |
|---|---|
| [0:125] | Reshape the partner hand into [5 slots, 5 colors, 5 ranks]. Entry 25*s+5*c+r is 1 for the card in partner slot s. An empty slot has 25 zeros. |
| [125:127] | Two missing-card flags in relative-player order. This flag takes value 1 for a hand shorter than five cards, rather than counting absent cards. |
| [127:167] | A 40-entry thermometer reports remaining draw-pile size, with 40 initial entries of 1. |
| [167:192] | Fireworks, reshaped [5 colors, 5 ranks]. A color is encoded one-hot at the rank ID of its highest successful play. An all-zero vector indicates height 0; a one at rank ID r means height r+1. These rows are not thermometers. |
| [192:200] | Information tokens, an 8-entry thermometer. |
| [200:203] | Lives, a 3-entry thermometer. |
| [203:253] | Discard counts, reshaped [5 colors, 10 entries]. Within each color the five rank subblocks are [0:3], [3:5], [5:7], [7:9], [9:10]. Subblocks encode discarded copies of their rank as thermometers, counting failed plays. Successful firework cards do not belong to this pile. |
| [253:308] | The most recent public action, detailed below, is encoded as all zeros on reset. |
| [308:658] | The native card-knowledge features reshape to [2 relative players, 5 slots, 35 entries] and are described below. |

### Most recent action: offsets relative to 253

| Local interval | Meaning |
|---|---|
| [0:2] | One-hot acting-player indicator, ordered relative to the observer. |
| [2:6] | Encode action type one-hot with category order play, discard, color hint, rank hint. |
| [6:8] | One-hot recipient of a hint in observer-relative order; play/discard leaves zero. |
| [8:13] | Color hints mark the named color one-hot; all other actions leave zero. |
| [13:18] | One-hot encoding of a rank hint's named rank ID, or zero otherwise. |
| [18:23] | Five flags indicate hint matches in recipient-slot order. Unmatched slots are 0. Zero for play/discard. |
| [23:28] | One-hot removed-slot indicator for play/discard, using the actor's hand before removal, and zero otherwise. |
| [28:53] | The revealed played/discarded identity uses [5 colors, 5 ranks] one-hot encoding; in other cases it is zero. |
| [53] | Boolean flag for a card being added to a firework by this action. |
| [54] | Boolean set by a play that completes a color and really increases information tokens; discards remain zero despite their token restoration. |

Each transition replaces the last-action block, so it stores no cumulative history. The hand, board, discard, and knowledge blocks show the resulting state, while the removed-slot field refers to the preceding state. Actors alternate and both seats may read each outcome.

### Native knowledge block: 35 entries per current slot

Entries [0:25] are the environment's native V0 weights, flattened color-major
as 5*c+r. Entries [25:30] are positive color-hint flags and [30:35] positive
rank-hint flags. Positive flags persist cumulatively on matching cards and shift on removal. Their absence alone must not be interpreted as encoding a negative hint.

An occupied slot's V0 is formed by multiplying the native literal-possibility mask by remaining public card-type counts, then scaling its 25 values to sum to one. The remaining public count is original multiplicity less discarded copies and less the copy already at that color/rank firework position. Cards visible in the partner's hand are not removed from it, nor is joint card-count consistency across slots enforced. Treat V0 as an observation feature, not as an exact joint hidden-hand distribution or a forecast of another policy's intent. The native mask is unrestricted initially, then constrained by matching and nonmatching hints, shifted on removal, and reset for each new draw. All V0 entries and positive hint flags are zero in empty slots. Both players' native knowledge appears in relative order, based solely on public hints rather than private policy memory.

## Rewards and episode termination

Shared reward is +1 on a successful play and 0 for hints, discards, or unsuccessful plays with lives left. Loss of the last life removes the current firework score, producing total episode return 0 in bomb-zero scoring.
The configured correction ends the episode immediately on final-life loss:
True. Completing all five fireworks ends the episode
with total return 25.

The native final-round counter is triggered by taking the last draw. In the absence of an earlier termination, the other player receives a turn and then the last-drawing player receives a turn, for two further active-player turns in total. Further draws are unavailable; hints still consume turns. At final-round completion the sum of firework heights is the return unless all lives were lost. Deck exhaustion alone does not end the episode immediately.
A safety limit of 128 transitions applies to each evaluation episode. Unfinished episodes do not receive ordinary success or failure scores; they are rejected.

After termination the adapter resets on its own. At the ensuing policy call reset is true, a newly shuffled deal supplies the observation, and the initializer resets private memory. Hidden state and reward do not enter policy inputs; evaluation tools report returns and public observations separately. Active illegal actions invalidate evaluation and the adapter forces inactive seats to no-op.


The specification.json file supplies numeric dimensions, resolved flags, and array schemas.
