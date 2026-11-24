# Two-player Hanabi: environment specification

## Goal, objects and initial state

The two players cooperate to build five public fireworks, one per color. Each
firework must contain ranks 1, 2, 3, 4, 5 in that order, without duplicates or
skipping ranks. The episode return is the team's score, shared by both players.
The maximum is 25. This specification describes game mechanics and data formats;
no extra meanings are assigned to a player's choice of action.

The game contains 50 cards: five colors, with 3 copies of rank 1, 2 of each of
ranks 2, 3 and 4, and 1 of rank 5 per color. Color IDs are 0..4. Array rank IDs
are also 0..4, corresponding to human ranks 1..5. There is no special color.
A fresh episode uniformly shuffles this fixed multiset, deals five cards to
player 0 and five to player 1, and leaves 40 cards in the draw pile. Player 0
acts first. All fireworks and the discard pile start empty, with eight
information tokens and three lives. There are no hints on the initial hands.

Each player sees the identities and ordering of the other player's cards but
not the identities of their own cards or the order of the draw pile. Both see
public firework heights, discard counts, information tokens, lives, deck size,
and the most recent action and its public outcome. Hints and the identities
of played/discarded cards are public. The partner's private memory is never an
observation. The player's own private memory may persist within an episode.

## Turn sequence and action space

One environment transition executes one active player's action, then changes
the active player. Both policy instances are called on every transition, so
both receive the public result of the previous turn. Only one seat is active.
The action is a scalar integer in 0..20; the action mask has 21 booleans.
A true mask entry means the action is allowed by the game rules, not that it
will score points. In particular, legality of playing a card does not reveal
whether that card will succeed.

| IDs | Action | Exact legality and effect |
|---|---|---|
| 0..4 | Discard own slot s: action = s | The slot must be occupied and fewer than 8 information tokens remain. Reveal and discard that card, restore one information token, then replace the card as described below. No life is lost. |
| 5..9 | Play own slot s: action = 5+s | Any occupied slot may be played. Reveal the card. It succeeds exactly when its rank ID equals that color's current firework height. Success increases the height by one. Otherwise discard it and lose one life. Replace the removed card. |
| 10..14 | Hint color c to the partner: action = 10+c | At least one information token remains and at least one partner card has color c. Spend one token and identify every matching slot. |
| 15..19 | Hint rank r to the partner: action = 15+r | At least one information token remains and at least one partner card has rank ID r. Spend one token and identify every matching slot. |
| 20 | Inactive no-op | This is the only legal action for the inactive seat; the active seat cannot pass. |

A successful rank-5 play also restores one information token, capped at eight.
An unsuccessful play does not restore a token. Hints may be repeated, even if
all their literal information was already known. An empty hint is illegal.
A hint refers to all matching cards; the sender cannot select only some of
them. For the named attribute, matched slots have that value and unmatched
occupied slots do not. Hinting changes no card positions and draws no cards.

Whenever a card is played or discarded, remove that slot, shift all later
slots one position left, and append the next draw at slot 4 if the deck was
nonempty. Otherwise append an empty slot. This happens regardless of whether
a play succeeds. Public hint records move with the shifted cards. A new draw
has no previous hints. Slots are current positions, not stable card IDs.
The newly drawn own card remains hidden to its owner and becomes visible to
the partner. Discarded cards and failed plays cannot be recovered.

## Observation arrays

The input is HanabiObservation with two fields:

- features: shape [658], dtype float32.
- legal_actions: shape [21], dtype bool.

Features are a flat vector of length 658. All intervals below are half-open.
A one-hot vector contains a single 1 at its indexed category; an all-zero vector indicates
absence where specified. A thermometer with value k contains k leading ones
and zeros thereafter. Counts are obtained by summing the indicated thermometer.
Indices for players are relative to the observer: 0 means self and 1 partner.
There is no separate absolute-seat or turn-counter feature. The no-op mask
entry identifies inactivity; it is false for the active seat.

| Feature interval | Shape / meaning |
|---|---|
| [0:125] | Partner hand, reshaped [5 slots, 5 colors, 5 ranks]. Entry 25*s+5*c+r is 1 for the card in partner slot s. An empty slot has 25 zeros. |
| [125:127] | Two missing-card flags in relative-player order. A flag is 1 when that hand has fewer than five cards; it is not a missing-card count. |
| [127:167] | Remaining draw-pile size, a 40-entry thermometer. Initially all 40 entries are 1. |
| [167:192] | Fireworks, reshaped [5 colors, 5 ranks]. Each color is one-hot at its highest successfully played rank ID. An all-zero vector indicates height 0; a one at rank ID r means height r+1. These rows are not thermometers. |
| [192:200] | Information tokens, an 8-entry thermometer. |
| [200:203] | Lives, a 3-entry thermometer. |
| [203:253] | Discard counts, reshaped [5 colors, 10 entries]. Within each color the five rank subblocks are [0:3], [3:5], [5:7], [7:9], [9:10]. Each subblock is a thermometer for copies of that rank discarded, including failed plays. Successful firework cards are not in this pile. |
| [253:308] | Most recent public action, detailed next. Entirely zero on reset. |
| [308:658] | Native per-card knowledge features, reshaped [2 relative players, 5 slots, 35 entries], detailed below. |

### Most recent action: offsets relative to 253

| Local interval | Meaning |
|---|---|
| [0:2] | Acting player, one-hot in observer-relative order. |
| [2:6] | Action type, one-hot in order play, discard, color hint, rank hint. |
| [6:8] | Hint recipient, observer-relative one-hot; zero for play/discard. |
| [8:13] | Named color, one-hot for a color hint; otherwise zero. |
| [13:18] | Named rank ID, one-hot for a rank hint; otherwise zero. |
| [18:23] | Five hint-match flags, indexed by recipient slot. Unmatched slots are 0. Zero for play/discard. |
| [23:28] | Removed slot, one-hot for play/discard, referring to the actor's hand before removal. Otherwise zero. |
| [28:53] | Revealed played/discarded identity, [5 colors, 5 ranks] one-hot. Otherwise zero. |
| [53] | Boolean: this action added a card to a firework. |
| [54] | Boolean: a play actually increased information tokens by completing a color. Zero for discards, even though a discard restores a token. |

The hand, board, discard and knowledge blocks describe the state after this
action; the removed-slot field refers to the state before it. The last-action
block is replaced on every transition and is not an accumulated history.
The active actor therefore alternates, but both seats can read each result.

### Native knowledge block: 35 entries per current slot

Entries [0:25] are the environment's native V0 weights, flattened color-major
as 5*c+r. Entries [25:30] are positive color-hint flags and [30:35] positive
rank-hint flags. Positive flags accumulate on matching cards and shift on
removal; absence of a positive flag does not by itself encode a negative hint.

For an occupied slot, V0 multiplies the native literal-possibility mask by the
remaining public counts for each card type, then normalizes this slot's 25
values to sum to one. Remaining public count is the original multiplicity
minus discarded copies minus the copy already in that color/rank firework
position. It does not subtract cards visible in the partner's hand or enforce
joint card-count consistency across slots. V0 is thus an observation feature,
not an exact joint distribution over hidden hands and not a prediction of
what another policy intends. The native possibility mask starts unrestricted,
is reduced by matching and nonmatching hint evidence, shifts with card removal,
and is reset for a new draw. For an empty slot, all V0 entries and positive
hint flags are zero. Both players' native knowledge records appear in relative order;
they are based on public hints, not private policy memory.

## Rewards and episode termination

Every successful play gives shared reward +1. Hints, discards and unsuccessful
plays that leave lives remaining give 0. On loss of the final life, the current
firework score is subtracted, making total episode return 0 (bomb-zero scoring).
The configured correction ends the episode immediately on final-life loss:
True. Completing all five fireworks ends the episode
with total return 25.

When the last draw is taken, the native final-round counter begins. There are
two further active-player turns: one for the other player and one for the
player who drew the last card, unless another termination condition occurs
first. There are no further draws. Hints still consume these turns. At final
round completion, the return is the sum of firework heights, unless all lives
were lost. The end of the deck alone does not immediately end the episode.
The safety cap is 128 transitions per evaluation episode; incomplete
episodes are rejected, not scored as ordinary successes or failures.

The adapter automatically resets after a terminal transition. At the following
policy call reset is true, the observation belongs to the next shuffled deal,
and private memory is reset to the value returned by its initializer. Policy inputs never include
hidden environment state or the reward. Evaluation tools report returns and
public observations separately. Illegal active actions invalidate evaluation;
the inactive seat is forced to no-op by the adapter.


Numeric dimensions, resolved flags and array schemas: specification.json.
