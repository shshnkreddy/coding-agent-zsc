# Two-player 9-card Yōkai: environment specification

## Goal, board and objects

The players cooperate to rearrange 9 cards so that all cards of each color
form one orthogonally connected group. There are 3 colors (IDs 0..2),
with 3 cards of each color. For connectivity, use horizontal and vertical neighbors rather than diagonals. A same-color connection requires a path made entirely of that color's cards. Colors may border one another, do not need empty gaps, and are not constrained to rows, columns, or any particular shape. Every legal move leaves all cards connected as a board by orthogonal adjacency regardless of their colors.

Cards occupy distinct cells of a 9×9 finite grid. Coordinates (x,y) use
x for the first array coordinate and y for the second; each is in 0..8.
There is no wraparound. All card positions, stable card IDs and locks are
public. Every card starts face down with its color hidden. Observing it privately reveals the color to that player only. Neither walls nor a draw pile exist, and there are no replacement cards, color changes, or removed cards.

There are also 4 hint tiles. Each shows a set of colors when revealed. A tile
can later be placed on a card, locking it for the rest of the episode. Hints
are separate objects from cards and do not participate in color connectivity.
Placing a hint on a card outside its color set is permitted by the rules. It is scored as an incorrect hint, not as an illegal move, and it does not reveal the hidden card color. The description supplies no convention for interpreting a player's choice of hint, observation, or movement.

## Action space and turn sequence

Choose a single scalar integer in 0..1067 per call. Its boolean legality mask, obs.legal_actions, has length 1068. The first active player is 0. Within a turn, that player executes substeps 0, 1, 2, 3 sequentially. Substep 3 ends the turn, and the other player becomes active at substep 0. Observations and policy calls occur for both seats at every substep. The inactive seat can legally choose only no-op 0 and is forced to no-op by the adapter. There is no pass for the active seat. Public rules determine legality; improving the hidden-color layout does not.

| Substep | Action and ID formula | Legal choices and effect |
|---|---|---|
| 0 | Observe card i: 1+i; or end game: 1067 | Reveal the color of any unlocked card privately for this turn, or immediately finish and score the present arrangement. |
| 1 | Observe card i: 1+i | Choose an unlocked card that has not already been observed this turn. No action to end the game is permitted at this point. |
| 2 | Move card i to (x,y): 10 + i*81 + x*9 + y | Relocate an unlocked card into an empty grid cell distinct from its current location, ensuring all 9 cards are connected in the new arrangement. The moved card may be outside the two observed cards. |
| 3 | Reveal hint h: 739+h; or place hint h at (x,y): 743 + h*81 + x*9 + y | Choose between publicly exposing an unrevealed hint's colors and placing a previously revealed, unplaced hint on an unlocked card at (x,y). |

The action intervals are: no-op [0:1], observe [1:10], move
[10:739], reveal-hint [739:743], place-hint [743:1067],
and end-game [1067:1068]. Index IDs i and h from zero. A place action names a grid cell, not a card ID, with an unlocked card required at that location.

Relocating a card leaves its ID, color, and both players' observation history intact. Only the destination and final whole-board connectivity matter for the move, not a traversed path or imagined intermediate removal. Distance has no cost, and the board remaining during that imagined removal need not be connected. Locked cards cannot move or be observed again. No swap, same-cell stay, or separately moving a hint is allowed.

A hint's reveal action and placement action occur on different turns. Revealed hints cannot be concealed again. Placed hints are immovable and cannot be replaced or removed. Since placement locks a card, only one hint can be attached to it. Revealing one hint or placing one hint concludes substep 3. All 4 hints being placed causes immediate termination. Otherwise the episode proceeds even when every color is connected, without an automatic connectivity-based ending.

## Observation arrays and exact fields

These fields constitute the YokaiObservation input:

| Field | Shape | Dtype |
|---|---|---|
| node_features | [13, 13] | int8 |
| adjacency_matrix | [13, 13] | uint8 |
| positions | [9, 2] | int32 |
| legal_actions | [1068] | bool |

There are 13 graph nodes: rows 0..8 are stable card IDs, and rows
9..12 are hint IDs 0..3. Movement never reorders the node rows. The graph assigns 13 features to each node. It stores small integer values and binary indicators directly and does not normalize coordinates or IDs.

| Feature columns | Card row | Hint row |
|---|---|---|
| [0:3] | Private true-color information, encoded one-hot; use all zero if unknown/masked. | Always zero. |
| [3:6] | Zero for all cards, including those bearing placed hints. | Colors in the revealed hint set are multi-hot; an unrevealed hint has all zero. |
| [6:8] | Current (x,y) position. | The current location (x,y) uses (h,-1) prior to placement and the target card's position following placement. |
| [8] | A value of 1 indicates a hint lock; absent a lock, 0. | 1 if placed, else 0. |
| [9] | The other seat's observation of this card is marked publicly, with timing governed by the memory description below. | Always zero. |
| [10] | Repeated across rows: 1 for an active observation recipient, else 0. | Same active-seat flag. |
| [11] | All rows contain the current substep 0..3. | Same substep. |
| [12] | Stable card ID 0..8. | The value -(h+1) is the negative stable ID of hint h. |

The all-zero card-color encoding denotes an unknown or masked color, without adding an extra color. Hint nodes store color sets; card-row hint-color columns do not mirror those sets. Cards may not use the off-board location of an unplaced hint as a destination.
The separate positions array is shaped [9,2]. Its rows reproduce the card-node coordinates; hint positions are excluded.

A binary, symmetric [13,13] matrix encodes adjacency, including each node's self-loop. Two distinct card nodes connect only and exactly when one orthogonal grid step separates their positions. A placed hint and its target card are linked bidirectionally. Unplaced hints have no edges except self-loops. There are no further hint-card edges or edges joining separate hints. Adjacency ignores colors, and two cards are not connected by a hint edge.

## Visibility and memory: standard

With perfect memory off, the current active turn's observed cards are the only colors exposed in own-color features. Those colors disappear at turn end; all own-color features are zero while inactive. The policy is free to retain its past observations in private memory, without the environment restoring them later.

The first observe action exposes its card in the observation for substep-1. Following the second observe action, both cards remain visible from substep 2 through substep 3, whether or not one moves. The current-turn reveal flags clear at the turn boundary. An unobserved card's color is not exposed by relocating it. Both seats continue to observe all positions, locks, revealed hint sets, placements, the substep, and active-seat flags, regardless of which player is active.

Whether partner-seen spans the episode or just the current turn depends on perfect memory. If enabled, it records whether the partner ever observed each card this episode; if disabled, it records observations added in the current turn and clears at every full-turn boundary. It identifies cards observed, never colors seen. Hint colors remain public from reveal through placement. A known locked card's previously observed color follows the configured memory rule, not an additional disclosure to the partner when it is locked.

An explicit last-action field, an episode counter, a hidden-color belief vector, and simulator state are all absent from observations. Memory and PRNG used by the policy are private per seat. Observations never recommend which card to observe, where to move it, what hint to use, or when to finish.

## Reset distribution and hint identities

Stable IDs receive a shuffled assignment of the fixed card-color multiset on reset. Card i
starts at (floor(i/3)+3, (i mod 3)+3), producing a centered
3×3 block. All cards start without locks or observations; all hints start unrevealed and unplaced at (h,-1). Player 0 begins at substep 0, and adjacency is computed from the initial positions. The true card colors and hint color sets are initially concealed from both seats.

A uniform draw from the three colors supplies hint 0's one color. Hints 1, 2, 3 each have two colors, using all three distinct unordered pairs shuffled into random order.
For the full episode, hint IDs preserve their sampled sets. The reset rule fixes the size category of each hint ID while keeping its sampled colors hidden until revelation. Whether a later placement card matches does not affect sampling. A reset board can have every color connected without explicitly signaling it in a separate observation flag.

## Terminal scoring and episode boundaries

With upstream_sparse configured, nonterminal steps always earn zero. If the terminal arrangement connects every color, the players' shared reward is

5*U + 2*R + C - I,

where U is the number of unrevealed hints, R the number revealed but unplaced, C the number of correct placements, and I the number of incorrect placements. Every one of the 4 hints falls in exactly one category. A placed hint is correct precisely when its revealed set contains the underlying color. Success need not yield a positive score: sufficiently many incorrect placements make it negative. When not every color is clustered, the terminal reward remains 0 irrespective of hint correctness or the explicit end action.

The configured return interval is -4..20. If every color's cards form a connected group, ending immediately scores 20; with every hint placed and each color forming a connected group, the score is C-I instead. If all 4 placed hints are incorrect while each color forms a connected group, the score is -4, accounting for the negative sparse-reward bound. If any color's cards are not connected to one another through cards of that color, the board gets the configured failure reward, without a separate cluster-formation bonus or any intermediate shaping reward.

Each hint may be revealed only once and placed only once. One of these actions must be taken at substep 3. There can therefore be at most 8 full turns (32 substeps) before every hint has been placed, unless an explicit end at substep 0 stops play sooner. The safety cap for evaluation is 32 substeps. The active player must act legally and terminate within the cap for evaluation to be valid. Termination is followed by a fresh reset observation from the adapter and private-memory reset via the next policy call's reset flag. Metrics and returns are logged separately rather than exposing rewards or hidden state as policy inputs.


Find numeric dimensions, resolved flags, and array schemas in specification.json.
