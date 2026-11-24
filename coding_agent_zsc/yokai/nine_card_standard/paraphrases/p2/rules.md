# Two-player 9-card Yōkai: environment specification

## Goal, board and objects

The players cooperate to rearrange 9 cards so that all cards of each color
form one orthogonally connected group. There are 3 colors (IDs 0..2),
with 3 cards of each color. Diagonal neighbors do not connect; horizontal and vertical neighbors do. Paths connecting cards of a color must stay within that color. Different colors need not be separated by empty cells and may touch. No row, column, or other prescribed shape is required for a color. Independently of colors, every legal move must leave the whole board connected through orthogonal card adjacency.

Cards occupy distinct cells of a 9×9 finite grid. Coordinates (x,y) use
x for the first array coordinate and y for the second; each is in 0..8.
There is no wraparound. All card positions, stable card IDs and locks are
public. Cards initially lie face down, and a private observation reveals a card's color only to the player making that observation. The game has no walls or draw pile, no replacement or removed cards, and no color changes.

There are also 4 hint tiles. Each shows a set of colors when revealed. A tile
can later be placed on a card, locking it for the rest of the episode. Hints
are separate objects from cards and do not participate in color connectivity.
Hint placement legality does not require the card's true color to belong to the hint's set. A mismatch is incorrect for scoring purposes but legal to place, without revealing the hidden card color. Interpretations of players' hint, observation, and movement choices have no supplied convention.

## Action space and turn sequence

Each call must output a scalar integer in 0..1067. The boolean vector obs.legal_actions has length 1068. The starting player is 0. The active player takes four substeps per turn, ordered 0, 1, 2, 3. Following substep 3, the other player starts at substep 0. Every substep calls both policy instances and supplies each with an observation. The inactive seat is restricted to no-op 0, enforced by the adapter; the active seat has no pass action. Mask legality follows public rules and does not indicate whether hidden-color organization improves.

| Substep | Action and ID formula | Legal choices and effect |
|---|---|---|
| 0 | Observe card i: 1+i; or end game: 1067 | Choose an unlocked card to reveal its color privately this turn, or stop at once and receive the current arrangement's score. |
| 1 | Observe card i: 1+i | Observe another unlocked card, excluding this turn's first observed card. The end-game action cannot be taken here. |
| 2 | Move card i to (x,y): 10 + i*81 + x*9 + y | Move an unlocked card to an unoccupied in-grid cell other than its present cell; all 9 cards must remain connected afterward. The card need not belong to the two inspected this turn. |
| 3 | Reveal hint h: 739+h; or place hint h at (x,y): 743 + h*81 + x*9 + y | Make an unrevealed hint's color set public, or attach an already revealed and unplaced hint to an unlocked card at (x,y). |

The action intervals are: no-op [0:1], observe [1:10], move
[10:739], reveal-hint [739:743], place-hint [743:1067],
and end-game [1067:1068]. IDs i and h use zero-based indexing. Placement targets a cell, not a card ID; the target must hold an unlocked card.

Only a move's destination and final whole-board connectivity are checked: the card is relocated, not carried along a path. No distance cost applies, nor must the remaining board be connected at an imagined removal stage. The moved card's ID, color, and both players' observation history persist. Locked cards allow neither movement nor further observation. There is no swapping, staying in place, or separate movement of hints.

Revelation and placement require separate turns and actions. Revealed hints never become hidden again; placed hints cannot move, be replaced, or be removed. A card holds at most one hint because placement locks the card. Either revealing or placing a hint finishes substep 3. The episode ends immediately upon placement of all 4 hints. If that has not happened, connected color groups do not automatically stop play.

## Observation arrays and exact fields

Input fields are provided in YokaiObservation:

| Field | Shape | Dtype |
|---|---|---|
| node_features | [13, 13] | int8 |
| adjacency_matrix | [13, 13] | uint8 |
| positions | [9, 2] | int32 |
| legal_actions | [1068] | bool |

There are 13 graph nodes: rows 0..8 are stable card IDs, and rows
9..12 are hint IDs 0..3. Node row ordering stays fixed as objects move. There are 13 features for each node; small integers and binary indicators appear as raw values, with no normalization of coordinates or IDs.

| Feature columns | Card row | Hint row |
|---|---|---|
| [0:3] | The true color in a private one-hot encoding, or all zero if unknown/masked. | Always zero. |
| [3:6] | Zero regardless of whether the card carries a placed hint. | Hint colors are multi-hot after reveal and all zero beforehand. |
| [6:8] | Current (x,y) position. | The current (x,y) location is (h,-1) before placement and the target card's location afterward. |
| [8] | A hint-locked card has 1; an unlocked card has 0. | 1 if placed, else 0. |
| [9] | Whether the other player observed this card is publicly indicated, using the memory timing below. | Always zero. |
| [10] | Every row carries 1 for an active observing seat or 0 for an inactive one. | Same active-seat flag. |
| [11] | Each row repeats the current substep 0..3. | Same substep. |
| [12] | Stable card ID 0..8. | A negative stable ID, with hint h encoded as -(h+1). |

Zero in every card-color entry means the color is unknown or masked, rather than an additional color category. Hint nodes, rather than copied values in card-row hint-color columns, contain hint color sets. A card cannot legally move to an unplaced hint's off-board coordinate.
The positions array, with shape [9,2], repeats the coordinate columns for card nodes only; it has no hint rows.

A symmetric binary matrix of shape [13,13] represents adjacency. Each node connects to itself. Different cards are adjacent precisely when their grid positions differ by one orthogonal step. A hint acquires a bidirectional edge to its target card when placed. Before placement, its only edge is its self-loop. Other hint-card edges and edges between distinct hints are absent. Colors play no part in adjacency, and a hint edge cannot connect a pair of cards.

## Visibility and memory: standard

Without perfect memory, own-color features are limited to observations made in the current active turn. The colors vanish when the turn ends, and inactive seats have all-zero own-color features. A policy may remember its own observations privately; later environment observations do not restore them.

The first inspected card becomes visible in the substep-1 observation. The second inspection makes both visible at substep 2, and they remain visible through substep 3 even if a card moves. Turn completion clears the current-turn reveal flags. An unobserved card stays color-hidden when moved. Positions, lock status, revealed hint sets, placements, substep, and active-seat indicators stay public to both the active and inactive seats.

The partner-seen indicator never reports which colors the other player saw, only which IDs they observed. With perfect memory it remembers observations throughout this episode; without it, only observations added during the current turn are reported, and every full-turn boundary clears the indicator. Revealed hint colors stay public after placement. A known locked card retains its previously observed color only as the configured memory rule specifies; locking itself does not convey the color to a partner.

Each seat's policy memory and PRNG are private. The observation has no explicit last-action field, episode counter, hidden-color belief vector, or simulator state, and gives no recommendation about what to observe, where to move, which hint to choose, or when to end.

## Reset distribution and hint identities

At reset, the fixed multiset of colors is shuffled among stable card IDs. Card i
starts at (floor(i/3)+3, (i mod 3)+3), producing a centered
3×3 block. Initially, every card is unlocked and unobserved, and hints are unrevealed and unplaced at (h,-1). Play starts with player 0 at substep 0. Initial adjacency is derived from these positions, with true card colors and hint color sets hidden from both seats.

One color is assigned to hint 0, while two colors are assigned to each of hints 1, 2, 3. The single color is drawn uniformly from the three colors; the two-color hints are a random ordering of all three distinct unordered pairs.
Hint IDs have episode-long sampled sets. This reset rule determines the size category for each hint ID; only revealing the hint exposes its sampled set. The sampling procedure does not consider whether the hint will match a later placement target. Reset can produce connected groups for every color, but no dedicated observation flag reports this.

## Terminal scoring and episode boundaries

Under upstream_sparse scoring, every nonterminal reward is zero. On termination with all colors connected, the shared reward equals

5*U + 2*R + C - I,

where U denotes unrevealed hints, R revealed-but-unplaced hints, C correctly placed hints, and I incorrectly placed hints. Together these categories account for all 4 hints. A placement is correct exactly when its revealed set includes the card's color, so enough incorrect placements can make a successful game's score negative. Any unclustered color forces terminal reward 0, independently of hint correctness and the explicit end action.

The return range is configured as -4..20. Immediate termination when every color's cards form a connected group earns 20; if that condition holds with every hint placed, the score is C-I. If all 4 hints are wrong while every color still forms a connected group, the return is -4, so sparse rewards still allow a negative minimum. The configured failure reward applies if the cards of any color do not form a connected group. Neither a standalone cluster bonus nor intermediate reward shaping is provided.

Each hint has a single reveal and a single placement available. Substep 3 is required to use one of those actions. This permits no more than 8 complete turns (32 substeps) before all hints are placed, or fewer if players end explicitly at substep 0. The evaluation limit is 32 substeps. Illegal actions by the active player and failure to finish within the cap both invalidate evaluation. Termination causes the adapter to return a fresh reset observation, and the following policy call resets private memory via its reset flag. Episode returns and metrics are separate records, not reward or hidden-state policy inputs.


Numeric sizes, resolved flags, and array schemas are recorded in specification.json.
