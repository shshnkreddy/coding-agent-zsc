# Two-player 16-card Yōkai: environment specification

## Goal, board and objects

The players cooperate to rearrange 16 cards so that all cards of each color
form one orthogonally connected group. There are 4 colors (IDs 0..3),
with 4 cards of each color. After a legal move, the whole board must still be connected by orthogonal card adjacency irrespective of color. Color connectivity has the additional requirement that paths use only cards of that color. Only horizontal and vertical neighbors connect, never diagonals. Different colors may touch without empty space between them, and a color is not required to form a row, column, or other designated shape.

Cards occupy distinct cells of a 10×10 finite grid. Coordinates (x,y) use
x for the first array coordinate and y for the second; each is in 0..9.
There is no wraparound. All card positions, stable card IDs and locks are
public. Cards are initially face down; a private observation makes a card's color visible to the observing player alone. There are no walls, no draw pile, no replacement cards, no changes of color, and no removed cards.

There are also 7 hint tiles. Each shows a set of colors when revealed. A tile
can later be placed on a card, locking it for the rest of the episode. Hints
are separate objects from cards and do not participate in color connectivity.
The hidden card color is not revealed by placing a hint. Placement is legal even if that color is missing from the hint's set, although such a hint is incorrect when scored. No convention is provided to interpret the player's selection of a hint, observation, or movement.

## Action space and turn sequence

A call produces one scalar integer from 0..2324, with legality given by the length 2325 boolean vector obs.legal_actions. Play begins with player 0; each turn gives the same active player substeps 0, 1, 2, 3 in that order. Once substep 3 finishes, the other player is active at substep 0. At each substep, both policies receive observations and are called. The adapter forces the inactive seat's sole legal choice, no-op 0. The active player must act rather than pass. A legal mask entry checks public constraints, without judging the hidden-color arrangement.

| Substep | Action and ID formula | Legal choices and effect |
|---|---|---|
| 0 | Observe card i: 1+i; or end game: 2324 | Privately reveal any unlocked card's color for the turn; alternatively, end now and score the existing arrangement. |
| 1 | Observe card i: 1+i | Select a second, different unlocked card for observation. This substep does not permit an end-game action. |
| 2 | Move card i to (x,y): 17 + i*100 + x*10 + y | Choose any unlocked card and move it to a different vacant grid cell with all 16 cards connected in the result. Observation of that card this turn is not required. |
| 3 | Reveal hint h: 1617+h; or place hint h at (x,y): 1624 + h*100 + x*10 + y | Reveal a previously unrevealed hint to both players; otherwise place a revealed, unplaced hint on an unlocked card located at (x,y). |

The action intervals are: no-op [0:1], observe [1:17], move
[17:1617], reveal-hint [1617:1624], place-hint [1624:2324],
and end-game [2324:2325]. Use zero-based IDs i and h. Place actions identify a grid cell instead of a card ID and require an unlocked card there.

The moved card keeps its ID and color as well as both players' observation history. Movement is relocation, with tests only of the destination and resulting whole-board connectivity, not path traversal. It has no distance cost and imposes no connectivity requirement on an imagined intermediate board after removal. Locked cards cannot move or be observed again. Moves cannot swap cards, stay in the same cell, or move a hint separately.

Use different turns for a hint's reveal and placement actions. Neither hiding a revealed hint nor moving, replacing, or removing a placed hint is allowed. The lock created by placement limits each card to one hint. One reveal or placement resolves substep 3. Once all 7 hints are placed, termination is immediate; before that, play may continue with every color connected because connectivity is not an automatic stopping condition.

## Observation arrays and exact fields

YokaiObservation supplies these fields to the policy:

| Field | Shape | Dtype |
|---|---|---|
| node_features | [23, 15] | int8 |
| adjacency_matrix | [23, 23] | uint8 |
| positions | [16, 2] | int32 |
| legal_actions | [2325] | bool |

There are 23 graph nodes: rows 0..15 are stable card IDs, and rows
16..22 are hint IDs 0..6. Object movement leaves node rows in the same order. Nodes each contain 15 features. Small integers and binary flags are represented directly rather than normalized, as are coordinates and IDs.

| Feature columns | Card row | Hint row |
|---|---|---|
| [0:4] | A privately observed true-color one-hot vector; unknown/masked colors give all zero. | Always zero. |
| [4:8] | Remains zero even for a card with a hint attached. | The revealed color set uses multi-hot encoding, with all zero while unrevealed. |
| [8:10] | Current (x,y) position. | Current position (x,y), initially (h,-1) while unplaced and matching the target card once placed. |
| [10] | Value 1 denotes locking by a hint, and 0 denotes no lock. | 1 if placed, else 0. |
| [11] | Publicly marks this card as observed by the other seat, subject to the memory timing below. | Always zero. |
| [12] | The observing seat's activity is encoded as 1 or otherwise 0 on every row. | Same active-seat flag. |
| [13] | Current substep value 0..3, identical on all rows. | Same substep. |
| [14] | Stable card ID 0..15. | The stable ID is negative; use -(h+1) for hint h. |

An all-zero card-color vector represents an unknown or masked color, not an extra color. Hint color sets reside on hint nodes and are never copied to the corresponding card-row columns. An unplaced hint is off-board; its coordinate is not a valid destination for a card.
A separate positions array of shape [16,2] copies card-node coordinate columns. It contains card positions exclusively, with no rows for hints.

Adjacency uses a binary [23,23] symmetric matrix with self-loops on all nodes. An edge between distinct card nodes exists exactly for positions separated by one orthogonal grid step. Placed hints connect both ways to their target cards; unplaced hints have self-edges only. There are no additional hint-card links and no links joining different hints. Neither colors nor hint-mediated card-to-card connections contribute to adjacency.

## Visibility and memory: perfect

With perfect memory enabled, earlier personal observations in this episode remain available through own-color features whenever your seat is active. This includes moved and subsequently locked cards, whose identity follows the stable card ID. During inactivity all these features are zero, even previously seen colors, and they return on becoming active again. Colors privately observed by the other player are never added.

Inspecting the first card reveals it in the returned substep-1 observation. Both inspected cards can then be seen at substep 2 after the second inspection and through substep 3 despite movement. The turn's reveal flags are erased at turn end. Movement alone never discloses the color of an unobserved card. All card positions, locks, revealed hint sets, placements, substep, and active-seat flags remain available to both seats even when inactive.

Perfect memory makes partner-seen report whether the other player has ever observed a card during the episode. Without perfect memory, the column contains observations added in the current turn and is reset at each full-turn boundary. What is indicated is the observed ID, never the observed color. Hint colors become public when revealed and stay public when placed. Previously seen colors on known locked cards remain subject to the memory configuration, and locking does not communicate those colors to the partner.

The observation supplies neither simulator state nor an explicit last-action field, episode counter, or hidden-color belief vector. Policy memory and PRNG belong privately to each seat. Advice about a card to observe, a destination, a hint, or an end decision is not part of the observation.

## Reset distribution and hint identities

The reset distribution permutes a fixed card-color multiset over stable IDs. Card i
starts at (floor(i/4)+3, (i mod 4)+3), producing a centered
4×4 block. No card starts locked or observed. Hint tiles begin unrevealed and unplaced at (h,-1), with player 0 active and substep 0 selected. The initial positions define adjacency. Both seats start without access to true card colors or hint color sets.

The one-color hints are 0, 1; the two-color hints are 2, 3, 4; the three-color hints are 5, 6. For each size, choose different subsets without replacement from all subsets of that size of the four colors.
A sampled color set stays with its hint ID for the entire episode. Size categories follow the reset rule for each hint ID, whereas sampled colors are concealed until reveal. Later placement choices and matches do not influence sampling. The initial board may already connect all colors, with no separate flag indicating this.

## Terminal scoring and episode boundaries

The configured upstream_sparse scoring supplies zero on nonterminal steps. At a terminal board where each color is connected, the common reward is

5*U + 2*R + C - I,

where the quantities are U for unrevealed hints, R for revealed unplaced hints, C for correct placements, and I for incorrect placements. These disjoint categories exhaust the 7 hints. Correctness means precisely that the placed hint's revealed set contains its card's color. A successful episode may consequently receive a negative score if too many placements are incorrect. The terminal reward is 0 whenever a color is unclustered; neither correct hint placement nor the explicit end action changes that.

Configured returns lie within -7..35. Ending immediately on an arrangement where each color forms a connected group gives 35; if each color forms a connected group and every hint is placed, the score becomes C-I. If all 7 hints are incorrect while each color forms a connected group, the score is -7, which accounts for the negative bound under sparse scoring. If any color's cards fail to form a connected group, use the configured failure reward; clustering produces no additional bonus and intermediate steps provide no shaping reward.

Revelation and placement can each occur once per hint; one of them is mandatory at substep 3. All hint placements must therefore finish in at most 14 full turns, or 56 substeps, unless the explicit end action is taken sooner at substep 0. Evaluation is capped at 56 substeps. An illegal active choice or nontermination within the cap makes evaluation invalid. The adapter follows termination with a new reset observation; the next policy call receives a reset flag to clear its private memory. Episode metrics and returns are kept separately, and policy inputs never include rewards or hidden state.


The specification.json file supplies numeric dimensions, resolved flags, and array schemas.
