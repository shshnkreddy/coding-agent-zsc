# Two-player 16-card Yōkai: environment specification

## Goal, board and objects

The players cooperate to rearrange 16 cards so that all cards of each color
form one orthogonally connected group. There are 4 colors (IDs 0..3),
with 4 cards of each color. A connection uses horizontal or vertical
neighbors, never diagonals. For a color to count as connected, paths between
its cards must use only cards of that color. Different colors may touch; there
is no requirement to separate them by empty space. Colors need not form rows,
columns or any prescribed shape. The entire board must remain connected by
orthogonal card adjacency after each legal move, regardless of colors.

Cards occupy distinct cells of a 10×10 finite grid. Coordinates (x,y) use
x for the first array coordinate and y for the second; each is in 0..9.
There is no wraparound. All card positions, stable card IDs and locks are
public. Cards begin face down, and private observations reveal their colors only
to the observing player. There are no walls, draw pile, replacement cards,
color changes or removed cards.

There are also 7 hint tiles. Each shows a set of colors when revealed. A tile
can later be placed on a card, locking it for the rest of the episode. Hints
are separate objects from cards and do not participate in color connectivity.
A hint may legally be placed on a card whose true color is absent from its
set. This is an incorrect hint for scoring, but the move is still legal and
does not reveal the hidden card color. No convention for interpreting a
player's choice of hint, observation or movement is supplied.

## Action space and turn sequence

Every call returns one scalar integer in 0..2324; obs.legal_actions is a
boolean vector of length 2325. Player 0 starts. Each full turn
contains four substeps by the same active player, in order 0, 1, 2, 3. After
substep 3 the other player becomes active at substep 0. Both policy instances
receive observations and are called at every substep; the inactive seat's
only legal action is no-op 0. The adapter forces that seat to no-op. The active
seat cannot pass. A true legal-action entry reflects public rule constraints,
not whether a move improves the hidden-color arrangement.

| Substep | Action and ID formula | Legal choices and effect |
|---|---|---|
| 0 | Observe card i: 1+i; or end game: 2324 | Observe any unlocked card, revealing its color privately for this turn, or terminate immediately and score the current arrangement. |
| 1 | Observe card i: 1+i | Observe a different unlocked card, not the one already observed this turn. No end-game action is allowed here. |
| 2 | Move card i to (x,y): 17 + i*100 + x*10 + y | Move any unlocked card to an empty in-grid cell, different from its current cell, such that all 16 cards are connected afterward. The moved card need not be one of the two observed cards. |
| 3 | Reveal hint h: 1617+h; or place hint h at (x,y): 1624 + h*100 + x*10 + y | Reveal one unrevealed hint, making its color set public; alternatively place one already revealed, unplaced hint on an unlocked card occupying (x,y). |

The action intervals are: no-op [0:1], observe [1:17], move
[17:1617], reveal-hint [1617:1624], place-hint [1624:2324],
and end-game [2324:2325]. IDs i and h are zero-based. Place actions address
a grid cell, not a card ID; that cell must currently contain an unlocked card.

A move is a relocation, not a path traversal: only the destination and the
resulting whole-board connectivity are tested. There is no movement distance
cost or requirement that the remaining board stay connected during an imagined
intermediate removal. The moved card retains its ID, color and both players'
observation history. Locked cards cannot move or be observed again. There is
no swap action, no staying in the same cell, and no moving a hint separately.

Revealing and placing a hint are separate actions on separate turns. A revealed
hint cannot be hidden again. A placed hint cannot be moved, replaced or removed.
Only one hint can occupy a card because placement locks it. Revealing or placing
one hint completes substep 3. The game terminates immediately when all 7 hints
have been placed. Otherwise play continues even if all color groups happen to
be connected: connectivity alone does not automatically end the episode.

## Observation arrays and exact fields

The input is YokaiObservation with these fields:

| Field | Shape | Dtype |
|---|---|---|
| node_features | [23, 15] | int8 |
| adjacency_matrix | [23, 23] | uint8 |
| positions | [16, 2] | int32 |
| legal_actions | [2325] | bool |

There are 23 graph nodes: rows 0..15 are stable card IDs, and rows
16..22 are hint IDs 0..6. Node rows do not reorder when objects move.
The graph has 15 features per node. Small integer values and binary
indicators are stored directly; coordinates and IDs are not normalized.

| Feature columns | Card row | Hint row |
|---|---|---|
| [0:4] | Privately visible true color, one-hot; all zero if unknown/masked. | Always zero. |
| [4:8] | Always zero, including cards with placed hints. | Revealed hint color set, multi-hot; all zero before reveal. |
| [8:10] | Current (x,y) position. | Current (x,y) position: before placement (h,-1), afterward the target card's position. |
| [10] | 1 if locked by a hint, else 0. | 1 if placed, else 0. |
| [11] | Public indicator that the other seat observed this card, with memory timing described below. | Always zero. |
| [12] | 1 if this observation's seat is active, else 0; repeated on every row. | Same active-seat flag. |
| [13] | Current substep 0..3; repeated on every row. | Same substep. |
| [14] | Stable card ID 0..15. | Negative stable ID: hint h has value -(h+1). |

An all-zero card-color vector indicates an unknown or masked color, not an additional color. Hint color sets are
found on hint nodes; they are not copied into the card row's hint-color columns.
An unplaced hint's off-board coordinate is not a legal card destination.
The separate positions array has shape [16,2] and contains only card positions;
it duplicates the card-node coordinate columns, without hint rows.

Adjacency is a symmetric binary [23,23] matrix. All nodes have self-loops.
Two different card nodes are connected exactly when their positions are one
orthogonal grid step apart. A placed hint node is connected to its target
card node in both directions. An unplaced hint has no non-self edges. There
are no other hint-card edges and no edges between distinct hint nodes.
Colors do not affect adjacency, and a hint edge does not connect two cards.

## Visibility and memory: perfect

With perfect memory enabled, during your active turns the own-color
features expose every card you have personally observed at any earlier time
in this episode, including moved or subsequently locked cards. Identity is
attached to stable card ID. While your seat is inactive, all own-color features
are zero, including previously observed colors. They reappear when your seat
is active again. The other player’s private color observations are never
added to your own-color features.

The first observation action reveals its card in the returned substep-1
observation. After the second observation, both are visible at substep 2 and
remain so through substep 3, even if one is moved. At turn end the current-turn
reveal flags are cleared. Movement does not reveal an unobserved card's color.
All positions, locks, revealed hint sets, placements, substep and active-seat
flags remain visible to both seats, including the inactive player.

With perfect memory, the partner-seen column records whether the other player
has ever observed each card in this episode. Without perfect memory, it reports
observations added during the current turn, and is cleared at every full-turn
boundary. This indicator says which IDs were observed, never what colors the
other player saw. Hint colors become public on reveal and remain public after
placement. A known locked card's previously observed color still follows the
configured memory rule; locking does not communicate that color to a partner.

There is no explicit last-action field, episode counter, hidden-color belief
vector or simulator state in the observation. Policy memory and PRNG are
private to each seat. The observation contains no recommendation of which
card to observe, where to move it, which hint to use, or when to end.

## Reset distribution and hint identities

A reset shuffles the fixed multiset of card colors over stable IDs. Card i
starts at (floor(i/4)+3, (i mod 4)+3), producing a centered
4×4 block. All cards are unlocked and unobserved; hint tiles are unrevealed
and unplaced at (h,-1). The starting player is player 0 and the substep is 0.
The initial adjacency follows these positions. Initially no true card colors
or hint color sets are exposed to either seat.

Hints 0, 1 each have one color, hints 2, 3, 4 each have two colors, and
hints 5, 6 each have three colors. Within each size, distinct color subsets
are sampled without replacement from all subsets of the four colors.
Hint IDs retain their sampled sets for the whole episode. The size category
of each hint ID is fixed by this reset rule; its sampled color set is hidden
until reveal. Sampling does not depend on whether a hint matches a card later
chosen for placement. A board may already have all colors connected at reset,
but that fact is not revealed by a separate observation flag.

## Terminal scoring and episode boundaries

Configured scoring is upstream_sparse. All nonterminal rewards are zero.
At termination, if every color is connected, the shared reward is

5*U + 2*R + C - I,

where U is the number of unrevealed hints, R revealed but unplaced hints,
C correctly placed hints and I incorrectly placed hints. These categories
partition the 7 hints. A placed hint is correct exactly when the underlying
card's color belongs to its revealed set. A successful game can therefore
have a negative score if enough placed hints are incorrect. If any color is not clustered, the terminal reward is 0 regardless of
hint correctness or the explicit end action.

The configured return bounds are -7..35. For example,
ending immediately on a board where each color forms a connected group gives 35; with all hints
placed on such a board the score is C-I. If each color forms a connected group and all 7
hints are wrong, the score is -7, explaining the negative lower bound even with sparse
rewards. If any color's cards do not form a connected group, the configured failure reward applies. There is no
separate bonus for forming a cluster and no intermediate shaping reward.

Each hint can be revealed once and placed once; substep 3 must do one of those
actions. Thus at most 14 full turns (56 substeps) occur before every hint
is placed, unless the players explicitly end earlier at substep 0. The
evaluation safety cap is 56 substeps. An illegal active action or
failure to terminate within the cap invalidates evaluation. After termination
the adapter returns a fresh reset observation, and the next policy call's
reset flag resets private memory. Episode metrics and returns are recorded
separately; rewards and hidden state are never policy inputs.


Numeric dimensions, resolved flags and array schemas: specification.json.
