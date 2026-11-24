# Two-player 9-card Yōkai: environment specification

## Goal, board and objects

The players cooperate to rearrange 9 cards so that all cards of each color
form one orthogonally connected group. There are 3 colors (IDs 0..2),
with 3 cards of each color. Horizontal or vertical card neighbors establish connections, whereas diagonal neighbors do not. A color's cards must connect by paths containing that color alone. There is no prescribed row, column, or shape for a color, and touching another color is allowed without an empty-space separator. Legal moves must also preserve whole-board connectivity under orthogonal adjacency, ignoring colors.

Cards occupy distinct cells of a 9×9 finite grid. Coordinates (x,y) use
x for the first array coordinate and y for the second; each is in 0..8.
There is no wraparound. All card positions, stable card IDs and locks are
public. Cards begin face down. Private observations expose their colors only to the player who observes them. Cards are not replaced, removed, or recolored, and there are neither walls nor a draw pile.

There are also 4 hint tiles. Each shows a set of colors when revealed. A tile
can later be placed on a card, locking it for the rest of the episode. Hints
are separate objects from cards and do not participate in color connectivity.
A hint set need not contain a card's true color for placement to be legal. When it does not, the hint is incorrect for scoring but the placement is still legal and keeps the card color hidden. There is no supplied convention governing how to interpret hint choices, observation choices, or movements.

## Action space and turn sequence

The call's output is a scalar integer in 0..1067; the corresponding obs.legal_actions boolean vector contains 1068 entries. Player 0 takes the initial turn. The same player stays active through the sequence 0, 1, 2, 3. After substep 3, activity switches to the other player and the substep becomes 0. Each policy is called with an observation on every substep, including the inactive seat, whose only allowed action is adapter-enforced no-op 0. The active seat may not pass. A true mask entry encodes public action restrictions, not hidden-color improvements.

| Substep | Action and ID formula | Legal choices and effect |
|---|---|---|
| 0 | Observe card i: 1+i; or end game: 1067 | Select any unlocked card for this turn's private color observation, or terminate at once using the current board for scoring. |
| 1 | Observe card i: 1+i | Inspect a different unlocked card from the one observed earlier this turn; ending the episode is not legal here. |
| 2 | Move card i to (x,y): 10 + i*81 + x*9 + y | An unlocked card may move to any other empty in-grid cell provided all 9 cards are connected afterward. It does not have to be one of the pair observed. |
| 3 | Reveal hint h: 739+h; or place hint h at (x,y): 743 + h*81 + x*9 + y | Publicly reveal one hidden hint's color set, or put a revealed hint that has not been placed on the unlocked card occupying (x,y). |

The action intervals are: no-op [0:1], observe [1:10], move
[10:739], reveal-hint [739:743], place-hint [743:1067],
and end-game [1067:1068]. IDs i and h are indexed from zero. The destination of a place action is a grid cell containing an unlocked card, not a card ID.

A movement action checks the destination and the whole board's connectivity after relocation; it does not trace a path. There is no charge for distance and no requirement for connectivity during an imagined intermediate removal. ID, color, and both players' observation histories stay with the moved card. Locked cards cannot be observed again or moved. Neither swapping nor staying in the same cell nor separate hint movement is an action.

Reveal and placement are two actions performed on separate turns. Once revealed, a hint stays revealed; once placed, it stays in place without replacement or removal. Placement locks the card, so it cannot receive another hint. To complete substep 3, reveal or place one hint. The last placement among the 4 hints ends the game immediately. Connecting every color group does not itself end an episode that would otherwise continue.

## Observation arrays and exact fields

The observation input uses YokaiObservation and the fields listed here:

| Field | Shape | Dtype |
|---|---|---|
| node_features | [13, 13] | int8 |
| adjacency_matrix | [13, 13] | uint8 |
| positions | [9, 2] | int32 |
| legal_actions | [1068] | bool |

There are 13 graph nodes: rows 0..8 are stable card IDs, and rows
9..12 are hint IDs 0..3. Rows retain their node identities after movement. A node carries 13 features, with direct small-integer and binary encodings and unnormalized coordinates and IDs.

| Feature columns | Card row | Hint row |
|---|---|---|
| [0:3] | The privately visible true color is one-hot, with all zero for unknown/masked colors. | Always zero. |
| [3:6] | Always contains zero; attaching a hint does not change this. | Multi-hot encoding of the hint's revealed set; before revelation, all zero. |
| [6:8] | Current (x,y) position. | Current (x,y) position, given by (h,-1) when unplaced or the target card's position when placed. |
| [8] | Set to 1 for a card locked by a hint, or 0 otherwise. | 1 if placed, else 0. |
| [9] | A public indication of observation by the other seat; the memory rules below determine its timing. | Always zero. |
| [10] | Use 1 if the seat receiving this observation is active, otherwise 0; all rows repeat it. | Same active-seat flag. |
| [11] | A repeated per-row value for the current substep 0..3. | Same substep. |
| [12] | Stable card ID 0..8. | Hint h has the fixed negative ID value -(h+1). |

An all-zero card-color vector encodes an unknown or masked color rather than an additional color. Only hint nodes carry hint color sets, without copying them to card-row hint-color columns. Off-board coordinates of unplaced hints are illegal destinations for cards.
Card positions alone populate the separate [9,2] positions array, duplicating card-node coordinates rather than including hint rows.

The binary adjacency matrix has shape [13,13] and is symmetric. Every node has a self-edge; separate card nodes connect exactly when they occupy orthogonally neighboring cells. Among hint nodes, only placed hints have non-self edges, linking it to its target card in both directions. No other hint-card edges exist, and different hint nodes never connect. Color is irrelevant to adjacency, while a hint edge does not create a connection between cards.

## Visibility and memory: perfect

Your own-color features do not include the other player's private color observations. Perfect memory instead exposes every card you personally observed earlier in this episode during your active turns. Stable card IDs preserve identity through movement or later locking. While inactive, your entire own-color feature block is zero, including known colors; those features reappear when you are active again.

The returned substep-1 observation exposes the first inspected card. Once the second is inspected, both are visible at substep 2 and throughout substep 3, even following a move. Reveal flags for the current turn reset at its end. A move provides no color revelation for an unobserved card. Each seat retains visibility of positions, locks, revealed hint sets, placements, substep, and active-seat flags; this includes the inactive player.

Hint colors are public once revealed and remain so after placement. The partner-seen column tells which card IDs the other player observed, not what their colors were. Perfect memory keeps this indicator over the episode; otherwise it reports observations added during the current turn and clears at every full-turn boundary. For a known locked card, its previously observed color still obeys the configured memory rule. Locking is not disclosure of that color to a partner.

No recommendation about observation choices, move locations, hint selection, or ending is encoded in the observation. It also has no explicit last-action field, episode counter, hidden-color belief vector, or simulator state. Every seat keeps its policy memory and PRNG private.

## Reset distribution and hint identities

Each reset shuffles colors from the fixed multiset across stable card IDs. Card i
starts at (floor(i/3)+3, (i mod 3)+3), producing a centered
3×3 block. At the outset, cards are unlocked and unobserved, while hints remain unrevealed and unplaced at (h,-1). The initial active player is 0 and the initial substep is 0. Adjacency reflects these positions, and neither player sees true card colors or hint color sets.

The single-color hint is 0, with its color chosen uniformly among the three colors. Each of hints 1, 2, 3 has two colors; together they are the three distinct unordered color pairs in random order.
The sampled set assigned to a hint ID never changes within an episode. Each hint ID's size category is set by the reset rule and its actual colors are hidden before reveal. Sampling ignores compatibility with any card chosen later for placement. Even if every color is connected at reset, observations contain no special flag for that fact.

## Terminal scoring and episode boundaries

Scoring uses upstream_sparse. No reward is given on nonterminal steps; if each color is connected at termination, the shared reward is

5*U + 2*R + C - I,

where U is the unrevealed-hint count; R is the revealed-but-unplaced count; C and I count correct and incorrect placements respectively. This partitions all 4 hints. A hint is correctly placed if and only if its card's color occurs in its revealed set. Enough wrong placements therefore allow a negative score despite success. A board with any unclustered color receives 0 at termination, whether hints are correct or the explicit end action was used.

The configured minimum and maximum define the range -4..20. Ending at once when every color forms a connected group returns 20. If each color forms a connected group with every hint placed, the score is C-I. With 4 hints all incorrect and each color forming a connected group, the score is -4, showing why sparse scoring has a negative lower bound. A board on which any color's cards do not form a connected group yields the configured failure reward. There is no extra reward for a cluster and no shaping between terminal outcomes.

Only one reveal and one placement are possible for each hint, with substep 3 required to choose one of the two. The limit before all hints are placed is thus 8 full turns (32 substeps), with earlier explicit termination possible at substep 0. The evaluation safety bound is 32 substeps. Evaluation fails for an illegal active action or an episode that has not ended within the cap. On termination the adapter supplies a fresh reset observation, followed by a policy call whose reset flag clears private memory. Rewards and hidden state are never passed to policies; episode returns and metrics are recorded separately.


Consult specification.json for resolved flags, numeric dimensions, and array schemas.
