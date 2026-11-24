# Two-player 9-card Yōkai: environment specification

## Goal, board and objects

The players cooperate to rearrange 9 cards so that all cards of each color
form one orthogonally connected group. There are 3 colors (IDs 0..2),
with 3 cards of each color. Connections run between horizontal or vertical neighbors, not diagonal ones. A color is connected only when its cards are linked through cards of that same color. Cards of different colors can touch, with no required empty gap between them. Neither rows, columns, nor another fixed shape are required. After every legal move, orthogonal adjacency must keep the entire board connected without regard to card colors.

Cards occupy distinct cells of a 9×9 finite grid. Coordinates (x,y) use
x for the first array coordinate and y for the second; each is in 0..8.
There is no wraparound. All card positions, stable card IDs and locks are
public. Cards start face down; a private observation reveals a card's color only to its observer. Walls, a draw pile, replacement cards, changing colors, and removed cards are absent.

There are also 4 hint tiles. Each shows a set of colors when revealed. A tile
can later be placed on a card, locking it for the rest of the episode. Hints
are separate objects from cards and do not participate in color connectivity.
A card can legally receive a hint even when its true color is not in the hint's set. That placement counts as an incorrect hint for scoring, yet remains legal and does not expose the hidden color. No supplied convention assigns meaning to a player's hint, observation, or movement choice.

## Action space and turn sequence

Return a scalar integer in 0..1067 on each call; obs.legal_actions provides a boolean mask of length 1068. Player 0 moves first. A full turn keeps one player active for four ordered substeps: 0, 1, 2, 3. Completing substep 3 hands activity to the other player at substep 0. Both policies are invoked with observations at each substep. Only no-op 0 is legal for the inactive seat, and the adapter enforces that no-op. Passing is unavailable to the active seat. A true entry in the action mask denotes compliance with public rules, not an improvement to hidden-color placement.

| Substep | Action and ID formula | Legal choices and effect |
|---|---|---|
| 0 | Observe card i: 1+i; or end game: 1067 | Either inspect any unlocked card for a private color reveal this turn, or end immediately and score the board as it stands. |
| 1 | Observe card i: 1+i | Inspect an unlocked card other than the one already seen this turn. Ending the game is unavailable at this substep. |
| 2 | Move card i to (x,y): 10 + i*81 + x*9 + y | Relocate any unlocked card to a different empty cell within the grid, leaving all 9 cards connected. Either observed or unobserved cards may be moved. |
| 3 | Reveal hint h: 739+h; or place hint h at (x,y): 743 + h*81 + x*9 + y | Either expose the color set of an unrevealed hint publicly, or place a revealed, not-yet-placed hint on the unlocked card at (x,y). |

The action intervals are: no-op [0:1], observe [1:10], move
[10:739], reveal-hint [739:743], place-hint [743:1067],
and end-game [1067:1068]. IDs i and h start at zero. A place action specifies a grid cell rather than a card ID, and an unlocked card must occupy that cell.

Movement relocates a card rather than following a path, so checks concern only the destination and the resulting connectivity of the whole board. Distance costs nothing, and the remaining cards need not stay connected during an imagined intermediate removal. The card keeps its ID, color, and observation history for both players. A locked card cannot be moved or observed again. Swapping, remaining in the same cell, and moving a hint by itself are unavailable.

A hint is revealed and placed through distinct actions on different turns. Reveal is irreversible, and a placed hint cannot be relocated, replaced, or taken away. Placing a hint locks its card, preventing a second hint on that card. Substep 3 ends after one hint is revealed or placed. Placing all 4 hints terminates play immediately. Otherwise, even a board with every color group connected continues: connectivity by itself does not trigger termination.

## Observation arrays and exact fields

The policy receives YokaiObservation with the following fields:

| Field | Shape | Dtype |
|---|---|---|
| node_features | [13, 13] | int8 |
| adjacency_matrix | [13, 13] | uint8 |
| positions | [9, 2] | int32 |
| legal_actions | [1068] | bool |

There are 13 graph nodes: rows 0..8 are stable card IDs, and rows
9..12 are hint IDs 0..3. Moving an object does not change its node-row index. Each graph node has 13 features, storing small integers and binary flags directly. Coordinates and IDs use their unnormalized values.

| Feature columns | Card row | Hint row |
|---|---|---|
| [0:3] | One-hot true color visible privately; all zero when unknown/masked. | Always zero. |
| [3:6] | Zero in every case, even when a hint is placed on the card. | A multi-hot vector for the revealed hint's colors; all zero until reveal. |
| [6:8] | Current (x,y) position. | Current (x,y) coordinates: (h,-1) until placement, then the target card's coordinates. |
| [8] | 1 when a hint locks the card; otherwise 0. | 1 if placed, else 0. |
| [9] | Public flag recording the other seat's observation of this card; retention timing is specified below. | Always zero. |
| [10] | 1 when the observing seat is active and 0 otherwise, copied to every row. | Same active-seat flag. |
| [11] | The current substep in 0..3, copied across rows. | Same substep. |
| [12] | Stable card ID 0..8. | Stable negative ID: the value for hint h is -(h+1). |

An entirely zero card-color vector denotes an unknown or masked color, not an additional color. Read hint color sets from hint nodes, not from the card row's hint-color columns, which do not receive copies. The off-board coordinate of an unplaced hint cannot be used as a card destination.
Only card locations appear in the separate positions array of shape [9,2]. These entries duplicate the card-node coordinates and omit hint rows.

The adjacency matrix is binary and symmetric with shape [13,13], and every node has a self-loop. Distinct card nodes share an edge if and only if their cells are orthogonal neighbors one grid step apart. Placement adds edges in both directions between a hint and its target card. An unplaced hint connects only to itself. No other hint-card links or links between different hints exist. Adjacency is independent of color, and hint edges never link two cards.

## Visibility and memory: perfect

When perfect memory is enabled and your seat is active, own-color features show every card you personally observed earlier in the episode, even if moved or later locked. Colors belong to stable card IDs. All own-color features, including previously observed colors, become zero during inactivity and reappear when your seat becomes active again. Your own-color features never incorporate the other player's private color observations.

The observation returned at substep-1 reveals the first card inspected. After the second inspection, both cards are visible at substep 2 and stay visible through substep 3, including after movement of either card. Current-turn reveal flags clear when the turn ends. Moving a previously unobserved card does not expose its color. Both players, including the inactive seat, can always see positions, locks, revealed hint sets, placements, the substep, and active-seat flags.

Under perfect memory, the partner-seen column indicates whether the partner has observed each card at any time this episode. Otherwise it reports observations added during the current turn and clears at each full-turn boundary. It reveals observed IDs, not the colors seen by the partner. Hint colors are public from reveal onward, including after placement. Previously observed colors of known locked cards obey the configured memory rule; locking does not tell the partner that color.

Observations include no explicit last-action field, episode counter, hidden-color belief vector, or simulator state. Each seat has its own private policy memory and PRNG. No observation recommends a card to observe, a move destination, a hint, or an ending time.

## Reset distribution and hint identities

Reset randomly assigns the fixed card-color multiset to stable IDs. Card i
starts at (floor(i/3)+3, (i mod 3)+3), producing a centered
3×3 block. Cards begin unlocked and not yet observed; hints start unrevealed and unplaced at (h,-1). Player 0 is initially active at substep 0. These positions determine the initial adjacency. Neither seat initially sees any true card color or hint color set.

Hint 0 contains one uniformly sampled color from the three available colors. Hints 1, 2, 3 contain two colors each and collectively use the three distinct unordered pairs in randomized order.
Each hint keeps its sampled set under the same ID throughout the episode. The reset rule fixes each hint ID's size category, but its sampled colors remain hidden until revelation. Sampling is independent of any card later selected for placement. All colors can already be connected at reset without a separate flag announcing that condition.

## Terminal scoring and episode boundaries

The scoring mode is upstream_sparse, with zero reward before termination. If every color is connected when play terminates, both players receive the shared reward

5*U + 2*R + C - I,

where U counts unrevealed hints, R counts revealed hints not yet placed, C counts correct placements, and I counts incorrect placements. The 4 hints belong to exactly these categories. Placement is correct if and only if the card's color is in the revealed hint set. Thus even success can score negatively when sufficiently many placements are wrong. If even one color is not clustered, termination gives 0 regardless of correct hints or use of the explicit end action.

Returns are bounded by -4..20 in this configuration. Ending immediately when each color's cards form a connected group yields 20. If each color forms a connected group and all hints are placed, the score is C-I; all 4 hints being wrong yields -4, hence the negative lower bound despite sparse rewards. The configured failure reward applies if any color's cards fail to form a connected group. Forming a cluster earns no separate bonus, and there are no intermediate shaping rewards.

A hint allows one reveal and one placement, and substep 3 must perform one of these actions. Consequently every hint is placed within 8 full turns (32 substeps), unless play is explicitly ended earlier at substep 0. Evaluation uses a safety cap of 32 substeps. Evaluation is invalid if the active action is illegal or the episode exceeds the termination cap. After termination, the adapter supplies a newly reset observation; the next call's reset flag clears private policy memory. Metrics and returns are recorded separately, with neither rewards nor hidden state given to policies.


See specification.json for numeric dimensions, resolved flags, and array schemas.
