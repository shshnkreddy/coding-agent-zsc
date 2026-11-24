# Overcooked V2: grounded_coord_simple

## Goal and episode

The cooperative task is to prepare the requested recipe and deliver it. Raw ingredients go into a pot and cook; an empty plate collects the cooked contents, and the player holding the plated dish interacts with the delivery station. The recipe is checked by counting ingredients, regardless of their insertion order. A mixed dish or another incorrect combination can cook without qualifying as a correct delivery.

At each environment timestep both players act simultaneously and obtain an identical team reward. The episode limit is **400 joint timesteps**.
Task score totals the team rewards during the episode, without counting them again for each player. The episode continues after delivery; players may finish multiple dishes before time runs out. The SP or XP policy-search objective is specified in `../task_spec.md`, independently of these unchanged game mechanics.

## Board, objects, and access

Board dimensions are fixed at **5 rows × 8 columns**. Below, **(row, column)** coordinates start at zero; larger rows are downward and larger columns rightward.
Spaces and `A` cells alone can be traversed. A layout spawn is marked by `A`, which does not remain as an in-play object. The other listed objects block movement.

```text
WW2WWWWW
W  WB  0
R ALPA X
W  WB  1
WW2WWWWW
```

| Object | Global (row, column) locations |
| --- | --- |
| Wall / counter (`W`) | (0, 0), (0, 1), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 3), (3, 0), (3, 3), (4, 0), (4, 1), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7) |
| Ingredient supply 2 | (0, 2), (4, 2) |
| Plate supply (`B`) | (1, 4), (3, 4) |
| Ingredient supply 0 | (1, 7) |
| Always-on recipe indicator (`R`) | (2, 0) |
| Button recipe indicator (`L`) | (2, 3) |
| Pot (`P`) | (2, 4) |
| Delivery station (`X`) | (2, 7) |
| Ingredient supply 1 | (3, 7) |

| Seat | Layout start (row, column) | Reachable floor / reset sampling region |
| --- | --- | --- |
| agent_0 | (2, 2) | (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2) |
| agent_1 | (2, 5) | (1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6) |

The layout is supplied in advance of play, but global player coordinates are not a separate policy input. The runtime inputs appear below.

## Recipes and supplies

There are **3 ingredient types** available, with the possible target recipes listed next.
The list `[0, 0, 0]` denotes three units of ingredient 0, not a set of three ingredient types.
At reset, every listed recipe entry is sampled with equal probability.

| Ingredient IDs in dish | Counts by ingredient ID | Sampling probability |
| --- | --- | --- |
| [0, 0, 0] | [3, 0, 0] | 1/2 per listed entry |
| [1, 1, 1] | [0, 3, 0] | 1/2 per listed entry |

A correct delivery causes a new uniform recipe draw for the next observation; drawing the previous recipe again is possible. Wrong deliveries do not trigger recipe resampling. The always-on indicator and still-active buttons reflect the new recipe. Active buttons track the current recipe instead of retaining an activation-time copy. A change of target recipe does not alter dishes being held or ingredients already in pots.

Ingredient IDs 0 through 2 are shared in meaning between the players and fixed within the episode; no per-player ingredient permutation is active.

## Action space and simultaneous steps

Return one scalar integer in **0–5** from each policy. All six IDs are valid in every state without a separate legal-action mask. Failed interactions and blocked movement still take a timestep.

| Action ID | Action | Effect |
| --- | --- | --- |
| 0 | RIGHT | Face toward the right, then attempt a one-column rightward move. |
| 1 | DOWN | Face downward, then attempt to advance one row. |
| 2 | LEFT | Face toward the left, then attempt a one-column leftward move. |
| 3 | UP | Face upward, then attempt to advance one row. |
| 4 | STAY | Leave location, facing, and inventory unchanged; the timestep still elapses. |
| 5 | INTERACT | Act on the cell immediately in front of the player; position and facing stay fixed. |

Facing direction IDs in the observation are **UP=0, DOWN=1, RIGHT=2, LEFT=3**. Do not equate these IDs with movement-action IDs. A move updates facing even if a wall, object, map boundary, or collision stops translation. STAY and INTERACT leave facing unchanged. Object interaction takes a separate step rather than happening when walking into it.

Each transition proceeds in this order:

1. Attempt movement for both players; destinations must be floor and remain on the board.
2. Check occupied-cell conflicts. Ending together or directly swapping cells is forbidden. Conflicting movement restores the old positions, not the attempted facing changes. Entering a cell the other player successfully leaves is allowed unless it forms a swap.
3. Execute INTERACT actions sequentially by seat (agent_0, then agent_1); later interactions see prior grid updates. For both deliveries, use the target recipe from the step's beginning.
4. Count down pot and button timers. Reaching zero sets a pot to cooked.
5. Resample after a correct delivery when required, move the episode clock forward, and emit local observations and the shared reward for the resulting state.

## Inventory and every object interaction

A player's inventory is limited to one slot: empty hands or one raw ingredient, empty plate, or plated cooked dish. There is no depletion of supplies. Neither remote pickup nor direct transfer of items to the partner is possible.

| Facing object | Required inventory/state | Result of INTERACT |
| --- | --- | --- |
| Ingredient supply (`0`, `1`, etc.) | Empty hands | Receive one raw ingredient unit; the supply stays available. |
| Plate supply (`B`) | Empty hands | Receive one empty plate; the supply stays available. |
| Empty wall/counter (`W`) | Holding any item | Deposit the whole inventory item on the counter, leaving hands empty. |
| Occupied wall/counter (`W`) | Empty hands | Collect the whole item from the counter and clear the counter. |
| Pot (`P`) | A raw ingredient in hand and an idle, uncooked pot with space remaining | Place the raw ingredient in the pot, leaving nothing held. |
| Cooked pot (`P`) | Holding an empty plate | Move all cooked contents from the pot to the plate; the pot becomes empty. |
| Delivery station (`X`) | Holding a cooked dish | Deliver the dish together with its plate; both are consumed whether the recipe is right or wrong, and the hands clear. |
| Always-on recipe indicator (`R`) | Any | Its recipe channels appear whenever the cell is visible, without any required or effective interaction. |
| Button recipe indicator (`L`) | Nothing held and no item present on the cell | Enable or refresh the displayed recipe, incurring the configured cost. |
| Floor or any unmet condition above | Any | Advance time without transferring items, delivering, or activating anything. |

Occupied hands cannot collect an item from an occupied counter, so counters do not support item swaps. Trash and chopping actions, and direct pickup of uncooked pot contents, are absent. A cooking/uncooked pot cannot be scooped with a plate. Adding ingredients to cooking or cooked contents is impossible. Finished food is not subject to burning or spoilage.

## Pot capacity, cooking, and exact timer timing

Each pot holds at most **3 raw ingredients**. Cooking begins automatically when an interaction raises the pot's ingredient count to 3, without a separate start-cooking action.
The configured cooking duration is **20 timesteps**. Set the timer in the initiating interaction and decrement it at the end of the same step.
Thus the first returned observation after cooking starts shows a timer value of **19**. The timer then decreases once each step. The timer reaches zero and sets the cooked flag after **19 additional steps**. Pickup first succeeds on the following interaction, rather than the finishing transition, because interaction happens before timer updates.

Pot states are distinguished by both contents and timer: empty means no contents,
no cooked flag and timer 0; partially filled idle means raw counts and timer 0;
cooking means timer > 0; ready means cooked flag 1 and timer 0. Do not interpret timer 0 alone as a ready dish. Its value is a raw integer, not a cooking-progress fraction. Cooked contents remain until pickup, which clears the pot and allows reuse.

## Recipe button duration and cost

The button's configured active duration is **10 timesteps**. The team immediately pays 5 reward for each successful activation, including one performed while already active.
Activation sets its internal timer to 10; the same step decrements it to
9. In the absence of reactivation, its recipe is therefore visible
in the first **9 post-step observations** after activation (if the
button lies inside the view), then disappears when its timer becomes zero.
The countdown is reset by reactivation. A held ingredient, plate, or dish prevents activation, while the static button channel remains present when inactive.

Nonzero recipe counts at an active button's cell represent its visible display. Its countdown is **not an observation channel**; only pot countdowns are numerically exposed. The always-on indicator has no expiry and can be viewed at no cost.

## Shared rewards and termination

| Event | Team reward |
| --- | --- |
| Cooked dish matches the current recipe exactly | +20 |
| Cooked dish has the wrong ingredient counts | -20 |
| Button activation | −5 each time |
| Movement, waiting, ordinary pickups/drops, adding ingredients, cooking, or ineffective actions | 0 |

Same-step events add together, including an activation and delivery on one step.
For example, a correct delivery with an immediate button cost contributes
20 − 5 = 15 when those two events coincide. No ordinary time penalty is applied, and any auxiliary shaping diagnostics the environment reports are excluded from the cooperative return used for policy scoring.

After the 400-th transition the episode is terminal; reward from that
transition still counts. The episode continues after delivery; players may finish multiple dishes before time runs out. Items held at termination and food still unfinished yield no terminal bonus. Neither preparation of an incorrect dish nor a button charge by itself ends the environment.

## Observation space: exact tensor and channels

Each seat receives its own **[5, 5, 40] int32 JAX array**,
indexed `obs[row, column, channel]`. The view radius is 2, giving a 5 × 5 square centered on self at local cell (2, 2). Local cell (i, j) corresponds to global (self_row + i − 2, self_column + j − 2). The crop keeps its orientation regardless of facing. Walls and objects cannot occlude cells in this square. Off-map cells receive zero padding, not evidence of walkable floor. Out-of-view cells, including any partner or recipe indicator outside the crop, are absent.
The first player block always encodes self and the second partner, so the representation is observer-relative rather than seat-number-dependent. Invisible partner features are zero. A one-hot direction at the player's cell represents facing, not a rotated grid; inventory features are confined to that cell as well.

The table enumerates all channels, without hidden or omitted ones. Slices in `specification.json` use half-open intervals `[start, stop)`.

| Channel | Meaning | Values / location |
| --- | --- | --- |
| 0 | Self position | That player's cell contains 1; all other cells contain 0. |
| 1 | Self facing UP | At the player's cell, represented one-hot. |
| 2 | Self facing DOWN | At the player's cell, represented one-hot. |
| 3 | Self facing RIGHT | At the player's cell, represented one-hot. |
| 4 | Self facing LEFT | At the player's cell, represented one-hot. |
| 5 | Self held plate flag | Stored at that player's cell; all other cells have 0. |
| 6 | Self held cooked flag | Stored at that player's cell; all other cells have 0. |
| 7 | Self held ingredient 0 count | Stored at that player's cell; all other cells have 0. |
| 8 | Self held ingredient 1 count | Stored at that player's cell; all other cells have 0. |
| 9 | Self held ingredient 2 count | Stored at that player's cell; all other cells have 0. |
| 10 | Partner position | That player's cell contains 1; all other cells contain 0. |
| 11 | Partner facing UP | At the player's cell, represented one-hot. |
| 12 | Partner facing DOWN | At the player's cell, represented one-hot. |
| 13 | Partner facing RIGHT | At the player's cell, represented one-hot. |
| 14 | Partner facing LEFT | At the player's cell, represented one-hot. |
| 15 | Partner held plate flag | Stored at that player's cell; all other cells have 0. |
| 16 | Partner held cooked flag | Stored at that player's cell; all other cells have 0. |
| 17 | Partner held ingredient 0 count | Stored at that player's cell; all other cells have 0. |
| 18 | Partner held ingredient 1 count | Stored at that player's cell; all other cells have 0. |
| 19 | Partner held ingredient 2 count | Stored at that player's cell; all other cells have 0. |
| 20 | wall | Matching static-object cells contain 1 and other cells contain 0. |
| 21 | goal | Matching static-object cells contain 1 and other cells contain 0. |
| 22 | pot | Matching static-object cells contain 1 and other cells contain 0. |
| 23 | recipe indicator | Matching static-object cells contain 1 and other cells contain 0. |
| 24 | button indicator | Matching static-object cells contain 1 and other cells contain 0. |
| 25 | plate pile | Matching static-object cells contain 1 and other cells contain 0. |
| 26 | Ingredient 0 supply | Cells with an unlimited supply of this ingredient contain 1. |
| 27 | Ingredient 1 supply | Cells with an unlimited supply of this ingredient contain 1. |
| 28 | Ingredient 2 supply | Cells with an unlimited supply of this ingredient contain 1. |
| 29 | Item on grid cell: plate flag | Counter and pot contents only; held items are excluded. |
| 30 | Item on grid cell: cooked flag | Counter and pot contents only; held items are excluded. |
| 31 | Item on grid cell: ingredient 0 count | Counter and pot contents only; held items are excluded. |
| 32 | Item on grid cell: ingredient 1 count | Counter and pot contents only; held items are excluded. |
| 33 | Item on grid cell: ingredient 2 count | Counter and pot contents only; held items are excluded. |
| 34 | Displayed target recipe: plate flag | Only R and currently active L locations carry these values. The recipe plate/cooked flags are zero. |
| 35 | Displayed target recipe: cooked flag | Only R and currently active L locations carry these values. The recipe plate/cooked flags are zero. |
| 36 | Displayed target recipe: ingredient 0 count | Only R and currently active L locations carry these values. The recipe plate/cooked flags are zero. |
| 37 | Displayed target recipe: ingredient 1 count | Only R and currently active L locations carry these values. The recipe plate/cooked flags are zero. |
| 38 | Displayed target recipe: ingredient 2 count | Only R and currently active L locations carry these values. The recipe plate/cooked flags are zero. |
| 39 | Remaining pot cooking time | Raw integer time remaining at pots; other cells have 0, without normalization. |

The flags for static objects, positions, facing, plates, and cooked status are 0/1. Ingredients are represented by counts (native range 0–3), not by one-hot recipe-class encodings. In each item/recipe block, the 5 entries are ordered as plate flag, cooked flag, and per-ingredient ID counts; the general width is 2 + number of ingredient types.
Empty items are represented by all-zero blocks. An empty plate sets just the plate flag, a raw ingredient contributes one ingredient count, and a cooked plated dish sets both flags together with its counts. A cooked pot sets the cooked flag plus counts, omitting the plate flag. Displayed recipes have only ingredient counts; neither flag is set.

Within the known map, floor is identified by the absence of static-object and ingredient-supply flags rather than by its own static channel. Contents of pots and counters occupy the grid-item block, while held contents occur only in player inventory blocks. Actual cell items and recipe display features are separate. A recipe block of all zeros means nothing is displayed, not that the target recipe is empty.

Explicit global position and global map state are absent, as are a step counter, teammate memory, previous action, and button countdown. The policy separately
receives its reset flag, private memory and PRNG; it does not receive reward as an
act-function input. The optional successful-delivery channel is disabled, so no delivery-success bit is present.
Observation recall and elapsed-step counting belong to the policy.

## Reset and randomness

Each seat starts at a uniformly sampled unoccupied cell in its listed connected floor region, never in the other room. All four facing directions have equal sampling probability. Position randomization includes facing randomization without requiring a separate orientation override.
Hands, counters, and pots all begin empty; every pot and button timer begins at zero and the buttons are inactive. The episode clock starts at zero and the recipe is sampled as
listed above. A correct delivery causes a new uniform recipe draw for the next observation; drawing the previous recipe again is possible. Wrong deliveries do not trigger recipe resampling. The static geometry and supplies' locations are constant. Randomness is due to configured reset and recipe sampling, plus the policy's own PRNG, rather than deviations from the movement, interaction, timer, or reward rules above.

Read `config.yaml` for the requested configuration and `specification.json` for resolved constants, object locations, spawn regions, and observation schema. `layout.md` reproduces the board with its legend. These files specify the selected environment but contain neither executable mechanics nor policy helpers.


Numeric sizes, resolved flags, and array schemas are recorded in specification.json.
