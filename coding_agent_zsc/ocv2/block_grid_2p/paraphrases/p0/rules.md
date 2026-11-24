# Overcooked V2: block_grid_2p

## Goal and episode

Two players cooperate to prepare and deliver dishes matching the currently requested
recipe. A dish is made by putting raw ingredients into a pot, letting them cook,
collecting the cooked contents with an empty plate, and interacting with a delivery
station while holding the plated dish. Recipe matching uses ingredient counts;
ingredient insertion order does not matter. A mixed or otherwise incorrect dish
can still cook, but is not a correct delivery.

Both players act once per environment timestep, simultaneously, and receive the
same team reward. The episode lasts at most **400 joint timesteps**.
The task score is the sum of team rewards over the episode, counted once rather
than once per player. Delivery does not end the episode. The players can complete multiple dishes until the time limit. The policy-search objective (SP or XP) is described
in `../task_spec.md`; it does not change these game mechanics.

## Board, objects, and access

The fixed board is **19 rows × 19 columns**. Coordinates below are
zero-based **(row, column)**; rows increase downward and columns rightward.
Only spaces and `A` cells are traversable floor. `A` denotes a layout spawn marker,
not an object remaining in play. Every other listed object blocks movement.

```text
WWWWWWWWWRWWWWWWWWW
W                 W
W W0W W1W W1W W0W W
W WWW WWW WWW WWW W
W WPW W2W W2W WPW W
W                 W
W WBW WBW WBW WBW W
W WWW WWW WWW WWW W
W WWW WLW WLW WWW W
W    A       A    W
W WWW WLW WLW WWW W
W WWW WWW WWW WWW W
W WBW WBW WBW WBW W
W                 W
W WPW W2W W2W WPW W
W WWW WWW WWW WWW W
W W0W W1W W1W W0W W
W                 W
WWWWWWWWWXWWWWWWWWW
```

| Object | Global (row, column) locations |
| --- | --- |
| Wall / counter (`W`) | (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (1, 0), (1, 18), (2, 0), (2, 2), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12), (2, 14), (2, 16), (2, 18), (3, 0), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8), (3, 10), (3, 11), (3, 12), (3, 14), (3, 15), (3, 16), (3, 18), (4, 0), (4, 2), (4, 4), (4, 6), (4, 8), (4, 10), (4, 12), (4, 14), (4, 16), (4, 18), (5, 0), (5, 18), (6, 0), (6, 2), (6, 4), (6, 6), (6, 8), (6, 10), (6, 12), (6, 14), (6, 16), (6, 18), (7, 0), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 10), (7, 11), (7, 12), (7, 14), (7, 15), (7, 16), (7, 18), (8, 0), (8, 2), (8, 3), (8, 4), (8, 6), (8, 8), (8, 10), (8, 12), (8, 14), (8, 15), (8, 16), (8, 18), (9, 0), (9, 18), (10, 0), (10, 2), (10, 3), (10, 4), (10, 6), (10, 8), (10, 10), (10, 12), (10, 14), (10, 15), (10, 16), (10, 18), (11, 0), (11, 2), (11, 3), (11, 4), (11, 6), (11, 7), (11, 8), (11, 10), (11, 11), (11, 12), (11, 14), (11, 15), (11, 16), (11, 18), (12, 0), (12, 2), (12, 4), (12, 6), (12, 8), (12, 10), (12, 12), (12, 14), (12, 16), (12, 18), (13, 0), (13, 18), (14, 0), (14, 2), (14, 4), (14, 6), (14, 8), (14, 10), (14, 12), (14, 14), (14, 16), (14, 18), (15, 0), (15, 2), (15, 3), (15, 4), (15, 6), (15, 7), (15, 8), (15, 10), (15, 11), (15, 12), (15, 14), (15, 15), (15, 16), (15, 18), (16, 0), (16, 2), (16, 4), (16, 6), (16, 8), (16, 10), (16, 12), (16, 14), (16, 16), (16, 18), (17, 0), (17, 18), (18, 0), (18, 1), (18, 2), (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (18, 10), (18, 11), (18, 12), (18, 13), (18, 14), (18, 15), (18, 16), (18, 17), (18, 18) |
| Always-on recipe indicator (`R`) | (0, 9) |
| Ingredient supply 0 | (2, 3), (2, 15), (16, 3), (16, 15) |
| Ingredient supply 1 | (2, 7), (2, 11), (16, 7), (16, 11) |
| Pot (`P`) | (4, 3), (4, 15), (14, 3), (14, 15) |
| Ingredient supply 2 | (4, 7), (4, 11), (14, 7), (14, 11) |
| Plate supply (`B`) | (6, 3), (6, 7), (6, 11), (6, 15), (12, 3), (12, 7), (12, 11), (12, 15) |
| Button recipe indicator (`L`) | (8, 7), (8, 11), (10, 7), (10, 11) |
| Delivery station (`X`) | (18, 9) |

| Seat | Layout start (row, column) | Reachable floor / reset sampling region |
| --- | --- | --- |
| agent_0 | (9, 5) | (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (2, 1), (2, 5), (2, 9), (2, 13), (2, 17), (3, 1), (3, 5), (3, 9), (3, 13), (3, 17), (4, 1), (4, 5), (4, 9), (4, 13), (4, 17), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 1), (6, 5), (6, 9), (6, 13), (6, 17), (7, 1), (7, 5), (7, 9), (7, 13), (7, 17), (8, 1), (8, 5), (8, 9), (8, 13), (8, 17), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (10, 1), (10, 5), (10, 9), (10, 13), (10, 17), (11, 1), (11, 5), (11, 9), (11, 13), (11, 17), (12, 1), (12, 5), (12, 9), (12, 13), (12, 17), (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (13, 16), (13, 17), (14, 1), (14, 5), (14, 9), (14, 13), (14, 17), (15, 1), (15, 5), (15, 9), (15, 13), (15, 17), (16, 1), (16, 5), (16, 9), (16, 13), (16, 17), (17, 1), (17, 2), (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (17, 15), (17, 16), (17, 17) |
| agent_1 | (9, 13) | (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (2, 1), (2, 5), (2, 9), (2, 13), (2, 17), (3, 1), (3, 5), (3, 9), (3, 13), (3, 17), (4, 1), (4, 5), (4, 9), (4, 13), (4, 17), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 1), (6, 5), (6, 9), (6, 13), (6, 17), (7, 1), (7, 5), (7, 9), (7, 13), (7, 17), (8, 1), (8, 5), (8, 9), (8, 13), (8, 17), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (10, 1), (10, 5), (10, 9), (10, 13), (10, 17), (11, 1), (11, 5), (11, 9), (11, 13), (11, 17), (12, 1), (12, 5), (12, 9), (12, 13), (12, 17), (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (13, 16), (13, 17), (14, 1), (14, 5), (14, 9), (14, 13), (14, 17), (15, 1), (15, 5), (15, 9), (15, 13), (15, 17), (16, 1), (16, 5), (16, 9), (16, 13), (16, 17), (17, 1), (17, 2), (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (17, 15), (17, 16), (17, 17) |

The layout is information available before play. Global player coordinates are not
separately supplied to the policy; runtime inputs are described below.

## Recipes and supplies

There are **3 ingredient types** and the following possible target recipes.
A list such as `[0, 0, 0]` means three units of ingredient 0, not three distinct types.
The environment samples uniformly from the listed recipe entries on reset.

| Ingredient IDs in dish | Counts by ingredient ID | Sampling probability |
| --- | --- | --- |
| [0, 0, 1] | [2, 1, 0] | 1/4 per listed entry |
| [1, 1, 2] | [0, 2, 1] | 1/4 per listed entry |
| [0, 2, 2] | [1, 0, 2] | 1/4 per listed entry |
| [0, 1, 2] | [1, 1, 1] | 1/4 per listed entry |

After any correct delivery, a fresh recipe is sampled uniformly for the next observation; it can equal the old one. Wrong deliveries do not resample it. The new recipe is used by both the always-on indicator and any
button that is still active. An active button displays the current recipe, not
a frozen copy of the recipe at activation. Existing pot contents and held dishes
are not changed when the target recipe changes.

Ingredient IDs 0 through 2 have the same meaning for both players and remain fixed across the episode. No per-player ingredient permutation is enabled.

## Action space and simultaneous steps

Each policy returns one scalar integer in **0–5**.
All six IDs are valid choices in every state; there is no separate legal-action
mask. An ineffective interaction or blocked movement still consumes a timestep.

| Action ID | Action | Effect |
| --- | --- | --- |
| 0 | RIGHT | Face right; attempt to move one column right. |
| 1 | DOWN | Face down; attempt to move one row down. |
| 2 | LEFT | Face left; attempt to move one column left. |
| 3 | UP | Face up; attempt to move one row up. |
| 4 | STAY | Keep position, facing, and inventory; time still advances. |
| 5 | INTERACT | Use the immediately adjacent cell in the current facing direction; do not move or turn. |

Facing direction IDs in the observation are **UP=0, DOWN=1, RIGHT=2, LEFT=3**. These IDs are not
the movement-action IDs. A movement action changes facing even if a wall, object,
map boundary, or collision prevents translation. STAY and INTERACT retain facing.
Walking into an object does not interact with it; interaction is a separate step.

The transition order is:

1. Attempt both movements. Only floor destinations are allowed; movement cannot leave the board.
2. Resolve occupied-cell conflicts. Players cannot end on the same cell or directly swap cells. Conflicting moves revert position, while the attempted facing change remains. Moving into a cell another player successfully vacates is possible if it is not a swap.
3. Process INTERACT actions in seat order (agent_0, then agent_1), so a later interaction sees earlier changes to the grid. Both deliveries are checked against the target recipe from the beginning of the step.
4. Decrement pot and button timers. A pot becomes cooked when its timer reaches zero.
5. Resample the recipe if a correct delivery requires it, advance the episode clock, and emit the resulting local observations and shared reward.

## Inventory and every object interaction

A player has one inventory slot: empty hands, one raw ingredient, one empty plate,
or one plated cooked dish. Supplies are inexhaustible. Objects cannot be picked
up from a distance, and items cannot be passed directly to the other player.

| Facing object | Required inventory/state | Result of INTERACT |
| --- | --- | --- |
| Ingredient supply (`0`, `1`, etc.) | Empty hands | Take one unit of that ingredient. Supply remains. |
| Plate supply (`B`) | Empty hands | Take one empty plate. Supply remains. |
| Empty wall/counter (`W`) | Holding any item | Put the entire held item on the counter; hands become empty. |
| Occupied wall/counter (`W`) | Empty hands | Pick up its entire item; counter becomes empty. |
| Pot (`P`) | Holding one raw ingredient; pot idle, uncooked, and below capacity | Add that ingredient; hands become empty. |
| Cooked pot (`P`) | Holding an empty plate | Transfer all cooked contents onto the plate and empty the pot. |
| Delivery station (`X`) | Holding a cooked dish | Deliver and consume the entire dish including its plate, whether correct or incorrect; hands become empty. |
| Always-on recipe indicator (`R`) | Any | No interaction is needed or has an effect. Its recipe channels are visible whenever the cell is in view. |
| Button recipe indicator (`L`) | Empty hands and no item on that cell | Activate or refresh the recipe display; apply the configured cost. |
| Floor or any unmet condition above | Any | No item transfer, delivery, or activation. Time still advances. |

There is no counter item swapping: occupied hands cannot pick up an item from an
occupied counter. There is no trash action, chopping action, or direct pickup of uncooked
pot contents. A plate cannot scoop a cooking/uncooked pot. Ingredients cannot be
added to cooking or cooked contents. Finished food does not burn or spoil.

## Pot capacity, cooking, and exact timer timing

Each pot holds at most **3 raw ingredients**. Cooking starts automatically when an interaction brings the pot's ingredient count to 3. There is no separate start-cooking action.
The configured cooking duration is **20 timesteps**. The timer is set during
the initiating interaction, then decremented at the end of that same step.
Consequently the first returned observation after starting cooking shows
**19**, followed by one decrement per step; after **19 additional steps**
the timer is zero and the cooked flag is set. Pickup can succeed on the following
interaction, not on the transition that first finishes cooking, because
interactions precede timer updates.

Pot states are distinguished by both contents and timer: empty means no contents,
no cooked flag and timer 0; partially filled idle means raw counts and timer 0;
cooking means timer > 0; ready means cooked flag 1 and timer 0. Timer 0 alone does
not mean a dish is ready. The timer is a raw integer, not a fraction of cooking
progress. The pot keeps its cooked contents until collected. Collection clears
the contents, so the same pot can be used again.

## Recipe button duration and cost

The button's configured active duration is **10 timesteps**. Every successful activation immediately costs the team 5 reward, including reactivation while already active.
Activation sets its internal timer to 10; the same step decrements it to
9. In the absence of reactivation, its recipe is therefore visible
in the first **9 post-step observations** after activation (if the
button lies inside the view), then disappears when its timer becomes zero.
Reactivation resets the countdown. Holding an ingredient, plate, or dish prevents
activation. The static button channel remains present even while inactive.

The button countdown is **not an observation channel**. Visibility is represented
by nonzero recipe counts at its cell while active. Only pots expose their numeric
countdown. The always-on recipe indicator does not expire and costs nothing to view.

## Shared rewards and termination

| Event | Team reward |
| --- | --- |
| Cooked dish matches the current recipe exactly | +20 |
| Cooked dish has the wrong ingredient counts | -20 |
| Button activation | −5 each time |
| Movement, waiting, ordinary pickups/drops, adding ingredients, cooking, or ineffective actions | 0 |

Same-step events add together, including an activation and delivery on one step.
For example, a correct delivery with an immediate button cost contributes
20 − 5 = 15 when those two events coincide. There is no
ordinary time penalty. The environment may report auxiliary shaping diagnostics,
but they are not added to the cooperative return used for policy scoring.

After the 400-th transition the episode is terminal; reward from that
transition still counts. Delivery does not end the episode. The players can complete multiple dishes until the time limit. Unfinished food and held items have no terminal
bonus. The environment does not terminate merely because an incorrect dish was
prepared or a button cost was incurred.

## Observation space: exact tensor and channels

Each seat receives its own **[5, 5, 40] int32 JAX array**,
indexed `obs[row, column, channel]`. The view radius is 2, giving a 5 × 5 square centered on self at local cell (2, 2). Local cell (i, j) corresponds to global (self_row + i − 2, self_column + j − 2). The crop is not rotated with facing. Walls and objects do not occlude cells inside this square. Cells outside the map are zero padded; zero padding is not evidence of traversable floor. Cells outside the view are absent, including any partner or recipe indicator outside the crop.
The encoding is observer-relative: the first player block always describes self,
the second the partner, independent of the seat number. Invisible partner features
are all zero. Facing is a one-hot direction at the player's cell, not a rotation
of the grid. Inventory features also occur only at that player's cell.

The table lists every channel individually; there are no omitted or hidden
channels. Slices in `specification.json` use half-open intervals `[start, stop)`.

| Channel | Meaning | Values / location |
| --- | --- | --- |
| 0 | Self position | 1 at that player's cell; 0 elsewhere. |
| 1 | Self facing UP | One-hot at that player's cell. |
| 2 | Self facing DOWN | One-hot at that player's cell. |
| 3 | Self facing RIGHT | One-hot at that player's cell. |
| 4 | Self facing LEFT | One-hot at that player's cell. |
| 5 | Self held plate flag | At that player's cell only; 0 elsewhere. |
| 6 | Self held cooked flag | At that player's cell only; 0 elsewhere. |
| 7 | Self held ingredient 0 count | At that player's cell only; 0 elsewhere. |
| 8 | Self held ingredient 1 count | At that player's cell only; 0 elsewhere. |
| 9 | Self held ingredient 2 count | At that player's cell only; 0 elsewhere. |
| 10 | Partner position | 1 at that player's cell; 0 elsewhere. |
| 11 | Partner facing UP | One-hot at that player's cell. |
| 12 | Partner facing DOWN | One-hot at that player's cell. |
| 13 | Partner facing RIGHT | One-hot at that player's cell. |
| 14 | Partner facing LEFT | One-hot at that player's cell. |
| 15 | Partner held plate flag | At that player's cell only; 0 elsewhere. |
| 16 | Partner held cooked flag | At that player's cell only; 0 elsewhere. |
| 17 | Partner held ingredient 0 count | At that player's cell only; 0 elsewhere. |
| 18 | Partner held ingredient 1 count | At that player's cell only; 0 elsewhere. |
| 19 | Partner held ingredient 2 count | At that player's cell only; 0 elsewhere. |
| 20 | wall | 1 at matching static objects; 0 elsewhere. |
| 21 | goal | 1 at matching static objects; 0 elsewhere. |
| 22 | pot | 1 at matching static objects; 0 elsewhere. |
| 23 | recipe indicator | 1 at matching static objects; 0 elsewhere. |
| 24 | button indicator | 1 at matching static objects; 0 elsewhere. |
| 25 | plate pile | 1 at matching static objects; 0 elsewhere. |
| 26 | Ingredient 0 supply | 1 at this ingredient's inexhaustible supply cells. |
| 27 | Ingredient 1 supply | 1 at this ingredient's inexhaustible supply cells. |
| 28 | Ingredient 2 supply | 1 at this ingredient's inexhaustible supply cells. |
| 29 | Item on grid cell: plate flag | Counters and pots; excludes held inventory. |
| 30 | Item on grid cell: cooked flag | Counters and pots; excludes held inventory. |
| 31 | Item on grid cell: ingredient 0 count | Counters and pots; excludes held inventory. |
| 32 | Item on grid cell: ingredient 1 count | Counters and pots; excludes held inventory. |
| 33 | Item on grid cell: ingredient 2 count | Counters and pots; excludes held inventory. |
| 34 | Displayed target recipe: plate flag | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 35 | Displayed target recipe: cooked flag | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 36 | Displayed target recipe: ingredient 0 count | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 37 | Displayed target recipe: ingredient 1 count | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 38 | Displayed target recipe: ingredient 2 count | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 39 | Remaining pot cooking time | Integer countdown at pots; 0 elsewhere. No normalization. |

Static, position, facing, plate and cooked flags are 0/1. Ingredient channels are
counts (0–3 in the native encoding), not one-hot recipe classes. The 5 entries
in each item/recipe block here are plate flag, cooked flag, then counts for each
ingredient ID; the block width in general is 2 + number of ingredient types.
An empty item block is all zero. An empty plate has only its plate flag; a raw
ingredient has one ingredient count; a plated cooked dish has both flags plus
its ingredient counts. A cooked pot has its cooked flag and counts, without a
plate flag. Recipe displays contain ingredient counts only, with both flags zero.

Floor has no dedicated static channel: within the known map it is a cell with
no static-object or ingredient-supply flag. Counter contents and pot contents
appear in the grid-item block; held contents appear only in the player inventory
blocks. Recipe display features are separate from actual items on those cells.
An all-zero recipe block means no recipe is displayed there; it does not denote
an empty target recipe.

There is no explicit global position, global map state, step counter, teammate
memory, previous action, or button countdown in this array. The policy separately
receives its reset flag, private memory and PRNG; it does not receive reward as an
act-function input. The optional successful-delivery channel is disabled, so no delivery-success bit is present.
Remembering past observations and counting elapsed steps are policy-owned work.

## Reset and randomness

Each seat is sampled uniformly within its own listed connected floor region, excluding already occupied cells. It cannot spawn in the other room. Each facing direction is sampled uniformly from the four directions. Position randomization itself also randomizes facing, even without a separate orientation override.
Both hands, all counters, and all pots start empty. All pot and button timers start at zero; buttons are inactive. The episode clock starts at zero and the recipe is sampled as
listed above. After any correct delivery, a fresh recipe is sampled uniformly for the next observation; it can equal the old one. Wrong deliveries do not resample it. Static geometry and supply locations never change.
Randomness comes from the configured reset and recipe sampling (and the policy's
own PRNG); movement, interactions, timers and rewards follow the rules above.

`config.yaml` records the requested configuration. `specification.json` records
the resolved constants, object coordinates, spawn regions and observation schema.
`layout.md` repeats the board and legend. These files describe the selected
environment without providing its executable implementation or policy helpers.


Numeric dimensions, resolved flags and array schemas: specification.json.
