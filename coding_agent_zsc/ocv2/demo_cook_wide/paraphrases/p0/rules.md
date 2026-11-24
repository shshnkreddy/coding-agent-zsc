# Overcooked V2: demo_cook_wide

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

The fixed board is **6 rows × 11 columns**. Coordinates below are
zero-based **(row, column)**; rows increase downward and columns rightward.
Only spaces and `A` cells are traversable floor. `A` denotes a layout spawn marker,
not an object remaining in play. Every other listed object blocks movement.

```text
WWWWBXBWWWW
WWW0 A 1WWW
WWWWWPWWWWW
W    A    W
0  W3R3W  0
W1WWWWWWW1W
```

| Object | Global (row, column) locations |
| --- | --- |
| Wall / counter (`W`) | (0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10), (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 0), (3, 10), (4, 3), (4, 7), (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10) |
| Plate supply (`B`) | (0, 4), (0, 6) |
| Delivery station (`X`) | (0, 5) |
| Ingredient supply 0 | (1, 3), (4, 0), (4, 10) |
| Ingredient supply 1 | (1, 7), (5, 1), (5, 9) |
| Pot (`P`) | (2, 5) |
| Ingredient supply 3 | (4, 4), (4, 6) |
| Always-on recipe indicator (`R`) | (4, 5) |

| Seat | Layout start (row, column) | Reachable floor / reset sampling region |
| --- | --- | --- |
| agent_0 | (1, 5) | (1, 4), (1, 5), (1, 6) |
| agent_1 | (3, 5) | (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 1), (4, 2), (4, 8), (4, 9) |

The layout is information available before play. Global player coordinates are not
separately supplied to the policy; runtime inputs are described below.

## Recipes and supplies

There are **4 ingredient types** and the following possible target recipes.
A list such as `[0, 0, 0]` means three units of ingredient 0, not three distinct types.
The environment samples uniformly from the listed recipe entries on reset.

| Ingredient IDs in dish | Counts by ingredient ID | Sampling probability |
| --- | --- | --- |
| [0, 0, 0] | [3, 0, 0, 0] | 1/2 per listed entry |
| [1, 1, 1] | [0, 3, 0, 0] | 1/2 per listed entry |

After any correct delivery, a fresh recipe is sampled uniformly for the next observation; it can equal the old one. Wrong deliveries do not resample it. The new recipe is used by both the always-on indicator and any
button that is still active. An active button displays the current recipe, not
a frozen copy of the recipe at activation. Existing pot contents and held dishes
are not changed when the target recipe changes.

Ingredient IDs 0 through 3 have the same meaning for both players and remain fixed across the episode. No per-player ingredient permutation is enabled.

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

Each seat receives its own **[5, 5, 45] int32 JAX array**,
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
| 10 | Self held ingredient 3 count | At that player's cell only; 0 elsewhere. |
| 11 | Partner position | 1 at that player's cell; 0 elsewhere. |
| 12 | Partner facing UP | One-hot at that player's cell. |
| 13 | Partner facing DOWN | One-hot at that player's cell. |
| 14 | Partner facing RIGHT | One-hot at that player's cell. |
| 15 | Partner facing LEFT | One-hot at that player's cell. |
| 16 | Partner held plate flag | At that player's cell only; 0 elsewhere. |
| 17 | Partner held cooked flag | At that player's cell only; 0 elsewhere. |
| 18 | Partner held ingredient 0 count | At that player's cell only; 0 elsewhere. |
| 19 | Partner held ingredient 1 count | At that player's cell only; 0 elsewhere. |
| 20 | Partner held ingredient 2 count | At that player's cell only; 0 elsewhere. |
| 21 | Partner held ingredient 3 count | At that player's cell only; 0 elsewhere. |
| 22 | wall | 1 at matching static objects; 0 elsewhere. |
| 23 | goal | 1 at matching static objects; 0 elsewhere. |
| 24 | pot | 1 at matching static objects; 0 elsewhere. |
| 25 | recipe indicator | 1 at matching static objects; 0 elsewhere. |
| 26 | button indicator | 1 at matching static objects; 0 elsewhere. |
| 27 | plate pile | 1 at matching static objects; 0 elsewhere. |
| 28 | Ingredient 0 supply | 1 at this ingredient's inexhaustible supply cells. |
| 29 | Ingredient 1 supply | 1 at this ingredient's inexhaustible supply cells. |
| 30 | Ingredient 2 supply | 1 at this ingredient's inexhaustible supply cells. |
| 31 | Ingredient 3 supply | 1 at this ingredient's inexhaustible supply cells. |
| 32 | Item on grid cell: plate flag | Counters and pots; excludes held inventory. |
| 33 | Item on grid cell: cooked flag | Counters and pots; excludes held inventory. |
| 34 | Item on grid cell: ingredient 0 count | Counters and pots; excludes held inventory. |
| 35 | Item on grid cell: ingredient 1 count | Counters and pots; excludes held inventory. |
| 36 | Item on grid cell: ingredient 2 count | Counters and pots; excludes held inventory. |
| 37 | Item on grid cell: ingredient 3 count | Counters and pots; excludes held inventory. |
| 38 | Displayed target recipe: plate flag | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 39 | Displayed target recipe: cooked flag | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 40 | Displayed target recipe: ingredient 0 count | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 41 | Displayed target recipe: ingredient 1 count | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 42 | Displayed target recipe: ingredient 2 count | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 43 | Displayed target recipe: ingredient 3 count | Only at R and currently active L cells. Recipe plate/cooked flags are zero. |
| 44 | Remaining pot cooking time | Integer countdown at pots; 0 elsewhere. No normalization. |

Static, position, facing, plate and cooked flags are 0/1. Ingredient channels are
counts (0–3 in the native encoding), not one-hot recipe classes. The 6 entries
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
