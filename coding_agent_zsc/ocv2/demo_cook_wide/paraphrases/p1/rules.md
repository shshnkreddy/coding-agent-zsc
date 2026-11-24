# Overcooked V2: demo_cook_wide

## Goal and episode

The two players work together to deliver dishes that match the current recipe. Preparing a dish requires placing raw ingredients in a pot, cooking them, taking the cooked contents with an empty plate, and using the delivery station while carrying that plated dish. Matching depends on ingredient counts, not the order in which ingredients entered the pot. Cooking is possible for mixed or otherwise incorrect dishes too, but delivering them does not count as a correct delivery.

Each environment timestep contains simultaneous actions by both players, who share the same team reward. An episode is limited to **400 joint timesteps**.
Sum the episode's team rewards once, not separately for each player, to obtain the task score. A delivery leaves the episode running, allowing further dishes to be completed until the time limit. See `../task_spec.md` for the policy-search objective (SP or XP); these mechanics are unaffected by that objective.

## Board, objects, and access

The board has a fixed size of **6 rows × 11 columns**. Coordinates use zero-based **(row, column)** indexing, increasing downward for rows and rightward for columns.
Traversable floor consists only of spaces and `A` cells. The `A` is a layout spawn marker rather than a persistent object. All other listed objects obstruct movement.

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

Players have the layout information before play begins. The policy does not receive a separate global-player-coordinate input; its runtime inputs are specified below.

## Recipes and supplies

The environment provides **4 ingredient types**. The possible target recipes are listed below.
For example, `[0, 0, 0]` specifies three units of ingredient 0 rather than three different types.
Reset selects a listed recipe entry uniformly.

| Ingredient IDs in dish | Counts by ingredient ID | Sampling probability |
| --- | --- | --- |
| [0, 0, 0] | [3, 0, 0, 0] | 1/2 per listed entry |
| [1, 1, 1] | [0, 3, 0, 0] | 1/2 per listed entry |

Every correct delivery triggers uniform sampling of a fresh recipe for the next observation, possibly the same recipe again. An incorrect delivery does not trigger resampling. Both the always-on indicator and any still-active button use the new recipe. A button that remains active shows the recipe currently requested rather than preserving the recipe from its activation. Changing the target leaves existing pot contents and carried dishes untouched.

Both players use the same meanings for ingredient IDs 0 through 3 throughout the episode. Per-player ingredient permutations are disabled.

## Action space and simultaneous steps

A policy outputs a scalar integer in **0–5**. Every state permits all six IDs and supplies no separate legal-action mask. Even an ineffective interaction or blocked move uses a timestep.

| Action ID | Action | Effect |
| --- | --- | --- |
| 0 | RIGHT | Turn to face right and try moving one column right. |
| 1 | DOWN | Turn to face down and try moving one row downward. |
| 2 | LEFT | Turn to face left and try moving one column left. |
| 3 | UP | Turn to face up and try moving one row upward. |
| 4 | STAY | Retain the current position, orientation, and held item while time advances. |
| 5 | INTERACT | Interact with the neighboring cell directly ahead, without moving or changing facing. |

Facing direction IDs in the observation are **UP=0, DOWN=1, RIGHT=2, LEFT=3**. These IDs differ from movement-action IDs. Movement changes facing even when translation is blocked by a wall, object, boundary, or collision. STAY and INTERACT preserve facing. Moving into an object does not interact with it; that requires a separate step.

Transitions follow this sequence:

1. Try both players' movements, permitting only floor destinations within the board.
2. Resolve cell-occupancy conflicts: players may neither finish in one cell nor swap cells directly. Conflicting moves restore positions but keep the attempted facing changes. A player may enter a successfully vacated cell provided the move is not a swap.
3. Apply INTERACT in seat order (agent_0, then agent_1), letting the later interaction observe the earlier grid changes. Check both deliveries against the recipe at the start of the step.
4. Reduce pot and button timers; a pot becomes cooked once its timer reaches zero.
5. If required by a correct delivery, resample the recipe; then advance the episode clock and return the resulting local observations and shared reward.

## Inventory and every object interaction

Each player has a single inventory slot, containing nothing, one raw ingredient, one empty plate, or one plated cooked dish. Supplies never run out. Pickup at a distance and direct handoff to the other player are not available.

| Facing object | Required inventory/state | Result of INTERACT |
| --- | --- | --- |
| Ingredient supply (`0`, `1`, etc.) | Empty hands | Pick up one unit of the ingredient without depleting the supply. |
| Plate supply (`B`) | Empty hands | Pick up an empty plate without depleting the supply. |
| Empty wall/counter (`W`) | Holding any item | Place the complete held item on the counter and clear the player's hands. |
| Occupied wall/counter (`W`) | Empty hands | Take the counter's complete item, leaving the counter empty. |
| Pot (`P`) | One raw ingredient held; the pot is idle, not cooked, and not full | Insert the ingredient into the pot and empty the hands. |
| Cooked pot (`P`) | Holding an empty plate | Fill the plate with the pot's entire cooked contents, clearing the pot. |
| Delivery station (`X`) | Holding a cooked dish | Submit the complete plated dish and consume both dish and plate regardless of correctness, leaving hands empty. |
| Always-on recipe indicator (`R`) | Any | Interacting is unnecessary and has no effect; recipe channels are visible whenever this cell lies in view. |
| Button recipe indicator (`L`) | Hands empty, with no item occupying the cell | Turn on or renew the recipe display and charge the configured cost. |
| Floor or any unmet condition above | Any | No transfer, delivery, or activation occurs, but the timestep elapses. |

Counter items cannot be swapped: a player whose hands are occupied cannot pick up from an occupied counter. No trash action, chopping action, or direct pickup of uncooked pot contents is available. Plates cannot scoop from cooking/uncooked pots, and no ingredients may be added to contents already cooking or cooked. Cooked food neither burns nor spoils.

## Pot capacity, cooking, and exact timer timing

Each pot holds at most **3 raw ingredients**. The interaction that fills the pot to its capacity of 3 ingredients starts cooking automatically; there is no distinct start-cooking action.
The configured cooking duration is **20 timesteps**. The initiating interaction sets the timer, which is decremented when that very step ends.
The first observation returned after cooking starts therefore shows **19**. Subsequent steps each decrement it once; after **19 additional steps**, the timer is zero and the cooked flag is set. Since interactions come before timer updates, pickup can succeed at the next interaction, not during the transition that finishes cooking.

Pot states are distinguished by both contents and timer: empty means no contents,
no cooked flag and timer 0; partially filled idle means raw counts and timer 0;
cooking means timer > 0; ready means cooked flag 1 and timer 0. A timer of 0 alone does not establish readiness. The timer records a raw integer rather than fractional cooking progress. Cooked contents stay in the pot until collected, and collection empties it for reuse.

## Recipe button duration and cost

The button's configured active duration is **10 timesteps**. Each successful activation, even reactivation of an active display, immediately costs the team 5 reward.
Activation sets its internal timer to 10; the same step decrements it to
9. In the absence of reactivation, its recipe is therefore visible
in the first **9 post-step observations** after activation (if the
button lies inside the view), then disappears when its timer becomes zero.
Reactivating restarts the countdown. Activation is blocked while holding an ingredient, plate, or dish. The static button channel persists even during inactivity.

The numeric countdown is exposed for pots only: the button countdown is **not an observation channel**. An active button's visible display consists of nonzero recipe counts at its cell. Viewing the always-on recipe indicator has no cost, and that indicator never expires.

## Shared rewards and termination

| Event | Team reward |
| --- | --- |
| Cooked dish matches the current recipe exactly | +20 |
| Cooked dish has the wrong ingredient counts | -20 |
| Button activation | −5 each time |
| Movement, waiting, ordinary pickups/drops, adding ingredients, cooking, or ineffective actions | 0 |

Same-step events add together, including an activation and delivery on one step.
For example, a correct delivery with an immediate button cost contributes
20 − 5 = 15 when those two events coincide. Ordinary elapsed time carries no penalty. Auxiliary shaping diagnostics may be reported by the environment, but the cooperative return used to score policies excludes them.

After the 400-th transition the episode is terminal; reward from that
transition still counts. A delivery leaves the episode running, allowing further dishes to be completed until the time limit. No terminal bonus is given for unfinished food or items in hand. Preparing an incorrect dish or incurring a button cost alone does not terminate the environment.

## Observation space: exact tensor and channels

Each seat receives its own **[5, 5, 45] int32 JAX array**,
indexed `obs[row, column, channel]`. The view radius is 2, giving a 5 × 5 square centered on self at local cell (2, 2). Local cell (i, j) corresponds to global (self_row + i − 2, self_column + j − 2). Facing does not rotate the crop. No wall or object occludes cells within the square. Off-map cells are padded with zeros, which do not indicate traversable floor. Anything outside the view is absent, including a partner or recipe indicator outside the crop.
Encoding is relative to the observer: regardless of seat, the first player block is self and the second is partner. All features for an invisible partner are zero. Facing is a one-hot direction at the player's cell rather than a grid rotation, and inventory features occur only at that cell too.

Every channel is individually listed in the table; none are hidden or omitted. Slices in `specification.json` use half-open intervals `[start, stop)`.

| Channel | Meaning | Values / location |
| --- | --- | --- |
| 0 | Self position | Value 1 at that player's position and 0 at every other cell. |
| 1 | Self facing UP | One-hot encoding located at that player's position. |
| 2 | Self facing DOWN | One-hot encoding located at that player's position. |
| 3 | Self facing RIGHT | One-hot encoding located at that player's position. |
| 4 | Self facing LEFT | One-hot encoding located at that player's position. |
| 5 | Self held plate flag | Present only at the player's location, with 0 at other cells. |
| 6 | Self held cooked flag | Present only at the player's location, with 0 at other cells. |
| 7 | Self held ingredient 0 count | Present only at the player's location, with 0 at other cells. |
| 8 | Self held ingredient 1 count | Present only at the player's location, with 0 at other cells. |
| 9 | Self held ingredient 2 count | Present only at the player's location, with 0 at other cells. |
| 10 | Self held ingredient 3 count | Present only at the player's location, with 0 at other cells. |
| 11 | Partner position | Value 1 at that player's position and 0 at every other cell. |
| 12 | Partner facing UP | One-hot encoding located at that player's position. |
| 13 | Partner facing DOWN | One-hot encoding located at that player's position. |
| 14 | Partner facing RIGHT | One-hot encoding located at that player's position. |
| 15 | Partner facing LEFT | One-hot encoding located at that player's position. |
| 16 | Partner held plate flag | Present only at the player's location, with 0 at other cells. |
| 17 | Partner held cooked flag | Present only at the player's location, with 0 at other cells. |
| 18 | Partner held ingredient 0 count | Present only at the player's location, with 0 at other cells. |
| 19 | Partner held ingredient 1 count | Present only at the player's location, with 0 at other cells. |
| 20 | Partner held ingredient 2 count | Present only at the player's location, with 0 at other cells. |
| 21 | Partner held ingredient 3 count | Present only at the player's location, with 0 at other cells. |
| 22 | wall | Value 1 on static objects of this type, otherwise 0. |
| 23 | goal | Value 1 on static objects of this type, otherwise 0. |
| 24 | pot | Value 1 on static objects of this type, otherwise 0. |
| 25 | recipe indicator | Value 1 on static objects of this type, otherwise 0. |
| 26 | button indicator | Value 1 on static objects of this type, otherwise 0. |
| 27 | plate pile | Value 1 on static objects of this type, otherwise 0. |
| 28 | Ingredient 0 supply | Value 1 marks cells supplying this ingredient without depletion. |
| 29 | Ingredient 1 supply | Value 1 marks cells supplying this ingredient without depletion. |
| 30 | Ingredient 2 supply | Value 1 marks cells supplying this ingredient without depletion. |
| 31 | Ingredient 3 supply | Value 1 marks cells supplying this ingredient without depletion. |
| 32 | Item on grid cell: plate flag | Located on counters and in pots, not in held inventory. |
| 33 | Item on grid cell: cooked flag | Located on counters and in pots, not in held inventory. |
| 34 | Item on grid cell: ingredient 0 count | Located on counters and in pots, not in held inventory. |
| 35 | Item on grid cell: ingredient 1 count | Located on counters and in pots, not in held inventory. |
| 36 | Item on grid cell: ingredient 2 count | Located on counters and in pots, not in held inventory. |
| 37 | Item on grid cell: ingredient 3 count | Located on counters and in pots, not in held inventory. |
| 38 | Displayed target recipe: plate flag | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 39 | Displayed target recipe: cooked flag | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 40 | Displayed target recipe: ingredient 0 count | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 41 | Displayed target recipe: ingredient 1 count | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 42 | Displayed target recipe: ingredient 2 count | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 43 | Displayed target recipe: ingredient 3 count | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 44 | Remaining pot cooking time | Unnormalized integer countdown on pot cells and 0 elsewhere. |

Static, position, facing, plate, and cooked indicators use 0/1 values. Ingredient channels hold counts, 0–3 in the native encoding, rather than one-hot recipe categories. Each item/recipe block has 6 entries in this configuration: plate flag, cooked flag, and then counts by ingredient ID. In general its width is 2 + number of ingredient types.
An item block with nothing in it is all zero. Only the plate flag marks an empty plate; one ingredient count marks a raw ingredient. A plated cooked dish includes both flags and its ingredient counts. Cooked pot contents use the cooked flag and counts but not a plate flag. Recipe displays use counts alone and keep both flags zero.

There is no dedicated static channel for floor. Inside the known map, a floor cell has neither a static-object flag nor an ingredient-supply flag. The grid-item block contains counter and pot contents; player inventory blocks alone contain held contents. Recipe displays are encoded separately from the actual items at their cells. An all-zero recipe block indicates absence of a displayed recipe, not an empty target recipe.

This array provides no explicit global position, global map state, step counter, teammate memory, previous action, or button countdown. The policy separately
receives its reset flag, private memory and PRNG; it does not receive reward as an
act-function input. The optional successful-delivery channel is disabled, so no delivery-success bit is present.
If needed, remembering observations and counting elapsed steps must be implemented by the policy itself.

## Reset and randomness

Sample each seat uniformly from unoccupied cells in its own listed connected floor region; it cannot spawn in the other room. Facing is sampled uniformly over four directions. Randomizing position also randomizes facing even when there is no separate orientation override.
Initially both hands, every counter, and every pot are empty. Pot and button timers are zero, and buttons are inactive. The episode clock starts at zero and the recipe is sampled as
listed above. Every correct delivery triggers uniform sampling of a fresh recipe for the next observation, possibly the same recipe again. An incorrect delivery does not trigger resampling. Geometry and supply locations remain fixed. The configured reset, recipe sampling, and the policy's own PRNG provide randomness; movement, interactions, timers, and rewards obey the rules above.

The requested settings are in `config.yaml`, and `specification.json` supplies resolved constants, object coordinates, spawn regions, and the observation schema. The board and legend are repeated in `layout.md`. These descriptive files do not supply the environment's executable implementation or policy helpers.


See specification.json for numeric dimensions, resolved flags, and array schemas.
