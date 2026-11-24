# Overcooked V2: pipeline_cook_2p

## Goal and episode

The two players work together to deliver dishes that match the current recipe. Preparing a dish requires placing raw ingredients in a pot, cooking them, taking the cooked contents with an empty plate, and using the delivery station while carrying that plated dish. Matching depends on ingredient counts, not the order in which ingredients entered the pot. Cooking is possible for mixed or otherwise incorrect dishes too, but delivering them does not count as a correct delivery.

Each environment timestep contains simultaneous actions by both players, who share the same team reward. An episode is limited to **400 joint timesteps**.
Sum the episode's team rewards once, not separately for each player, to obtain the task score. A delivery leaves the episode running, allowing further dishes to be completed until the time limit. See `../task_spec.md` for the policy-search objective (SP or XP); these mechanics are unaffected by that objective.

## Board, objects, and access

The board has a fixed size of **9 rows × 27 columns**. Coordinates use zero-based **(row, column)** indexing, increasing downward for rows and rightward for columns.
Traversable floor consists only of spaces and `A` cells. The `A` is a layout spawn marker rather than a persistent object. All other listed objects obstruct movement.

```text
WW0W1W2WWWWWWPWWWWWWWWWWWWW
W       WW       WW       W
R       WW       WW       W
0  0R1  WL  PBP  WL  BWB  W
1 AWWW      WWW      WWWA X
2  2WW  WW  PBP  WW  BWB  W
R       WW       WW       W
W       WW       WW       W
WW0W1W2WWWWWWPWWWWWWWWWWWWW
```

| Object | Global (row, column) locations |
| --- | --- |
| Wall / counter (`W`) | (0, 0), (0, 1), (0, 3), (0, 5), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (1, 0), (1, 8), (1, 9), (1, 17), (1, 18), (1, 26), (2, 8), (2, 9), (2, 17), (2, 18), (2, 26), (3, 8), (3, 17), (3, 22), (3, 26), (4, 3), (4, 4), (4, 5), (4, 12), (4, 13), (4, 14), (4, 21), (4, 22), (4, 23), (5, 4), (5, 5), (5, 8), (5, 9), (5, 17), (5, 18), (5, 22), (5, 26), (6, 8), (6, 9), (6, 17), (6, 18), (6, 26), (7, 0), (7, 8), (7, 9), (7, 17), (7, 18), (7, 26), (8, 0), (8, 1), (8, 3), (8, 5), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26) |
| Ingredient supply 0 | (0, 2), (3, 0), (3, 3), (8, 2) |
| Ingredient supply 1 | (0, 4), (3, 5), (4, 0), (8, 4) |
| Ingredient supply 2 | (0, 6), (5, 0), (5, 3), (8, 6) |
| Pot (`P`) | (0, 13), (3, 12), (3, 14), (5, 12), (5, 14), (8, 13) |
| Always-on recipe indicator (`R`) | (2, 0), (3, 4), (6, 0) |
| Button recipe indicator (`L`) | (3, 9), (3, 18) |
| Plate supply (`B`) | (3, 13), (3, 21), (3, 23), (5, 13), (5, 21), (5, 23) |
| Delivery station (`X`) | (4, 26) |

| Seat | Layout start (row, column) | Reachable floor / reset sampling region |
| --- | --- | --- |
| agent_0 | (4, 2) | (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (3, 1), (3, 2), (3, 6), (3, 7), (3, 10), (3, 11), (3, 15), (3, 16), (3, 19), (3, 20), (3, 24), (3, 25), (4, 1), (4, 2), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20), (4, 24), (4, 25), (5, 1), (5, 2), (5, 6), (5, 7), (5, 10), (5, 11), (5, 15), (5, 16), (5, 19), (5, 20), (5, 24), (5, 25), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 19), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (7, 25) |
| agent_1 | (4, 24) | (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (3, 1), (3, 2), (3, 6), (3, 7), (3, 10), (3, 11), (3, 15), (3, 16), (3, 19), (3, 20), (3, 24), (3, 25), (4, 1), (4, 2), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20), (4, 24), (4, 25), (5, 1), (5, 2), (5, 6), (5, 7), (5, 10), (5, 11), (5, 15), (5, 16), (5, 19), (5, 20), (5, 24), (5, 25), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 19), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (7, 25) |

Players have the layout information before play begins. The policy does not receive a separate global-player-coordinate input; its runtime inputs are specified below.

## Recipes and supplies

The environment provides **3 ingredient types**. The possible target recipes are listed below.
For example, `[0, 0, 0]` specifies three units of ingredient 0 rather than three different types.
Reset selects a listed recipe entry uniformly.

| Ingredient IDs in dish | Counts by ingredient ID | Sampling probability |
| --- | --- | --- |
| [0, 0, 1] | [2, 1, 0] | 1/4 per listed entry |
| [1, 1, 2] | [0, 2, 1] | 1/4 per listed entry |
| [0, 2, 2] | [1, 0, 2] | 1/4 per listed entry |
| [0, 1, 2] | [1, 1, 1] | 1/4 per listed entry |

Every correct delivery triggers uniform sampling of a fresh recipe for the next observation, possibly the same recipe again. An incorrect delivery does not trigger resampling. Both the always-on indicator and any still-active button use the new recipe. A button that remains active shows the recipe currently requested rather than preserving the recipe from its activation. Changing the target leaves existing pot contents and carried dishes untouched.

Both players use the same meanings for ingredient IDs 0 through 2 throughout the episode. Per-player ingredient permutations are disabled.

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

Each seat receives its own **[5, 5, 40] int32 JAX array**,
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
| 10 | Partner position | Value 1 at that player's position and 0 at every other cell. |
| 11 | Partner facing UP | One-hot encoding located at that player's position. |
| 12 | Partner facing DOWN | One-hot encoding located at that player's position. |
| 13 | Partner facing RIGHT | One-hot encoding located at that player's position. |
| 14 | Partner facing LEFT | One-hot encoding located at that player's position. |
| 15 | Partner held plate flag | Present only at the player's location, with 0 at other cells. |
| 16 | Partner held cooked flag | Present only at the player's location, with 0 at other cells. |
| 17 | Partner held ingredient 0 count | Present only at the player's location, with 0 at other cells. |
| 18 | Partner held ingredient 1 count | Present only at the player's location, with 0 at other cells. |
| 19 | Partner held ingredient 2 count | Present only at the player's location, with 0 at other cells. |
| 20 | wall | Value 1 on static objects of this type, otherwise 0. |
| 21 | goal | Value 1 on static objects of this type, otherwise 0. |
| 22 | pot | Value 1 on static objects of this type, otherwise 0. |
| 23 | recipe indicator | Value 1 on static objects of this type, otherwise 0. |
| 24 | button indicator | Value 1 on static objects of this type, otherwise 0. |
| 25 | plate pile | Value 1 on static objects of this type, otherwise 0. |
| 26 | Ingredient 0 supply | Value 1 marks cells supplying this ingredient without depletion. |
| 27 | Ingredient 1 supply | Value 1 marks cells supplying this ingredient without depletion. |
| 28 | Ingredient 2 supply | Value 1 marks cells supplying this ingredient without depletion. |
| 29 | Item on grid cell: plate flag | Located on counters and in pots, not in held inventory. |
| 30 | Item on grid cell: cooked flag | Located on counters and in pots, not in held inventory. |
| 31 | Item on grid cell: ingredient 0 count | Located on counters and in pots, not in held inventory. |
| 32 | Item on grid cell: ingredient 1 count | Located on counters and in pots, not in held inventory. |
| 33 | Item on grid cell: ingredient 2 count | Located on counters and in pots, not in held inventory. |
| 34 | Displayed target recipe: plate flag | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 35 | Displayed target recipe: cooked flag | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 36 | Displayed target recipe: ingredient 0 count | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 37 | Displayed target recipe: ingredient 1 count | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 38 | Displayed target recipe: ingredient 2 count | Present only on R cells and active L cells; recipe plate/cooked flags remain zero. |
| 39 | Remaining pot cooking time | Unnormalized integer countdown on pot cells and 0 elsewhere. |

Static, position, facing, plate, and cooked indicators use 0/1 values. Ingredient channels hold counts, 0–3 in the native encoding, rather than one-hot recipe categories. Each item/recipe block has 5 entries in this configuration: plate flag, cooked flag, and then counts by ingredient ID. In general its width is 2 + number of ingredient types.
An item block with nothing in it is all zero. Only the plate flag marks an empty plate; one ingredient count marks a raw ingredient. A plated cooked dish includes both flags and its ingredient counts. Cooked pot contents use the cooked flag and counts but not a plate flag. Recipe displays use counts alone and keep both flags zero.

There is no dedicated static channel for floor. Inside the known map, a floor cell has neither a static-object flag nor an ingredient-supply flag. The grid-item block contains counter and pot contents; player inventory blocks alone contain held contents. Recipe displays are encoded separately from the actual items at their cells. An all-zero recipe block indicates absence of a displayed recipe, not an empty target recipe.

This array provides no explicit global position, global map state, step counter, teammate memory, previous action, or button countdown. The policy separately
receives its reset flag, private memory and PRNG; it does not receive reward as an
act-function input. The optional successful-delivery channel is disabled, so no delivery-success bit is present.
If needed, remembering observations and counting elapsed steps must be implemented by the policy itself.

## Reset and randomness

Seats start at the listed layout start cells. Both players initially face UP.
Initially both hands, every counter, and every pot are empty. Pot and button timers are zero, and buttons are inactive. The episode clock starts at zero and the recipe is sampled as
listed above. Every correct delivery triggers uniform sampling of a fresh recipe for the next observation, possibly the same recipe again. An incorrect delivery does not trigger resampling. Geometry and supply locations remain fixed. The configured reset, recipe sampling, and the policy's own PRNG provide randomness; movement, interactions, timers, and rewards obey the rules above.

The requested settings are in `config.yaml`, and `specification.json` supplies resolved constants, object coordinates, spawn regions, and the observation schema. The board and legend are repeated in `layout.md`. These descriptive files do not supply the environment's executable implementation or policy helpers.


See specification.json for numeric dimensions, resolved flags, and array schemas.
