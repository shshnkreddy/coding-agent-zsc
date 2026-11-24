# Overcooked V2: grounded_coord_ring

## Goal and episode

Delivering a dish is correct when its ingredient counts match the currently requested recipe. The two players cooperate to do this by loading raw ingredients into a pot, cooking them, collecting the cooked contents on an empty plate, and interacting with a delivery station while carrying the plated dish. Insertion order has no effect on recipe matching. A mixed or otherwise incorrect dish can be cooked but is not a correct delivery.

An environment timestep gives each player one simultaneous action and the same team reward. An episode has an upper limit of **400 joint timesteps**.
Episode team rewards are summed to score the task; they are counted once, not once for each player. Players may complete several dishes before the time limit because delivery does not terminate the episode. Choosing the SP or XP policy-search objective described in `../task_spec.md` does not alter these mechanics.

## Board, objects, and access

The fixed grid measures **9 rows × 9 columns**. Its **(row, column)** coordinates are zero-based: row values increase downward, column values rightward.
The only walkable cells are spaces and `A` cells. A cell marked `A` is a layout spawn marker and does not retain an object during play. Other listed objects prevent movement.

```text
WWW2R2WWW
W       W
W WWLWW W
2 0   B 2
RAXAP X R
2 1   B 2
W WWLWW W
W       W
WWW2R2WWW
```

| Object | Global (row, column) locations |
| --- | --- |
| Wall / counter (`W`) | (0, 0), (0, 1), (0, 2), (0, 6), (0, 7), (0, 8), (1, 0), (1, 8), (2, 0), (2, 2), (2, 3), (2, 5), (2, 6), (2, 8), (6, 0), (6, 2), (6, 3), (6, 5), (6, 6), (6, 8), (7, 0), (7, 8), (8, 0), (8, 1), (8, 2), (8, 6), (8, 7), (8, 8) |
| Ingredient supply 2 | (0, 3), (0, 5), (3, 0), (3, 8), (5, 0), (5, 8), (8, 3), (8, 5) |
| Always-on recipe indicator (`R`) | (0, 4), (4, 0), (4, 8), (8, 4) |
| Button recipe indicator (`L`) | (2, 4), (6, 4) |
| Ingredient supply 0 | (3, 2) |
| Plate supply (`B`) | (3, 6), (5, 6) |
| Delivery station (`X`) | (4, 2), (4, 6) |
| Pot (`P`) | (4, 4) |
| Ingredient supply 1 | (5, 2) |

| Seat | Layout start (row, column) | Reachable floor / reset sampling region |
| --- | --- | --- |
| agent_0 | (4, 1) | (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 1), (2, 7), (3, 1), (3, 7), (4, 1), (4, 7), (5, 1), (5, 7), (6, 1), (6, 7), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7) |
| agent_1 | (4, 3) | (3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5) |

Knowing the layout before play does not mean receiving global player coordinates as a separate policy input. See below for the runtime inputs.

## Recipes and supplies

The game has **3 ingredient types** and the target-recipe possibilities below.
In a list such as `[0, 0, 0]`, ingredient 0 occurs three times, rather than representing three types.
Each reset samples one of the listed recipe entries uniformly.

| Ingredient IDs in dish | Counts by ingredient ID | Sampling probability |
| --- | --- | --- |
| [0, 0, 0] | [3, 0, 0] | 1/2 per listed entry |
| [1, 1, 1] | [0, 3, 0] | 1/2 per listed entry |

The next observation after any correct delivery uses a newly sampled uniform recipe, which may match the old one. Incorrect deliveries do not cause a new sample. A held dish or existing pot contents do not change with the target recipe. The always-on indicator and any button still active do use the new target: an active button displays the current recipe, not the one captured at activation.

There is no per-player ingredient permutation: ingredient IDs 0 through 2 mean the same thing to both players for the entire episode.

## Action space and simultaneous steps

Every policy chooses a scalar integer from **0–5**. The six IDs are always valid choices; a separate legal-action mask is not supplied. A timestep elapses even for a blocked move or ineffective interaction.

| Action ID | Action | Effect |
| --- | --- | --- |
| 0 | RIGHT | Orient rightward and try to move one column to the right. |
| 1 | DOWN | Orient downward and try to move down one row. |
| 2 | LEFT | Orient leftward and try to move one column to the left. |
| 3 | UP | Orient upward and try to move up one row. |
| 4 | STAY | Position, facing, and inventory stay the same, but the clock advances. |
| 5 | INTERACT | Interact directly ahead with the adjacent cell, remaining in place and facing the same way. |

Facing direction IDs in the observation are **UP=0, DOWN=1, RIGHT=2, LEFT=3**. These IDs are different from the movement-action IDs. A movement action still changes facing when translation is prevented by a wall, object, boundary, or collision. Facing is retained by STAY and INTERACT. An object is not interacted with merely by walking into it; interaction is separate.

A transition executes the following steps in order:

1. Try to move both players, without leaving the board or entering non-floor destinations.
2. Resolve occupancy conflicts without allowing a common final cell or a direct cell swap. Conflicting moves undo position changes while preserving attempted facing. Successfully vacated cells can be entered if this does not swap players.
3. Process INTERACT by seat (agent_0, then agent_1), exposing earlier grid changes to the later interaction. Both delivery checks use the target recipe from the beginning of this step.
4. Tick down the pot and button timers. A pot becomes cooked when its timer reaches zero.
5. Apply recipe resampling when a correct delivery calls for it, increment the episode clock, and provide the resulting local observations and common reward.

## Inventory and every object interaction

The single inventory slot can be empty or hold one raw ingredient, one empty plate, or one plated cooked dish. Supply objects are inexhaustible. Players cannot take objects from a distance or pass held items directly to one another.

| Facing object | Required inventory/state | Result of INTERACT |
| --- | --- | --- |
| Ingredient supply (`0`, `1`, etc.) | Empty hands | Collect one ingredient unit; its supply does not run out. |
| Plate supply (`B`) | Empty hands | Collect one empty plate; the supply does not run out. |
| Empty wall/counter (`W`) | Holding any item | Set the held item down in full on the counter, emptying the hands. |
| Occupied wall/counter (`W`) | Empty hands | Lift the full item off the counter, emptying that counter. |
| Pot (`P`) | The player holds one raw ingredient; the pot must be idle, uncooked, and below capacity | Deposit the ingredient in the pot and clear the held item. |
| Cooked pot (`P`) | Holding an empty plate | Plate everything cooked in the pot and clear its contents. |
| Delivery station (`X`) | Holding a cooked dish | Turn in the entire plated dish, consuming its plate as well even for an incorrect dish; nothing remains held. |
| Always-on recipe indicator (`R`) | Any | The recipe channels are shown whenever the cell is in view. An interaction is neither required nor effective. |
| Button recipe indicator (`L`) | The player holds nothing and the cell has no item | Activate the recipe indicator or renew its activation, with the configured cost applied. |
| Floor or any unmet condition above | Any | The clock advances despite no item transfer, delivery, or activation. |

A counter cannot exchange its item for a held item; occupied hands cannot pick up an item from an occupied counter. There is no trash or chopping action, or action for direct pickup of uncooked pot contents. A plate cannot scoop a cooking/uncooked pot, and cooking or cooked contents accept no additional ingredients. Finished food never burns or spoils.

## Pot capacity, cooking, and exact timer timing

Each pot holds at most **3 raw ingredients**. Cooking starts automatically during the interaction that brings the pot's ingredient count to 3. A separate start-cooking action is absent.
The configured cooking duration is **20 timesteps**. The timer is initialized by the initiating interaction and already decreases at the end of that step.
Accordingly, **19** appears in the first observation after cooking starts. With one decrement per step thereafter, **19 additional steps** leave a zero timer and a set cooked flag. Pickup is possible on the next interaction, not on the transition that finishes cooking: interactions run before timer updates.

Pot states are distinguished by both contents and timer: empty means no contents,
no cooked flag and timer 0; partially filled idle means raw counts and timer 0;
cooking means timer > 0; ready means cooked flag 1 and timer 0. The timer is an unnormalized integer. Timer 0 alone is not proof of readiness. Cooked food remains in the pot until collection; collecting it clears the contents so the pot can be reused.

## Recipe button duration and cost

The button's configured active duration is **10 timesteps**. Successful activation always incurs an immediate team cost of 5 reward, even if the button is already active.
Activation sets its internal timer to 10; the same step decrements it to
9. In the absence of reactivation, its recipe is therefore visible
in the first **9 post-step observations** after activation (if the
button lies inside the view), then disappears when its timer becomes zero.
A reactivation resets the countdown, but holding an ingredient, plate, or dish prevents activation. Even an inactive button retains its static channel.

While a button is active, nonzero recipe counts at its cell encode display visibility. The button countdown is **not an observation channel**. Numeric countdown observations belong only to pots. The always-on recipe indicator neither expires nor charges for viewing.

## Shared rewards and termination

| Event | Team reward |
| --- | --- |
| Cooked dish matches the current recipe exactly | +20 |
| Cooked dish has the wrong ingredient counts | -20 |
| Button activation | −5 each time |
| Movement, waiting, ordinary pickups/drops, adding ingredients, cooking, or ineffective actions | 0 |

Same-step events add together, including an activation and delivery on one step.
For example, a correct delivery with an immediate button cost contributes
20 − 5 = 15 when those two events coincide. Time has no ordinary penalty. Although auxiliary shaping diagnostics may appear in environment reports, they do not contribute to the cooperative return that scores the policy.

After the 400-th transition the episode is terminal; reward from that
transition still counts. Players may complete several dishes before the time limit because delivery does not terminate the episode. Unfinished food and carried items earn no bonus on termination. The environment does not end solely from preparing the wrong dish or paying a button cost.

## Observation space: exact tensor and channels

Each seat receives its own **[5, 5, 40] int32 JAX array**,
indexed `obs[row, column, channel]`. The view radius is 2, giving a 5 × 5 square centered on self at local cell (2, 2). Local cell (i, j) corresponds to global (self_row + i − 2, self_column + j − 2). The square crop does not rotate when facing changes, and walls or objects do not occlude its cells. Zero padding marks cells outside the map but does not establish that they are floor. Cells outside the view are omitted, including a partner or recipe indicator beyond the crop.
Seat number does not change block meanings: self comes first and partner second in this observer-relative encoding. Invisible partner features contain only zeros. Facing is encoded one-hot at the player's cell, without rotating the grid. Inventory features are also located only at that cell.

The table lists every channel individually; none are hidden. Slices in `specification.json` use half-open intervals `[start, stop)`.

| Channel | Meaning | Values / location |
| --- | --- | --- |
| 0 | Self position | A value of 1 marks the player's location, with 0 elsewhere. |
| 1 | Self facing UP | A one-hot feature on the cell occupied by that player. |
| 2 | Self facing DOWN | A one-hot feature on the cell occupied by that player. |
| 3 | Self facing RIGHT | A one-hot feature on the cell occupied by that player. |
| 4 | Self facing LEFT | A one-hot feature on the cell occupied by that player. |
| 5 | Self held plate flag | Located on the cell occupied by that player and 0 outside it. |
| 6 | Self held cooked flag | Located on the cell occupied by that player and 0 outside it. |
| 7 | Self held ingredient 0 count | Located on the cell occupied by that player and 0 outside it. |
| 8 | Self held ingredient 1 count | Located on the cell occupied by that player and 0 outside it. |
| 9 | Self held ingredient 2 count | Located on the cell occupied by that player and 0 outside it. |
| 10 | Partner position | A value of 1 marks the player's location, with 0 elsewhere. |
| 11 | Partner facing UP | A one-hot feature on the cell occupied by that player. |
| 12 | Partner facing DOWN | A one-hot feature on the cell occupied by that player. |
| 13 | Partner facing RIGHT | A one-hot feature on the cell occupied by that player. |
| 14 | Partner facing LEFT | A one-hot feature on the cell occupied by that player. |
| 15 | Partner held plate flag | Located on the cell occupied by that player and 0 outside it. |
| 16 | Partner held cooked flag | Located on the cell occupied by that player and 0 outside it. |
| 17 | Partner held ingredient 0 count | Located on the cell occupied by that player and 0 outside it. |
| 18 | Partner held ingredient 1 count | Located on the cell occupied by that player and 0 outside it. |
| 19 | Partner held ingredient 2 count | Located on the cell occupied by that player and 0 outside it. |
| 20 | wall | A matching static object is marked 1; other locations are 0. |
| 21 | goal | A matching static object is marked 1; other locations are 0. |
| 22 | pot | A matching static object is marked 1; other locations are 0. |
| 23 | recipe indicator | A matching static object is marked 1; other locations are 0. |
| 24 | button indicator | A matching static object is marked 1; other locations are 0. |
| 25 | plate pile | A matching static object is marked 1; other locations are 0. |
| 26 | Ingredient 0 supply | The ingredient's nondepleting supply locations are marked 1. |
| 27 | Ingredient 1 supply | The ingredient's nondepleting supply locations are marked 1. |
| 28 | Ingredient 2 supply | The ingredient's nondepleting supply locations are marked 1. |
| 29 | Item on grid cell: plate flag | Applies to counters and pots rather than items being held. |
| 30 | Item on grid cell: cooked flag | Applies to counters and pots rather than items being held. |
| 31 | Item on grid cell: ingredient 0 count | Applies to counters and pots rather than items being held. |
| 32 | Item on grid cell: ingredient 1 count | Applies to counters and pots rather than items being held. |
| 33 | Item on grid cell: ingredient 2 count | Applies to counters and pots rather than items being held. |
| 34 | Displayed target recipe: plate flag | Restricted to R and active L cells. The plate/cooked flags for a recipe are zero. |
| 35 | Displayed target recipe: cooked flag | Restricted to R and active L cells. The plate/cooked flags for a recipe are zero. |
| 36 | Displayed target recipe: ingredient 0 count | Restricted to R and active L cells. The plate/cooked flags for a recipe are zero. |
| 37 | Displayed target recipe: ingredient 1 count | Restricted to R and active L cells. The plate/cooked flags for a recipe are zero. |
| 38 | Displayed target recipe: ingredient 2 count | Restricted to R and active L cells. The plate/cooked flags for a recipe are zero. |
| 39 | Remaining pot cooking time | Pot cells hold an unnormalized integer countdown, while other locations hold 0. |

Static-object, position, direction, plate, and cooked flags take 0/1 values. Ingredient channels encode native counts of 0–3 rather than one-hot recipe labels. The item/recipe blocks use 5 entries here: first plate flag, then cooked flag, then ingredient ID counts. Their general size is 2 + number of ingredient types.
When no item is present, its item block is entirely zero. An empty plate carries only its plate flag, raw ingredients carry one ingredient count, and plated cooked food carries both flags and ingredient counts. A cooked pot lacks a plate flag but has the cooked flag and counts. Recipe indicators display ingredient counts with both flags kept zero.

Counter and pot contents use the grid-item block; only player inventory blocks show held contents. Recipe display features are distinct from actual cell items, and an all-zero recipe block means no recipe is displayed rather than an empty target. No static channel is dedicated to floor: inside the known map, floor cells lack static-object and ingredient-supply flags.

The array omits explicit global position, global map state, step count, teammate memory, previous action, and button countdown. The policy separately
receives its reset flag, private memory and PRNG; it does not receive reward as an
act-function input. The optional successful-delivery channel is disabled, so no delivery-success bit is present.
Keeping past observations and an elapsed-step count is work for the policy.

## Reset and randomness

Choose each seat's spawn uniformly among unoccupied cells of its own listed connected floor region. Spawning in the other room is impossible. The facing distribution is uniform over four directions, and position randomization itself randomizes facing without an additional orientation override.
At the start, both players' hands and all counters and pots are empty. All pot and button timers are zero and no button is active. The episode clock starts at zero and the recipe is sampled as
listed above. The next observation after any correct delivery uses a newly sampled uniform recipe, which may match the old one. Incorrect deliveries do not cause a new sample. The geometry and supply locations stay unchanged. Only configured reset, recipe sampling, and the policy's own PRNG introduce randomness; the above rules govern movement, interaction, timers, and rewards.

Requested configuration is recorded in `config.yaml`. Resolved constants, object coordinates, spawn regions, and observation schema are recorded in `specification.json`, while `layout.md` repeats the board and legend. This is an environment description, not executable environment code or policy helpers.


Consult specification.json for resolved flags, numeric dimensions, and array schemas.
