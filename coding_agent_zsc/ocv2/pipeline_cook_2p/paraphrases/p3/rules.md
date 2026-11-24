# Overcooked V2: pipeline_cook_2p

## Goal and episode

Players jointly prepare and deliver dishes for the recipe currently requested. To make a dish, add raw ingredients to a pot, allow cooking to finish, collect its cooked contents with an empty plate, then interact with the delivery station while holding the plated result. Ingredient counts determine a match; the insertion sequence is irrelevant. Mixed and other incorrect dishes may cook, although they are not correct deliveries.

The players act simultaneously once each per environment timestep; both receive the same team reward. Play continues for no more than **400 joint timesteps** per episode.
The score adds team rewards across the episode, with each reward counted once rather than per player. Deliveries do not terminate play, so multiple dishes can be completed within the time limit. These mechanics remain the same under the SP or XP objective documented in `../task_spec.md`.

## Board, objects, and access

A fixed **9 rows × 27 columns** board is used. The zero-based coordinates are **(row, column)**, with rows increasing downward and columns increasing rightward.
Movement is allowed only on floor marked by spaces or `A`. Here `A` marks a spawn in the layout, not an object left in play. Every other listed object is impassable.

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

Before the episode, the layout is available as information. Runtime inputs, described below, do not separately provide global player coordinates.

## Recipes and supplies

There are **3 ingredient types**; possible target recipes are listed here.
A recipe list like `[0, 0, 0]` counts three units of ingredient 0; it does not denote three distinct types.
A reset draws uniformly among the listed recipe entries.

| Ingredient IDs in dish | Counts by ingredient ID | Sampling probability |
| --- | --- | --- |
| [0, 0, 1] | [2, 1, 0] | 1/4 per listed entry |
| [1, 1, 2] | [0, 2, 1] | 1/4 per listed entry |
| [0, 2, 2] | [1, 0, 2] | 1/4 per listed entry |
| [0, 1, 2] | [1, 1, 1] | 1/4 per listed entry |

For the observation after a correct delivery, sample a recipe uniformly anew, allowing the old recipe to recur. No resampling follows a wrong delivery. After a recipe change, the always-on indicator and each active button use that new recipe. Button displays are current, not snapshots taken at activation. Existing pot contents and held dishes remain as they were.

The meanings of ingredient IDs 0 through 2 stay fixed for the episode and agree between players. The configuration enables no per-player ingredient permutation.

## Action space and simultaneous steps

Policy output is a scalar integer in **0–5**. There is no separate legal-action mask, because all six IDs are permitted in every state. Blocked movement and ineffective interactions consume time as one timestep.

| Action ID | Action | Effect |
| --- | --- | --- |
| 0 | RIGHT | Set facing right and attempt to advance one column in that direction. |
| 1 | DOWN | Set facing down and attempt to move one row in that direction. |
| 2 | LEFT | Set facing left and attempt to advance one column in that direction. |
| 3 | UP | Set facing up and attempt to move one row in that direction. |
| 4 | STAY | Wait without altering position, facing, or inventory; time continues. |
| 5 | INTERACT | Use the adjacent cell in front, with neither translation nor turning. |

Facing direction IDs in the observation are **UP=0, DOWN=1, RIGHT=2, LEFT=3**. Movement-action IDs are distinct from these IDs. Walls, objects, map boundaries, or collisions can block translation without preventing a movement action's facing change. STAY and INTERACT keep facing unchanged. Walking into an object is not interaction; interaction needs its own step.

The ordered transition steps are:

1. Process both movement attempts, restricting destinations to floor cells on the board.
2. Handle occupied-cell conflicts by disallowing shared destinations and direct swaps. On conflict, position reverts but attempted facing is retained. Moving to a cell the other player successfully vacates is allowed when it is not a swap.
3. Handle INTERACT for agent_0, then agent_1, so the latter sees grid changes already made. Evaluate both deliveries against the target recipe held at the start of the step.
4. Decrease both pot and button timers, marking a pot cooked when its timer reaches zero.
5. Perform any recipe resampling required by a correct delivery, advance the episode clock, then produce the resulting local observations and shared reward.

## Inventory and every object interaction

Supplies are unlimited, but each player can hold only one raw ingredient, one empty plate, or one plated cooked dish in their single inventory slot; otherwise their hands are empty. Items cannot be picked up remotely or handed directly to the other player.

| Facing object | Required inventory/state | Result of INTERACT |
| --- | --- | --- |
| Ingredient supply (`0`, `1`, etc.) | Empty hands | Acquire one unit of this ingredient, leaving its supply unchanged. |
| Plate supply (`B`) | Empty hands | Acquire a single empty plate, leaving the supply unchanged. |
| Empty wall/counter (`W`) | Holding any item | Transfer everything held to the counter; the player then holds nothing. |
| Occupied wall/counter (`W`) | Empty hands | Transfer the entire counter item into the player's hands; nothing remains on the counter. |
| Pot (`P`) | Holding a single raw ingredient with the pot idle, uncooked, and under capacity | Transfer the ingredient to the pot; the inventory becomes empty. |
| Cooked pot (`P`) | Holding an empty plate | Collect the full cooked contents on the plate, leaving an empty pot. |
| Delivery station (`X`) | Holding a cooked dish | Consume the full dish and its plate upon delivery, correct or not; the inventory is emptied. |
| Always-on recipe indicator (`R`) | Any | Viewing the cell exposes its recipe channels. Interacting with it is unnecessary and has no effect. |
| Button recipe indicator (`L`) | Empty inventory and an item-free target cell | Start or extend the recipe display and apply its configured cost. |
| Floor or any unmet condition above | Any | Items do not transfer and no delivery or activation occurs; time proceeds. |

Finished food remains unburned and unspoiled. Ingredients may not be added once contents are cooking or cooked, and plates cannot scoop cooking/uncooked pots. Direct pickup of uncooked pot contents, trash actions, and chopping actions do not exist. Nor can items be swapped on counters: occupied hands cannot pick up from an occupied counter.

## Pot capacity, cooking, and exact timer timing

Each pot holds at most **3 raw ingredients**. Filling the pot to its capacity of 3 ingredients in an interaction automatically initiates cooking; no separate start-cooking action exists.
The configured cooking duration is **20 timesteps**. The interaction that initiates cooking sets the timer before that step's final decrement.
After starting cooking, the first returned observation contains **19** on the timer. One decrement per subsequent step brings it to zero with the cooked flag set after **19 additional steps**. Interactions precede timer updates, so pickup must wait for the following interaction instead of succeeding in the finishing transition.

Pot states are distinguished by both contents and timer: empty means no contents,
no cooked flag and timer 0; partially filled idle means raw counts and timer 0;
cooking means timer > 0; ready means cooked flag 1 and timer 0. Timer 0 does not by itself identify a ready dish. This is an integer timer, not a fractional progress measure. The cooked contents persist until collection empties the pot for another use.

## Recipe button duration and cost

The button's configured active duration is **10 timesteps**. An immediate team cost of 5 reward applies to every successful activation, including reactivation.
Activation sets its internal timer to 10; the same step decrements it to
9. In the absence of reactivation, its recipe is therefore visible
in the first **9 post-step observations** after activation (if the
button lies inside the view), then disappears when its timer becomes zero.
Reactivation starts the countdown over. Hands holding an ingredient, plate, or dish cannot activate the button. Its static channel stays present while inactive.

The always-on recipe indicator remains available without expiring, and viewing it is free. For a button, visibility is given by nonzero recipe counts at its cell while active. The button countdown is **not an observation channel**: only pots provide a numeric countdown.

## Shared rewards and termination

| Event | Team reward |
| --- | --- |
| Cooked dish matches the current recipe exactly | +20 |
| Cooked dish has the wrong ingredient counts | -20 |
| Button activation | −5 each time |
| Movement, waiting, ordinary pickups/drops, adding ingredients, cooking, or ineffective actions | 0 |

Same-step events add together, including an activation and delivery on one step.
For example, a correct delivery with an immediate button cost contributes
20 − 5 = 15 when those two events coincide. The cooperative return used for policy scoring does not include any reported auxiliary shaping diagnostics. There is no ordinary penalty for time passing.

After the 400-th transition the episode is terminal; reward from that
transition still counts. Deliveries do not terminate play, so multiple dishes can be completed within the time limit. A button cost or the preparation of an incorrect dish is not, by itself, a termination trigger. There is no terminal bonus for unfinished food or held items.

## Observation space: exact tensor and channels

Each seat receives its own **[5, 5, 40] int32 JAX array**,
indexed `obs[row, column, channel]`. The view radius is 2, giving a 5 × 5 square centered on self at local cell (2, 2). Local cell (i, j) corresponds to global (self_row + i − 2, self_column + j − 2). There is no facing-dependent crop rotation or wall/object occlusion within the square. Map-exterior cells are zero padded rather than identified as floor. Cells beyond the view, including an out-of-crop partner or recipe indicator, are absent.
Player blocks are observer-relative, with self first and partner second for either seat number. If the partner is invisible, its features are all zero. Facing uses a one-hot direction at the player's cell instead of rotating the grid. Inventory features likewise exist only at that player's cell.

No channels are omitted or hidden: the table lists each one separately. Slices in `specification.json` use half-open intervals `[start, stop)`.

| Channel | Meaning | Values / location |
| --- | --- | --- |
| 0 | Self position | Use 1 for that player's cell and 0 for other cells. |
| 1 | Self facing UP | Encoded one-hot only at that player's location. |
| 2 | Self facing DOWN | Encoded one-hot only at that player's location. |
| 3 | Self facing RIGHT | Encoded one-hot only at that player's location. |
| 4 | Self facing LEFT | Encoded one-hot only at that player's location. |
| 5 | Self held plate flag | Only that player's cell carries the value; elsewhere it is 0. |
| 6 | Self held cooked flag | Only that player's cell carries the value; elsewhere it is 0. |
| 7 | Self held ingredient 0 count | Only that player's cell carries the value; elsewhere it is 0. |
| 8 | Self held ingredient 1 count | Only that player's cell carries the value; elsewhere it is 0. |
| 9 | Self held ingredient 2 count | Only that player's cell carries the value; elsewhere it is 0. |
| 10 | Partner position | Use 1 for that player's cell and 0 for other cells. |
| 11 | Partner facing UP | Encoded one-hot only at that player's location. |
| 12 | Partner facing DOWN | Encoded one-hot only at that player's location. |
| 13 | Partner facing RIGHT | Encoded one-hot only at that player's location. |
| 14 | Partner facing LEFT | Encoded one-hot only at that player's location. |
| 15 | Partner held plate flag | Only that player's cell carries the value; elsewhere it is 0. |
| 16 | Partner held cooked flag | Only that player's cell carries the value; elsewhere it is 0. |
| 17 | Partner held ingredient 0 count | Only that player's cell carries the value; elsewhere it is 0. |
| 18 | Partner held ingredient 1 count | Only that player's cell carries the value; elsewhere it is 0. |
| 19 | Partner held ingredient 2 count | Only that player's cell carries the value; elsewhere it is 0. |
| 20 | wall | Use 1 where the static object matches, and 0 elsewhere. |
| 21 | goal | Use 1 where the static object matches, and 0 elsewhere. |
| 22 | pot | Use 1 where the static object matches, and 0 elsewhere. |
| 23 | recipe indicator | Use 1 where the static object matches, and 0 elsewhere. |
| 24 | button indicator | Use 1 where the static object matches, and 0 elsewhere. |
| 25 | plate pile | Use 1 where the static object matches, and 0 elsewhere. |
| 26 | Ingredient 0 supply | Use 1 on the inexhaustible supply cells for this ingredient. |
| 27 | Ingredient 1 supply | Use 1 on the inexhaustible supply cells for this ingredient. |
| 28 | Ingredient 2 supply | Use 1 on the inexhaustible supply cells for this ingredient. |
| 29 | Item on grid cell: plate flag | Includes contents of counters and pots but not player inventory. |
| 30 | Item on grid cell: cooked flag | Includes contents of counters and pots but not player inventory. |
| 31 | Item on grid cell: ingredient 0 count | Includes contents of counters and pots but not player inventory. |
| 32 | Item on grid cell: ingredient 1 count | Includes contents of counters and pots but not player inventory. |
| 33 | Item on grid cell: ingredient 2 count | Includes contents of counters and pots but not player inventory. |
| 34 | Displayed target recipe: plate flag | Shown at R cells and at L cells while active, with recipe plate/cooked flags set to zero. |
| 35 | Displayed target recipe: cooked flag | Shown at R cells and at L cells while active, with recipe plate/cooked flags set to zero. |
| 36 | Displayed target recipe: ingredient 0 count | Shown at R cells and at L cells while active, with recipe plate/cooked flags set to zero. |
| 37 | Displayed target recipe: ingredient 1 count | Shown at R cells and at L cells while active, with recipe plate/cooked flags set to zero. |
| 38 | Displayed target recipe: ingredient 2 count | Shown at R cells and at L cells while active, with recipe plate/cooked flags set to zero. |
| 39 | Remaining pot cooking time | At pots, the countdown is an integer without normalization; at other cells it is 0. |

Values for static, position, facing, plate, and cooked flags are 0/1; ingredient values are counts, with native range 0–3, instead of one-hot classes for recipes. Here an item/recipe block contains 5 entries, beginning with the plate and cooked flags and continuing with counts for every ingredient ID. Its width generally equals 2 + number of ingredient types.
A block is all zero when no item is present. For an empty plate only the plate flag is present; for a raw ingredient there is one ingredient count; for a plated cooked dish both flags accompany its counts. Pot contents that are cooked have counts and the cooked flag, without the plate flag. Displayed recipes have counts with both flags zero.

Recipe display features do not describe the actual items at those cells. All-zero recipe blocks indicate no displayed recipe rather than an empty target. Counter and pot contents are in the grid-item block, and held contents are only in player inventory blocks. Floor has no separate static channel: a floor cell in the known map has no static-object or ingredient-supply flag.

Neither explicit global position nor global map state appears in the array; neither do a step counter, teammate memory, previous action, or button countdown. The policy separately
receives its reset flag, private memory and PRNG; it does not receive reward as an
act-function input. The optional successful-delivery channel is disabled, so no delivery-success bit is present.
If observation recall or an elapsed-step count is needed, the policy must implement it.

## Reset and randomness

Seats start at the listed layout start cells. Both players initially face UP.
The initial state has empty hands, counters, and pots, with zero pot/button timers and inactive buttons. The episode clock starts at zero and the recipe is sampled as
listed above. For the observation after a correct delivery, sample a recipe uniformly anew, allowing the old recipe to recur. No resampling follows a wrong delivery. Static geometry and supply positions do not change. Randomness arises from the configured reset, sampled recipes, and the policy's own PRNG; movements, interactions, timers, and rewards follow the preceding rules.

`config.yaml` contains the requested settings. Resolved constants, coordinates, spawn regions, and the observation schema appear in `specification.json`; `layout.md` restates the board and legend. The files document the selected environment without including its executable implementation or helper policies.


The specification.json file supplies numeric dimensions, resolved flags, and array schemas.
