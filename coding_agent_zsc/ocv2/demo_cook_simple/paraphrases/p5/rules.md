# Overcooked V2: demo_cook_simple

## Goal and episode

The shared task is delivery of dishes matching the current request. Both players cooperate in a preparation process: put raw ingredients into a pot, let them cook, collect the cooked contents using an empty plate, and interact with a delivery station while holding the plated dish. Counts of ingredients, rather than insertion order, decide recipe matching. Mixed or otherwise incorrect dishes still can cook, but do not constitute correct deliveries.

Both players take one action simultaneously at each environment timestep and share its team reward. At most **400 joint timesteps** make up an episode.
The team's episode reward sum is the task score, without multiplying it by the number of players. Delivering a dish is not an episode ending; additional dishes can be completed until the time limit. The objective for policy search, SP or XP, appears in `../task_spec.md` and leaves the game mechanics unchanged.

## Board, objects, and access

This board is fixed at **5 rows × 11 columns**. Positions below use zero-based **(row, column)** coordinates; increasing rows go downward and increasing columns go rightward.
Floor that can be traversed is marked by spaces and `A`. The marker `A` specifies a layout spawn rather than a remaining game object. All other listed objects block passage.

```text
WWWWWR2W0WW
0      W  B
W     APA X
1      W  B
WWWWWR2W1WW
```

| Object | Global (row, column) locations |
| --- | --- |
| Wall / counter (`W`) | (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10), (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 7), (4, 9), (4, 10) |
| Always-on recipe indicator (`R`) | (0, 5), (4, 5) |
| Ingredient supply 2 | (0, 6), (4, 6) |
| Ingredient supply 0 | (0, 8), (1, 0) |
| Plate supply (`B`) | (1, 10), (3, 10) |
| Pot (`P`) | (2, 7) |
| Delivery station (`X`) | (2, 10) |
| Ingredient supply 1 | (3, 0), (4, 8) |

| Seat | Layout start (row, column) | Reachable floor / reset sampling region |
| --- | --- | --- |
| agent_0 | (2, 6) | (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6) |
| agent_1 | (2, 8) | (1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9) |

The available pre-play information includes this layout. No separate global player coordinates are passed to the policy; the runtime input description follows below.

## Recipes and supplies

Possible target recipes for the **3 ingredient types** appear below.
Reading `[0, 0, 0]` as a recipe means three units of ingredient 0, not three different ingredient types.
The listed recipe entries are equally likely when sampled at reset.

| Ingredient IDs in dish | Counts by ingredient ID | Sampling probability |
| --- | --- | --- |
| [0, 0, 0] | [3, 0, 0] | 1/2 per listed entry |
| [1, 1, 1] | [0, 3, 0] | 1/2 per listed entry |

Correct deliveries resample the recipe uniformly for the subsequent observation, with repetition of the old recipe allowed. Wrong deliveries do not resample. New target recipes update both the always-on indicator and all buttons that are still active. Thus an active display is not frozen to its activation-time recipe. Pots retain their existing contents, and held dishes are unchanged.

For both players, ingredient IDs 0 through 2 retain identical meanings across the episode. Per-player ingredient permutation is not enabled.

## Action space and simultaneous steps

The action returned by a policy is one scalar integer in **0–5**. All six IDs remain available in every state, with no separate legal-action mask. Ineffective interactions and blocked moves still consume a timestep.

| Action ID | Action | Effect |
| --- | --- | --- |
| 0 | RIGHT | Attempt a one-column move right while setting facing to right. |
| 1 | DOWN | Attempt a one-row downward move while setting facing to down. |
| 2 | LEFT | Attempt a one-column move left while setting facing to left. |
| 3 | UP | Attempt a one-row upward move while setting facing to up. |
| 4 | STAY | Advance time without changing position, direction faced, or inventory. |
| 5 | INTERACT | Target the immediately neighboring cell along the current facing direction; no movement or turn occurs. |

Facing direction IDs in the observation are **UP=0, DOWN=1, RIGHT=2, LEFT=3**. The movement-action IDs are not these IDs. Facing changes on a movement action even if a wall, object, map boundary, or collision blocks translation. STAY and INTERACT retain the current facing. Walking into an object does not perform interaction, which takes a separate step.

The sequence within a transition is:

1. Attempt the two movements; only destinations on the board's floor are permitted.
2. Prevent players from ending in the same cell or exchanging cells directly. For conflicting moves, revert positions and retain the attempted facing updates. A cell successfully vacated by the partner can be entered unless the move is a swap.
3. Run INTERACT actions in order (agent_0, then agent_1). The second interaction sees the first's grid changes, while both deliveries are tested against the recipe from the start of the step.
4. Decrement the timers for pots and buttons, with pots becoming cooked as their timers reach zero.
5. Do any recipe resampling required following a correct delivery, advance episode time, and output the resulting local observations together with the shared reward.

## Inventory and every object interaction

Inventory consists of one slot per player. Its possibilities are empty hands, one raw ingredient, one empty plate, and one plated cooked dish. Supplies do not run out, and neither distant pickup nor direct player-to-player item passing is allowed.

| Facing object | Required inventory/state | Result of INTERACT |
| --- | --- | --- |
| Ingredient supply (`0`, `1`, etc.) | Empty hands | Take a single unit of the ingredient while preserving the supply. |
| Plate supply (`B`) | Empty hands | Take an empty plate while preserving the supply. |
| Empty wall/counter (`W`) | Holding any item | Move the entire held item onto the counter and leave the inventory empty. |
| Occupied wall/counter (`W`) | Empty hands | Acquire the counter's item in full; the counter is then vacant. |
| Pot (`P`) | One raw ingredient held and room in an idle, uncooked pot | The ingredient enters the pot, and the player's hands are cleared. |
| Cooked pot (`P`) | Holding an empty plate | Empty the pot by transferring its complete cooked contents to the plate. |
| Delivery station (`X`) | Holding a cooked dish | The whole dish, including the plate, is consumed by delivery irrespective of correctness, and the player is left empty-handed. |
| Always-on recipe indicator (`R`) | Any | Recipe channels remain visible while this cell is within view, and interactions are unnecessary and ineffective. |
| Button recipe indicator (`L`) | No held item and no item at that cell | Switch on or refresh recipe visibility while charging the configured cost. |
| Floor or any unmet condition above | Any | Time passes with no transfer of items, no delivery, and no activation. |

No chopping, trash, or direct uncooked-pot pickup action is provided. A plate does not scoop a cooking/uncooked pot; ingredients cannot enter contents that are cooking or already cooked. Food that has finished cooking does not burn or spoil. Counter swapping is also absent: occupied hands cannot pick up from an occupied counter.

## Pot capacity, cooking, and exact timer timing

Each pot holds at most **3 raw ingredients**. Cooking is initiated automatically by the interaction that fills the pot to its capacity of 3 ingredients; it cannot be started by a separate start-cooking action.
The configured cooking duration is **20 timesteps**. Initialization happens during the initiating interaction; the timer then decrements at that step's end.
The initial post-start observation consequently shows a timer value of **19**. The timer decreases once per step; after **19 additional steps**, it reaches zero and the cooked flag is set. Because timer updates follow interactions, a successful pickup occurs on the following interaction, not on the transition that first completes cooking.

Pot states are distinguished by both contents and timer: empty means no contents,
no cooked flag and timer 0; partially filled idle means raw counts and timer 0;
cooking means timer > 0; ready means cooked flag 1 and timer 0. Readiness cannot be inferred from timer 0 alone. The timer is a raw integer instead of a fraction of progress. A pot retains cooked contents until they are collected, clearing it for subsequent use.

## Recipe button duration and cost

The button's configured active duration is **10 timesteps**. Every successful activation charges 5 reward to the team immediately; reactivating an active button is included.
Activation sets its internal timer to 10; the same step decrements it to
9. In the absence of reactivation, its recipe is therefore visible
in the first **9 post-step observations** after activation (if the
button lies inside the view), then disappears when its timer becomes zero.
The countdown restarts on reactivation. Activation requires not holding an ingredient, plate, or dish, and the static button channel is still present when inactive.

Only pots reveal their numeric countdowns. The button countdown is **not an observation channel**; instead, its cell has nonzero recipe counts when the display is active and visible. There is no viewing cost or expiration for the always-on recipe indicator.

## Shared rewards and termination

| Event | Team reward |
| --- | --- |
| Cooked dish matches the current recipe exactly | +20 |
| Cooked dish has the wrong ingredient counts | -20 |
| Button activation | −5 each time |
| Movement, waiting, ordinary pickups/drops, adding ingredients, cooking, or ineffective actions | 0 |

Same-step events add together, including an activation and delivery on one step.
For example, a correct delivery with an immediate button cost contributes
20 − 5 = 15 when those two events coincide. Policy scoring uses cooperative return without adding auxiliary shaping diagnostics that the environment may report. Ordinary time passage is not penalized.

After the 400-th transition the episode is terminal; reward from that
transition still counts. Delivering a dish is not an episode ending; additional dishes can be completed until the time limit. Merely preparing an incorrect dish or incurring a button cost does not end the environment. Unfinished food and items still held receive no terminal bonus.

## Observation space: exact tensor and channels

Each seat receives its own **[5, 5, 40] int32 JAX array**,
indexed `obs[row, column, channel]`. The view radius is 2, giving a 5 × 5 square centered on self at local cell (2, 2). Local cell (i, j) corresponds to global (self_row + i − 2, self_column + j − 2). Facing leaves crop orientation unchanged; cells inside the square are not occluded by walls or objects. Cells beyond the map are zero padded, without implying floor. Cells outside the view do not appear, including any partner or recipe indicator outside the crop.
In this observer-relative representation, the first player block describes self and the second describes partner regardless of seat number. Features of an unseen partner are all zero. The grid does not rotate for facing, which is one-hot at the player's cell; inventory features occur only there too.

All channels are shown separately below; no additional channels are omitted or hidden. Slices in `specification.json` use half-open intervals `[start, stop)`.

| Channel | Meaning | Values / location |
| --- | --- | --- |
| 0 | Self position | The cell occupied by that player is 1, and other cells are 0. |
| 1 | Self facing UP | The player's position carries the one-hot representation. |
| 2 | Self facing DOWN | The player's position carries the one-hot representation. |
| 3 | Self facing RIGHT | The player's position carries the one-hot representation. |
| 4 | Self facing LEFT | The player's position carries the one-hot representation. |
| 5 | Self held plate flag | The value appears at that player's position alone; other cells are 0. |
| 6 | Self held cooked flag | The value appears at that player's position alone; other cells are 0. |
| 7 | Self held ingredient 0 count | The value appears at that player's position alone; other cells are 0. |
| 8 | Self held ingredient 1 count | The value appears at that player's position alone; other cells are 0. |
| 9 | Self held ingredient 2 count | The value appears at that player's position alone; other cells are 0. |
| 10 | Partner position | The cell occupied by that player is 1, and other cells are 0. |
| 11 | Partner facing UP | The player's position carries the one-hot representation. |
| 12 | Partner facing DOWN | The player's position carries the one-hot representation. |
| 13 | Partner facing RIGHT | The player's position carries the one-hot representation. |
| 14 | Partner facing LEFT | The player's position carries the one-hot representation. |
| 15 | Partner held plate flag | The value appears at that player's position alone; other cells are 0. |
| 16 | Partner held cooked flag | The value appears at that player's position alone; other cells are 0. |
| 17 | Partner held ingredient 0 count | The value appears at that player's position alone; other cells are 0. |
| 18 | Partner held ingredient 1 count | The value appears at that player's position alone; other cells are 0. |
| 19 | Partner held ingredient 2 count | The value appears at that player's position alone; other cells are 0. |
| 20 | wall | Cells with the indicated static object have 1, with 0 at other cells. |
| 21 | goal | Cells with the indicated static object have 1, with 0 at other cells. |
| 22 | pot | Cells with the indicated static object have 1, with 0 at other cells. |
| 23 | recipe indicator | Cells with the indicated static object have 1, with 0 at other cells. |
| 24 | button indicator | Cells with the indicated static object have 1, with 0 at other cells. |
| 25 | plate pile | Cells with the indicated static object have 1, with 0 at other cells. |
| 26 | Ingredient 0 supply | A value of 1 identifies this ingredient's unlimited supply cells. |
| 27 | Ingredient 1 supply | A value of 1 identifies this ingredient's unlimited supply cells. |
| 28 | Ingredient 2 supply | A value of 1 identifies this ingredient's unlimited supply cells. |
| 29 | Item on grid cell: plate flag | Represents counter and pot items, excluding anything held. |
| 30 | Item on grid cell: cooked flag | Represents counter and pot items, excluding anything held. |
| 31 | Item on grid cell: ingredient 0 count | Represents counter and pot items, excluding anything held. |
| 32 | Item on grid cell: ingredient 1 count | Represents counter and pot items, excluding anything held. |
| 33 | Item on grid cell: ingredient 2 count | Represents counter and pot items, excluding anything held. |
| 34 | Displayed target recipe: plate flag | These values occur only at R or currently active L cells; the recipe's plate/cooked flags equal zero. |
| 35 | Displayed target recipe: cooked flag | These values occur only at R or currently active L cells; the recipe's plate/cooked flags equal zero. |
| 36 | Displayed target recipe: ingredient 0 count | These values occur only at R or currently active L cells; the recipe's plate/cooked flags equal zero. |
| 37 | Displayed target recipe: ingredient 1 count | These values occur only at R or currently active L cells; the recipe's plate/cooked flags equal zero. |
| 38 | Displayed target recipe: ingredient 2 count | These values occur only at R or currently active L cells; the recipe's plate/cooked flags equal zero. |
| 39 | Remaining pot cooking time | The countdown appears as a raw integer at pots and 0 elsewhere; it is not normalized. |

All static, position, facing, plate, and cooked flags are binary 0/1. The ingredient channels use counts (0–3 natively), not one-hot indicators of recipe class. For this configuration, every item/recipe block stores 5 entries: plate flag followed by cooked flag and counts by ingredient ID; generally, block width is 2 + number of ingredient types.
An empty item has an all-zero block. Empty plates use their plate flag alone, raw ingredients use one ingredient count, and plated cooked dishes use both flags and their counts. Cooked pots encode counts and the cooked flag rather than the plate flag. Recipe displays encode only ingredient counts, leaving both flags zero.

Floor is not assigned a dedicated static channel. A cell within the known map is floor when neither static-object nor ingredient-supply flags are present. Grid-item blocks show counter and pot contents, and player inventory blocks are the sole location of held contents. Recipe displays are distinct from actual items on their cells. An all-zero recipe block signifies no display, not an empty requested recipe.

No explicit global position, global map state, step counter, teammate memory, previous action, or button countdown is encoded in the array. The policy separately
receives its reset flag, private memory and PRNG; it does not receive reward as an
act-function input. The optional successful-delivery channel is disabled, so no delivery-success bit is present.
The policy, not the observation array, maintains remembered observations and counts elapsed steps.

## Reset and randomness

Each seat spawns uniformly in its own listed connected floor region with occupied cells excluded; the other room is not a possible spawn. Each of the four directions is equally likely for facing. Randomizing positions also randomizes facing, even without an independent orientation override.
Initially nothing is held or stored on counters or in pots. All pot and button timers begin at zero with the buttons inactive. The episode clock starts at zero and the recipe is sampled as
listed above. Correct deliveries resample the recipe uniformly for the subsequent observation, with repetition of the old recipe allowed. Wrong deliveries do not resample. Supply locations and static geometry are fixed throughout. Randomness comes through the configured reset and recipe sampling and the policy's own PRNG, while movement, interactions, timers, and rewards follow the rules above.

Use `config.yaml` to inspect requested settings, `specification.json` to inspect resolved constants, object coordinates, spawn regions, and observation schema, and `layout.md` to inspect the repeated board and legend. These files describe the environment without exposing its executable implementation or policy helpers.


Find numeric dimensions, resolved flags, and array schemas in specification.json.
