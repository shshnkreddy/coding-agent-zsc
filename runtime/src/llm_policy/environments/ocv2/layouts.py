"""Custom OvercookedV2 layouts, resolved by name in make_env.

Small reductions of the extended cat-dog problem. In static_cat_dog and
static_cat_dog_matrix both agents are locked in single-cell rooms (blocked
moves still turn the agent), so the game approximates a matrix game with a
cooking pipeline. In static_cat_dog_button_only the speaker remains locked,
while the listener moves between two cells to complete the whole pipeline.
The speaker reads the recipe indicator R; the listener picks between
ingredient piles 0 and 1.

Variants:
- static_cat_dog: grounded channel. The speaker has a button recipe
  indicator L that reveals the recipe to the listener at a reward cost.
  With agent_view_size=2 the listener also sees the speaker, so a free
  direction/inventory cheap-talk channel coexists with the button.
- static_cat_dog2: no button. The delivery goal is directly above the
  speaker, in the tile occupied by L in static_cat_dog; the old delivery tile
  to the speaker's left is a wall.
- static_cat_dog_3: retains the grounded button and original delivery goal,
  but swaps the ingredient piles: dark-green ingredient 1 is above the
  listener and yellow ingredient 0 is below.
- static_cat_dog_3_ingredient: extends static_cat_dog_3 to three recipes and
  three ingredients by replacing the wall immediately right of the listener
  with ingredient pile 2.
- static_cat_dog_matrix: no button. The only channel is the listener
  observing the speaker's direction/inventory (arbitrary conventions).
- static_cat_dog_button_only: line of sight broken. The listener (in a 1x2
  room, doing the full cooking pipeline) sees the button L but not the
  speaker or R. Static, non-depleting plate piles separate the rooms, so the
  button is the only shared mutable object and the only communication channel.
"""

from llm_policy.environments.ocv2.restaurant_layouts import RESTAURANT_LAYOUTS
from llm_policy.environments.ocv2.shared_layouts import SHARED_LAYOUTS


_STATIC_CAT_DOG = """
RLW0W
XAPAW
WBW1W
"""

_STATIC_CAT_DOG2 = """
RXW0W
WAPAW
WBW1W
"""

_STATIC_CAT_DOG_3 = """
RLW1W
XAPAW
WBW0W
"""

_STATIC_CAT_DOG_3_INGREDIENT = """
RLW1W
XAPA2
WBW0W
"""

_STATIC_CAT_DOG_MATRIX = """
RWW0W
XAPAW
WBW1W
"""

_STATIC_CAT_DOG_BUTTON_ONLY = """
WWWW0WW
RALBA1W
WWWB PW
WWWWXWW
"""

# Exact geometry and agent ordering from JaxMARL's legacy ``overcooked``
# layouts. Upstream OvercookedV2 reverses agent identities for cramped_room
# and counter_circuit, whereas legacy forced_coord has the reverse of ASCII
# scan order. Resolving the layouts here gives each one its legacy agent order
# so directed agent_0/agent_1 cross-play remains aligned too.
_LEGACY_CRAMPED_ROOM = """
WWPWW
OA AO
W   W
WBWXW
"""

_LEGACY_ASYMM_ADVANTAGES = """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
"""

_LEGACY_COORD_RING = """
WWWPW
W A P
BAW W
O   W
WOXWW
"""

_LEGACY_FORCED_COORD = """
WWWPW
O WAP
OAW W
B W W
WWWXW
"""

_LEGACY_COUNTER_CIRCUIT = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""

_CAT_DOG_RECIPES = [[0, 0, 0], [1, 1, 1]]
_CAT_DOG_3_INGREDIENT_RECIPES = [
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2],
]
_STATIC_CAT_DOG_DEFAULT_MAX_STEPS = 50
_STATIC_CAT_DOG_BUTTON_ONLY_DEFAULT_MAX_STEPS = 400
_LEGACY_OVERCOOKED_DEFAULT_MAX_STEPS = 400
_LEGACY_OVERCOOKED_RECIPES = [[0, 0, 0]]
_LEGACY_OVERCOOKED_LAYOUT_NAMES = {
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "forced_coord",
    "counter_circuit",
}
_LEGACY_OVERCOOKED_SWAP_AGENT_NAMES = {"forced_coord"}

# name -> (ascii grid, possible_recipes)
CUSTOM_OVERCOOKED_V2_LAYOUTS = {
    **RESTAURANT_LAYOUTS,
    **SHARED_LAYOUTS,
    "cramped_room": (_LEGACY_CRAMPED_ROOM, _LEGACY_OVERCOOKED_RECIPES),
    "asymm_advantages": (
        _LEGACY_ASYMM_ADVANTAGES,
        _LEGACY_OVERCOOKED_RECIPES,
    ),
    "coord_ring": (_LEGACY_COORD_RING, _LEGACY_OVERCOOKED_RECIPES),
    "forced_coord": (_LEGACY_FORCED_COORD, _LEGACY_OVERCOOKED_RECIPES),
    "counter_circuit": (
        _LEGACY_COUNTER_CIRCUIT,
        _LEGACY_OVERCOOKED_RECIPES,
    ),
    "static_cat_dog": (_STATIC_CAT_DOG, _CAT_DOG_RECIPES),
    "static_cat_dog2": (_STATIC_CAT_DOG2, _CAT_DOG_RECIPES),
    "static_cat_dog_3": (_STATIC_CAT_DOG_3, _CAT_DOG_RECIPES),
    "static_cat_dog_3_ingredient": (
        _STATIC_CAT_DOG_3_INGREDIENT,
        _CAT_DOG_3_INGREDIENT_RECIPES,
    ),
    "static_cat_dog_matrix": (_STATIC_CAT_DOG_MATRIX, _CAT_DOG_RECIPES),
    "static_cat_dog_button_only": (_STATIC_CAT_DOG_BUTTON_ONLY, _CAT_DOG_RECIPES),
}


def resolve_custom_layout(env_kwargs):
    """Replace a custom layout name in env_kwargs with a jaxmarl Layout object.

    Unknown or non-string layouts are passed through untouched so the built-in
    jaxmarl layout names keep working.
    """
    layout = env_kwargs.get("layout")
    if not isinstance(layout, str) or layout not in CUSTOM_OVERCOOKED_V2_LAYOUTS:
        return env_kwargs

    from jaxmarl.environments.overcooked_v2.layouts import Layout

    grid, default_recipes = CUSTOM_OVERCOOKED_V2_LAYOUTS[layout]
    resolved = dict(env_kwargs)
    # Upstream accepts recipes only as part of the Layout object.  Consume an
    # optional experiment override here without duplicating the ASCII layout.
    # Layout.from_string hard-rejects non-three-item recipes, so parse the grid
    # with its default recipes and then construct a Layout with the validated
    # reduced recipes.
    recipe_override = resolved.pop("possible_recipes", None)
    parsed_layout = Layout.from_string(
        grid,
        possible_recipes=default_recipes,
        swap_agents=layout in _LEGACY_OVERCOOKED_SWAP_AGENT_NAMES,
    )
    if recipe_override is None:
        resolved["layout"] = parsed_layout
    else:
        recipes = [list(recipe) for recipe in recipe_override]
        recipe_lengths = {len(recipe) for recipe in recipes}
        if not recipes or len(recipe_lengths) != 1:
            raise ValueError(
                "possible_recipes must be a non-empty list of equally sized "
                "recipes."
            )
        recipe_size = next(iter(recipe_lengths))
        if recipe_size < 1 or recipe_size > 3:
            raise ValueError(
                "possible_recipes entries must contain between 1 and 3 "
                f"ingredients, got {recipe_size}."
            )
        if any(
            ingredient < 0 or ingredient >= parsed_layout.num_ingredients
            for recipe in recipes
            for ingredient in recipe
        ):
            raise ValueError(
                "possible_recipes contains an ingredient without a matching "
                "pile in the layout."
            )
        resolved["layout"] = Layout(
            agent_positions=parsed_layout.agent_positions,
            static_objects=parsed_layout.static_objects,
            num_ingredients=parsed_layout.num_ingredients,
            possible_recipes=recipes,
        )
    if layout in _LEGACY_OVERCOOKED_LAYOUT_NAMES:
        default_max_steps = _LEGACY_OVERCOOKED_DEFAULT_MAX_STEPS
    elif layout in RESTAURANT_LAYOUTS or layout in SHARED_LAYOUTS:
        default_max_steps = 400
    elif layout == "static_cat_dog_button_only":
        default_max_steps = _STATIC_CAT_DOG_BUTTON_ONLY_DEFAULT_MAX_STEPS
    else:
        default_max_steps = _STATIC_CAT_DOG_DEFAULT_MAX_STEPS
    resolved.setdefault("max_steps", default_max_steps)
    return resolved


def fix_overcooked_v2_obs_shape(env):
    """Correct OvercookedV2.obs_shape in place on jaxmarl versions where
    _get_obs_shape disagrees with the layers get_obs_default actually builds.

    PyPI jaxmarl 0.1.0 clamps the view window to the grid size but still
    slices from the agent position (off-center window on grids smaller than
    2r+1), and its channel formula is short by num_ingredients-1, so the
    obs slice silently drops the trailing layers (pot timer, and for 3+
    ingredients the highest-ingredient recipe channel).

    The layer-building and masking code is shape-driven and identical across
    versions, so claiming the correct shape restores a centered
    (2r+1)x(2r+1) window with all channels. No-op on fixed versions.
    Only applies to the default observation type.
    """
    obs_type = getattr(env, "observation_type", "default")
    if not isinstance(obs_type, str) or obs_type != "default":
        return env

    num_ing = env.layout.num_ingredients
    num_layers = 17 + num_ing + 4 * (num_ing + 2)
    if getattr(env, "indicate_successful_delivery", False):
        num_layers += 1

    if env.agent_view_size:
        view = 2 * env.agent_view_size + 1
        correct = (view, view, num_layers)
    else:
        correct = (env.height, env.width, num_layers)

    if tuple(env.obs_shape) != correct:
        env.obs_shape = correct
    return env


if __name__ == "__main__":
    # Scripted verification of static_cat_dog: information asymmetry, the
    # grounded button channel, movement lock, and the full cook/deliver
    # pipeline. Version-agnostic w.r.t. the installed jaxmarl: works whether
    # agent_view_size crops the obs to a (2r+1)x(2r+1) window or masks the
    # full grid in place. Run: python src/envs/overcooked_v2_layouts.py
    import jax
    import jax.numpy as jnp
    import jaxmarl
    from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject
    from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

    print(f"jaxmarl: {jaxmarl.__file__}")

    VIEW_R = 2
    kwargs = resolve_custom_layout(
        {"layout": "static_cat_dog", "agent_view_size": VIEW_R, "negative_rewards": True}
    )
    env = fix_overcooked_v2_obs_shape(OvercookedV2(**kwargs))
    H, W = env.height, env.width

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)
    print(f"obs shape: {obs['agent_0'].shape} (full grid is ({H}, {W}))")
    num_layers = 17 + env.layout.num_ingredients + 4 * (env.layout.num_ingredients + 2)
    assert obs["agent_0"].shape == (2 * VIEW_R + 1, 2 * VIEW_R + 1, num_layers), (
        "fix_overcooked_v2_obs_shape did not restore the centered full-channel window"
    )

    # Absolute geometry: speaker agent_0 at (1,1), listener agent_1 at (3,1),
    # R at (0,0), L at (1,0), pot at (2,1) -> grid[1, 2].
    SPEAKER, LISTENER = (1, 1), (3, 1)

    # Decode the sampled recipe: ingredient i occupies 2 bits from bit 2+2i.
    num_ing = env.layout.num_ingredients
    counts = [(int(state.recipe) >> (2 * (i + 1))) & 0x3 for i in range(num_ing)]
    recipe_idx = counts.index(3)
    print(f"sampled recipe: 3x ingredient {recipe_idx}")

    # Locate the recipe-count channel AND the view-window offset empirically:
    # at reset, all inventories/counters/pots are empty, so the only obs value
    # of 3 anywhere is the recipe count (recipes here are pure 3x one
    # ingredient) at the R cell as seen by the speaker. Different jaxmarl
    # versions slice the view window differently (centered crop vs clamped to
    # the grid, e.g. 0.1.0 clamps to [y-r, y] rows on this 3-row layout), but
    # all slice relative to the agent position, so one detection calibrates
    # the mapping for every agent.
    hits = jnp.argwhere(obs["agent_0"] == 3)
    assert hits.shape[0] == 1, (
        f"expected exactly one obs value of 3 (recipe count at R) in the "
        f"speaker's obs at reset, found {hits.shape[0]} — speaker cannot "
        f"read R, or this jaxmarl version encodes recipes differently"
    )
    r_row, r_col, recipe_ch = (int(v) for v in hits[0])
    # R is at absolute (x=0, y=0); speaker at SPEAKER. offset = obs_idx - (cell - pos)
    c_row = r_row - (0 - SPEAKER[1])
    c_col = r_col - (0 - SPEAKER[0])
    print(f"recipe-count channel: {recipe_ch}, window offset: ({c_row}, {c_col})")

    def cell_index(obs_a, agent_pos, cell_xy):
        """Map an absolute grid cell to (row, col) in this agent's obs frame."""
        row = cell_xy[1] - agent_pos[1] + c_row
        col = cell_xy[0] - agent_pos[0] + c_col
        assert 0 <= row < obs_a.shape[0] and 0 <= col < obs_a.shape[1], (
            f"cell {cell_xy} falls outside the obs window of agent at "
            f"{agent_pos} — view geometry differs from the expected design"
        )
        return row, col

    # 1. Information asymmetry at reset: listener must see no recipe info.
    assert not bool((obs["agent_1"] == 3).any()), (
        "listener can see the recipe at reset — no information asymmetry; "
        "check agent_view_size handling in the installed jaxmarl"
    )
    print("PASS: speaker sees R, listener sees no recipe info")

    total_reward = 0.0

    def step(a0, a1):
        global key, state, total_reward
        key, subkey = jax.random.split(key)
        actions = {"agent_0": jnp.array(int(a0)), "agent_1": jnp.array(int(a1))}
        obs, state, rewards, dones, info = env.step(subkey, state, actions)
        assert not bool(dones["__all__"]), "episode ended mid-test"
        total_reward += float(rewards["agent_0"])
        return obs, float(rewards["agent_0"])

    # 2. Movement lock: both agents boxed in, positions never change.
    for act in (Actions.right, Actions.down, Actions.left, Actions.up):
        step(act, act)
        assert (state.agents.pos.x == jnp.array([1, 3])).all()
        assert (state.agents.pos.y == jnp.array([1, 1])).all()
    print("PASS: agents are locked in place")

    # 3. Button press: -5, recipe becomes visible at the listener's L cell.
    obs, reward = step(Actions.interact, Actions.stay)  # speaker faces up after (2)
    assert reward == -5.0, f"expected -5 button cost, got {reward}"
    li_l = cell_index(obs["agent_1"], LISTENER, (1, 0))
    assert int(obs["agent_1"][li_l[0], li_l[1], recipe_ch]) == 3, (
        "listener cannot read the activated button — grounded signal broken"
    )
    print("PASS: button press costs 5 and reveals the recipe to the listener")

    # 4. Listener fills the pot (3x pick + place), speaker fetches a plate.
    ing_act = Actions.up if recipe_idx == 0 else Actions.down
    speaker_script = [Actions.down, Actions.interact]
    listener_script = [ing_act, Actions.interact, Actions.left, Actions.interact] * 3
    for i in range(len(listener_script)):
        a0 = speaker_script[i] if i < len(speaker_script) else Actions.stay
        obs, _ = step(a0, listener_script[i])
    assert int(state.grid[1, 2, 1]) == int(state.recipe), "pot contents != recipe"
    assert int(state.grid[1, 2, 2]) > 0, "pot did not auto-start cooking"
    assert int(state.agents.inventory[0]) == int(DynamicObject.PLATE)
    # pot-timer layer (the channel jaxmarl 0.1.0 silently drops) is visible:
    sp_pot = cell_index(obs["agent_0"], SPEAKER, (2, 1))
    assert int(obs["agent_0"][sp_pot[0], sp_pot[1], -1]) == int(state.grid[1, 2, 2]), (
        "pot-timer channel missing from obs — obs shape fix not applied?"
    )
    print("PASS: pot cooking with the correct recipe, timer visible, speaker holds a plate")

    # 5. Wait out the cook, then plate and deliver.
    while int(state.grid[1, 2, 2]) > 0:
        obs, _ = step(Actions.stay, Actions.stay)
    assert int(state.grid[1, 2, 1]) & int(DynamicObject.COOKED)
    # button activation (~10 steps) has expired again by now
    assert int(obs["agent_1"][li_l[0], li_l[1], recipe_ch]) == 0, (
        "button indicator did not expire"
    )
    step(Actions.right, Actions.stay)
    step(Actions.interact, Actions.stay)
    plated = int(state.recipe) | int(DynamicObject.COOKED) | int(DynamicObject.PLATE)
    assert int(state.agents.inventory[0]) == plated, "dish pickup failed"
    step(Actions.left, Actions.stay)
    obs, reward = step(Actions.interact, Actions.stay)
    assert reward == 20.0, f"expected +20 correct delivery, got {reward}"
    assert total_reward == 15.0, f"expected net +15 (=20-5), got {total_reward}"
    print(f"PASS: correct delivery, net episode reward {total_reward}")
    print("STATIC_CAT_DOG CHECKS PASSED")

    # Verify the exact production configuration of static_cat_dog_button_only.
    # In particular, random_agent_positions keeps each agent in its original
    # enclosed room, but places the listener at either end of its 1x2 room and
    # randomizes both directions. Cover both listener starts and both recipes.
    print("\nVerifying static_cat_dog_button_only (production settings)")
    button_kwargs = resolve_custom_layout(
        {
            "layout": "static_cat_dog_button_only",
            "agent_view_size": VIEW_R,
            "negative_rewards": True,
            "sample_recipe_on_delivery": True,
            "indicate_successful_delivery": True,
            "random_agent_positions": True,
            "max_steps": _STATIC_CAT_DOG_BUTTON_ONLY_DEFAULT_MAX_STEPS,
        }
    )
    button_env = fix_overcooked_v2_obs_shape(OvercookedV2(**button_kwargs))

    BUTTON = (2, 1)
    RECIPE_INDICATOR = (0, 1)
    SPEAKER = (1, 1)
    LISTENER_X = 4
    LISTENER_TOP_Y = 1
    LISTENER_BOTTOM_Y = 2
    VISIBLE_PLATE_BARRIER = (3, 1)
    POT = (5, 2)
    GOAL = (4, 3)

    button_num_layers = (
        17
        + button_env.layout.num_ingredients
        + 4 * (button_env.layout.num_ingredients + 2)
        + 1  # indicate_successful_delivery
    )
    expected_shape = (2 * VIEW_R + 1, 2 * VIEW_R + 1, button_num_layers)
    assert button_env.max_steps == 400
    assert button_env.obs_shape == expected_shape

    def decode_recipe(recipe):
        counts = [
            (int(recipe) >> (2 * (i + 1))) & 0x3
            for i in range(button_env.layout.num_ingredients)
        ]
        assert sorted(counts) == [0, 3], f"unexpected recipe counts: {counts}"
        return counts.index(3)

    # Find deterministic reset seeds for all four combinations instead of
    # assuming how JAX maps a particular seed to a recipe or start cell.
    reset_cases = {}
    for reset_seed in range(1024):
        candidate_obs, candidate_state = button_env.reset(
            jax.random.PRNGKey(reset_seed)
        )
        recipe_idx = decode_recipe(candidate_state.recipe)
        listener_x = int(candidate_state.agents.pos.x[1])
        listener_y = int(candidate_state.agents.pos.y[1])
        assert listener_x == LISTENER_X
        assert listener_y in (LISTENER_TOP_Y, LISTENER_BOTTOM_Y)
        reset_cases.setdefault((recipe_idx, listener_y), reset_seed)
        if len(reset_cases) == 4:
            break
    assert len(reset_cases) == 4, (
        "failed to sample both recipes at both listener starts in 1024 resets: "
        f"found {sorted(reset_cases)}"
    )
    print(
        "PASS: random resets cover both recipes and listener starts: "
        + ", ".join(
            f"recipe={recipe_idx}, y={listener_y}, seed={seed}"
            for (recipe_idx, listener_y), seed in sorted(reset_cases.items())
        )
    )

    # The listener can interact with the visible B separator, but plate piles
    # are static and non-depleting. Picking up a plate must therefore leave the
    # speaker's complete observation unchanged: no counter-based side channel.
    barrier_seed = reset_cases[(0, LISTENER_TOP_Y)]
    barrier_obs, barrier_state = button_env.reset(jax.random.PRNGKey(barrier_seed))
    speaker_obs_before_barrier_use = barrier_obs["agent_0"]
    for barrier_step, listener_action in enumerate(
        (Actions.left, Actions.interact), start=1
    ):
        barrier_obs, barrier_state, rewards, dones, info = button_env.step(
            jax.random.fold_in(jax.random.PRNGKey(barrier_seed), barrier_step),
            barrier_state,
            {
                "agent_0": jnp.array(int(Actions.stay)),
                "agent_1": jnp.array(int(listener_action)),
            },
        )
        assert not bool(dones["__all__"])
        assert float(rewards["agent_0"]) == 0.0
        assert float(rewards["agent_1"]) == 0.0
        assert bool(
            jnp.array_equal(
                barrier_obs["agent_0"], speaker_obs_before_barrier_use
            )
        ), "listener's use of the plate barrier changed the speaker observation"
    assert int(barrier_state.agents.inventory[1]) == int(DynamicObject.PLATE)
    assert int(
        barrier_state.grid[
            VISIBLE_PLATE_BARRIER[1], VISIBLE_PLATE_BARRIER[0], 1
        ]
    ) == 0
    print("PASS: visible plate barrier provides no listener-to-speaker signal")

    def run_button_only_case(expected_recipe_idx, expected_listener_y, reset_seed):
        obs, state = button_env.reset(jax.random.PRNGKey(reset_seed))
        recipe_idx = decode_recipe(state.recipe)
        assert recipe_idx == expected_recipe_idx
        assert tuple(int(v) for v in state.agents.pos.x) == (SPEAKER[0], LISTENER_X)
        assert tuple(int(v) for v in state.agents.pos.y) == (
            SPEAKER[1],
            expected_listener_y,
        )
        assert obs["agent_0"].shape == expected_shape
        assert obs["agent_1"].shape == expected_shape

        # Locate the active recipe-count channel and the observation-window
        # origin from R in the speaker's initial observation. At reset, R is
        # the only cell containing a count of three.
        recipe_hits = jnp.argwhere(obs["agent_0"] == 3)
        assert recipe_hits.shape[0] == 1, (
            "speaker must see exactly one recipe count at R; found "
            f"{recipe_hits.shape[0]} values equal to three"
        )
        r_row, r_col, recipe_ch = (int(v) for v in recipe_hits[0])
        center_row = r_row - (RECIPE_INDICATOR[1] - SPEAKER[1])
        center_col = r_col - (RECIPE_INDICATOR[0] - SPEAKER[0])

        def observed_cell(agent_idx, cell_xy, channel):
            agent_x = int(state.agents.pos.x[agent_idx])
            agent_y = int(state.agents.pos.y[agent_idx])
            row = cell_xy[1] - agent_y + center_row
            col = cell_xy[0] - agent_x + center_col
            agent_obs = obs[f"agent_{agent_idx}"]
            assert 0 <= row < agent_obs.shape[0]
            assert 0 <= col < agent_obs.shape[1]
            return int(agent_obs[row, col, channel])

        assert observed_cell(0, RECIPE_INDICATOR, recipe_ch) == 3
        assert observed_cell(1, BUTTON, recipe_ch) == 0
        assert not bool((obs["agent_1"] == 3).any()), (
            "listener sees recipe information before the button is pressed"
        )

        step_count = 0
        total_raw_reward = 0.0
        button_activated = False
        saw_button_expire = False

        def assert_rooms():
            assert (int(state.agents.pos.x[0]), int(state.agents.pos.y[0])) == SPEAKER
            assert int(state.agents.pos.x[1]) == LISTENER_X
            assert int(state.agents.pos.y[1]) in (
                LISTENER_TOP_Y,
                LISTENER_BOTTOM_Y,
            )

        def assert_button_observation():
            nonlocal saw_button_expire
            timer = int(state.grid[BUTTON[1], BUTTON[0], 2])
            observed_recipe_count = observed_cell(1, BUTTON, recipe_ch)
            expected_count = 3 if timer > 0 else 0
            assert observed_recipe_count == expected_count, (
                "button observation disagrees with its timer: "
                f"timer={timer}, observed={observed_recipe_count}"
            )
            if timer == 0:
                saw_button_expire = True

        def step(
            speaker_action,
            listener_action,
            *,
            raw_reward=0.0,
            shaped_rewards=(0.0, 0.0),
        ):
            nonlocal obs, state, step_count, total_raw_reward
            step_key = jax.random.fold_in(
                jax.random.PRNGKey(reset_seed), step_count + 1
            )
            actions = {
                "agent_0": jnp.array(int(speaker_action)),
                "agent_1": jnp.array(int(listener_action)),
            }
            obs, state, rewards, dones, info = button_env.step(
                step_key, state, actions
            )
            step_count += 1
            assert int(state.time) == step_count
            assert not bool(dones["__all__"]), (
                f"episode terminated at step {step_count} before delivery"
            )
            for agent in button_env.agents:
                assert float(rewards[agent]) == raw_reward, (
                    f"{agent} raw reward {float(rewards[agent])}, "
                    f"expected {raw_reward} at step {step_count}"
                )
            actual_shaped = tuple(
                float(info["shaped_reward"][agent]) for agent in button_env.agents
            )
            assert actual_shaped == shaped_rewards, (
                f"shaped rewards {actual_shaped}, expected {shaped_rewards} "
                f"at step {step_count}"
            )
            total_raw_reward += float(rewards["agent_0"])
            assert_rooms()
            if button_activated:
                assert_button_observation()

        # A direction action turns the speaker toward the blocked L cell; the
        # following interaction must activate it without moving the speaker.
        step(Actions.right, Actions.stay)
        step(
            Actions.interact,
            Actions.stay,
            raw_reward=-5.0,
            shaped_rewards=(0.0, 0.0),
        )
        button_activated = True
        assert int(state.grid[BUTTON[1], BUTTON[0], 2]) == 9, (
            "button should have nine visible steps remaining immediately after "
            "activation (the environment decrements it in the activation step)"
        )
        assert_button_observation()
        assert observed_cell(1, BUTTON, recipe_ch) == 3

        # The listener may start next to the plate/pot or next to the
        # ingredient piles. Normalize to the top cell before cooking.
        if int(state.agents.pos.y[1]) == LISTENER_BOTTOM_Y:
            step(Actions.stay, Actions.up)
        assert int(state.agents.pos.y[1]) == LISTENER_TOP_Y

        ingredient_action = Actions.up if recipe_idx == 0 else Actions.right
        ingredient_encoding = 1 << (2 * (recipe_idx + 1))
        for ingredient_number in range(3):
            step(Actions.stay, ingredient_action)
            step(Actions.stay, Actions.interact)
            assert int(state.agents.inventory[1]) == ingredient_encoding

            step(Actions.stay, Actions.down)
            assert int(state.agents.pos.y[1]) == LISTENER_BOTTOM_Y
            step(Actions.stay, Actions.right)
            step(
                Actions.stay,
                Actions.interact,
                shaped_rewards=(0.0, 3.0),
            )
            assert int(state.agents.inventory[1]) == 0

            if ingredient_number < 2:
                step(Actions.stay, Actions.up)
                assert int(state.agents.pos.y[1]) == LISTENER_TOP_Y

        assert int(state.grid[POT[1], POT[0], 1]) == int(state.recipe), (
            "listener placed ingredients that do not match the sampled recipe"
        )
        assert int(state.grid[POT[1], POT[0], 2]) > 0, (
            "pot did not begin cooking automatically after the third ingredient"
        )

        # Pick up the plate while the pot cooks. This is useful according to
        # OvercookedV2's shaping rules because the pot is non-empty.
        step(Actions.stay, Actions.left)
        step(
            Actions.stay,
            Actions.interact,
            shaped_rewards=(0.0, 3.0),
        )
        assert int(state.agents.inventory[1]) == int(DynamicObject.PLATE)

        while int(state.grid[POT[1], POT[0], 2]) > 0:
            step(Actions.stay, Actions.stay)
        assert saw_button_expire, "button indicator never expired"
        assert int(state.grid[POT[1], POT[0], 1]) == (
            int(state.recipe) | int(DynamicObject.COOKED)
        )

        # Plate the cooked recipe and deliver it from the bottom cell.
        step(Actions.stay, Actions.right)
        step(
            Actions.stay,
            Actions.interact,
            shaped_rewards=(0.0, 5.0),
        )
        plated_recipe = (
            int(state.recipe)
            | int(DynamicObject.COOKED)
            | int(DynamicObject.PLATE)
        )
        assert int(state.agents.inventory[1]) == plated_recipe

        step(Actions.stay, Actions.down)
        step(
            Actions.stay,
            Actions.interact,
            raw_reward=20.0,
            shaped_rewards=(0.0, 0.0),
        )
        assert bool(state.new_correct_delivery)
        assert int(state.agents.inventory[1]) == 0
        assert observed_cell(1, GOAL, -1) == 1, (
            "listener did not observe the successful-delivery pulse at X"
        )
        assert total_raw_reward == 15.0
        assert step_count <= button_env.max_steps
        assert step_count in (42, 43), (
            f"unexpected scripted trajectory length {step_count}"
        )
        print(
            "PASS: button_only "
            f"recipe={recipe_idx}, listener_y={expected_listener_y}, "
            f"steps={step_count}, raw_return={total_raw_reward}"
        )

    for (recipe_idx, listener_y), reset_seed in sorted(reset_cases.items()):
        run_button_only_case(recipe_idx, listener_y, reset_seed)

    print("ALL OVERCOOKED_V2 LAYOUT CHECKS PASSED")
