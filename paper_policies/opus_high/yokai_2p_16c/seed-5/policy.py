"""Cooperative 16-card Yokai policy.

Coordination idea (nothing here requires the partner to run this same code):

* Home columns.  Colour ``c`` belongs in grid column ``x = 3 + c``.  The reset
  block already occupies columns 3..6 with four cards each and colours carry a
  canonical index in the observation, so this target is computable by any seat
  without communication and keeps the whole board connected by construction.
* Complementary scouting.  ``node_features[:, 11]`` publicly marks the cards the
  other seat has observed, so each seat prefers cards that neither seat has
  seen.  Two seats following that preference cover disjoint cards.
* Hints as public colour labels.  A revealed hint is placed only on a card whose
  colour this seat knows and that already sits in a settled cell of its home
  column.  Such a placement is certainly correct, locks a finished card against
  later damage, and its public colour set is evidence for the partner.  Narrow
  hint sets are revealed first, and a placed singleton hint is read back as a
  public colour observation.
* Ending.  Terminal reward is ``35 - 3R - 4C - 6I``, so each extra turn costs
  two to three points.  The seat ends the game as soon as the board must be
  solved: every card is known to this seat, seen by the partner, or publicly
  labelled by a consistent hint; each home column holds exactly four cards; and
  the colouring implied by known colours plus column membership is connected.

Move choice is a gain: every card/destination pair is scored by the improvement
in a static cell-quality function, so bringing a card home, compacting a colour
run and balancing column occupancy all fall out of one comparison.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'column_sorter',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 4,
    'deck_size': 16,
    'grid_size': 10,
    'num_hints': 7,
    'num_nodes': 23,
    'node_features': 15,
    'num_actions': 2325,
    'policy_structure': 'shared_local',
}

G = 10
NCARD = 16
NHINT = 7
NCOLOR = 4
NACT = 2325

# Fixed by the reset rule: hints 0,1 carry one colour, 2,3,4 two, 5,6 three.
HINT_SIZE_LIST = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]

W_HOME = 1.0e6      # a card whose colour this seat knows sits in its home column
W_STAY = 3.0e5      # an unobserved card is presumed to belong where it stands
W_LEAVE = 1.0e5     # ... unless its column already holds four cards of its own
W_CONN = 5.0e4      # size of the colour group the card belongs to (capped at 4)
W_OFFBAND = 2.0e4
W_BALANCE = 3.0e4
W_VERT = 250.0
W_OBS_NEW = 100.0     # prefer looking at a card whose colour this seat lacks
W_OBS_UNSEEN = 50.0   # ... that the partner has not looked at either
W_OBS_WRONG = 30.0    # ... and that is provably in the wrong column
W_REVEAL_SIZE = 0.0  # reveal wide colour sets first: they are easier to use
W_PLACE_SEEN = 0.0   # prefer locking a card the partner has already seen
W_OBS_CROWD = 20.0    # ... or sits in a column that does not hold four cards
W_DEG = 400.0       # cells with two occupied neighbours create no cut vertex
# Canonical rows for a finished column: 2..5.  Parking above the reset block
# and emptying row 6 last makes the four cards of a column end up in one run
# without ever needing a separate compaction move.
ROW_PREF = [-300.0, -100.0, 250.0, 300.0, 300.0, 280.0, 60.0, -150.0,
            -300.0, -400.0]

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _shift(grid, dx, dy, fill):
    """Return out with out[..., x, y] == grid[..., x + dx, y + dy]."""
    out = jnp.roll(grid, shift=(-dx, -dy), axis=(-2, -1))
    if dx > 0:
        out = out.at[..., G - dx:, :].set(fill)
    elif dx < 0:
        out = out.at[..., :-dx, :].set(fill)
    if dy > 0:
        out = out.at[..., :, G - dy:].set(fill)
    elif dy < 0:
        out = out.at[..., :, :-dy].set(fill)
    return out


def init_agent():
    return jnp.full((NCARD,), -1, dtype=jnp.int8)


def agent_act(obs, reset, memory, rng):
    xg = jnp.arange(G, dtype=jnp.int32)[:, None] * jnp.ones((1, G), jnp.int32)
    yg = jnp.ones((G, 1), jnp.int32) * jnp.arange(G, dtype=jnp.int32)[None, :]
    canon = (yg >= 2) & (yg <= 5)
    band = (xg >= 3) & (xg <= 6)
    colcolor = jnp.clip(xg - 3, 0, NCOLOR - 1)
    colors = jnp.arange(NCOLOR, dtype=jnp.int32)
    hint_size = jnp.asarray(HINT_SIZE_LIST, dtype=jnp.float32)

    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions

    card = nf[:NCARD]
    hint = nf[NCARD:]
    own_onehot = card[:, 0:NCOLOR]
    seen_now = own_onehot.sum(axis=1) > 0
    color_now = jnp.argmax(own_onehot, axis=1)
    partner_seen = card[:, 11] > 0
    locked = card[:, 10] > 0
    active = nf[0, 12] > 0

    hint_set = hint[:, 4:8]
    hint_pos = hint[:, 8:10]
    hint_placed = hint[:, 10] > 0
    hint_len = hint_set.sum(axis=1)

    px = pos[:, 0]
    py = pos[:, 1]

    # ---- public evidence carried by placed hints --------------------------
    on_card = (hint_placed[:, None]
               & (hint_pos[:, 0:1] == px[None, :])
               & (hint_pos[:, 1:2] == py[None, :]))          # [NHINT, NCARD]
    single = hint_placed & (hint_len == 1)
    hint_color = jnp.argmax(hint_set, axis=1)
    pub_color = jnp.max(jnp.where(on_card & single[:, None],
                                  hint_color[:, None], -1), axis=0)

    # ---- private colour memory -------------------------------------------
    fresh = jnp.full((NCARD,), -1, dtype=jnp.int8)
    prev = jnp.where(reset, fresh, memory)
    private = jnp.where(seen_now, color_now.astype(jnp.int8), prev)
    known = jnp.where(private >= 0, private, pub_color.astype(jnp.int8))
    kn = known >= 0
    kc = jnp.clip(known.astype(jnp.int32), 0, NCOLOR - 1)
    home = 3 + kc

    # ---- board summaries --------------------------------------------------
    occ = jnp.zeros((G, G), jnp.int32).at[px, py].add(1)
    occb = occ > 0
    colcount = occ.sum(axis=1)                       # cards per grid column
    cgrid = jnp.full((G, G), -1, jnp.int32).at[px, py].set(known.astype(jnp.int32))
    lgrid = jnp.zeros((G, G), bool).at[px, py].set(locked)

    vert_adj = _shift(occb, 0, -1, False) | _shift(occb, 0, 1, False)
    degree = sum(_shift(occb, dx, dy, False).astype(jnp.float32)
                 for dx, dy in DIRS)
    adj_self = (jnp.abs(xg[None] - px[:, None, None])
                + jnp.abs(yg[None] - py[:, None, None])) == 1

    # Working colouring: observed colour where known, column membership
    # otherwise.  This is exactly the colouring the end-game test believes in,
    # so hill-climbing on it drives the board towards a terminable state.
    est = jnp.where(kn, kc, colcolor[px, py])
    adjacent = (jnp.abs(px[:, None] - px[None, :])
                + jnp.abs(py[:, None] - py[None, :])) == 1
    same = est[:, None] == est[None, :]
    reach = ((adjacent & same) | jnp.eye(NCARD, dtype=bool)).astype(jnp.float32)
    for _ in range(4):
        reach = (reach @ reach > 0).astype(jnp.float32)
    comp_size = jnp.minimum(reach.sum(axis=1), 4.0)               # [NCARD]

    estgrid = jnp.full((G, G), -1, jnp.int32).at[px, py].set(est)
    compgrid = jnp.zeros((G, G), jnp.float32).at[px, py].set(comp_size)
    per_color = ((estgrid[None] == colors[:, None, None]).astype(jnp.float32)
                 * compgrid[None])                                # [NCOLOR, G, G]
    nbsum = sum(_shift(per_color, dx, dy, 0.0) for dx, dy in DIRS)
    merge_known = jnp.take(nbsum, kc, axis=0)                     # [NCARD, G, G]
    merge_any = jnp.take_along_axis(nbsum, colcolor[None], axis=0)[0]
    merge = jnp.where(kn[:, None, None], merge_known, merge_any[None])
    # discount the moving card's own component when the target abuts its origin
    own_here = jnp.where(kn[:, None, None], True,
                         xg[None] == px[:, None, None])
    merge = merge - (adj_self & own_here).astype(jnp.float32) * comp_size[:, None, None]
    merge = jnp.clip(1.0 + merge, 1.0, 4.0)

    # A column already holding four cards this seat knows to be its colour
    # cannot host anything else, so its remaining cards are certainly wrong.
    at_home_known = (kn & (px == home)).astype(jnp.float32)
    khome = jnp.zeros((G,), jnp.float32).at[px].add(at_home_known)
    must_leave = (~kn) & (khome[px] >= 4.0)
    room = ((khome < 4.0)[:, None] & band).astype(jnp.float32)

    # ---- substep 0 / 1: which card to observe -----------------------------
    obs_score = (W_OBS_NEW * (~kn).astype(jnp.float32)
                 + W_OBS_UNSEEN * (~partner_seen).astype(jnp.float32)
                 + W_OBS_WRONG * (must_leave & (~kn)).astype(jnp.float32)
                 + W_OBS_CROWD * (colcount[px] != 4).astype(jnp.float32)
                 - 0.1 * jnp.arange(NCARD, dtype=jnp.float32))

    # ---- substep 2: cell quality and the gain of each move ----------------
    cellq = (jnp.asarray(ROW_PREF, dtype=jnp.float32)[yg]
             + W_VERT * vert_adj.astype(jnp.float32)
             + W_DEG * jnp.minimum(degree, 2.0)
             - W_OFFBAND * (~band).astype(jnp.float32))       # [G, G]
    belongs = jnp.where(
        kn[:, None, None],
        W_HOME * (xg[None] == home[:, None, None]),
        jnp.where(must_leave[:, None, None],
                  W_LEAVE * room[None] * (xg[None] != px[:, None, None]),
                  W_STAY * (xg[None] == px[:, None, None])))
    quality = cellq[None] + W_CONN * merge + belongs
    cur_q = (cellq[px, py] + W_CONN * comp_size
             + jnp.where(kn, W_HOME * (px == home),
                         jnp.where(must_leave, 0.0, W_STAY)))  # [NCARD]

    # column-occupancy balance: |count - 4| should fall
    def imbalance(x_idx, count):
        inband = (x_idx >= 3) & (x_idx <= 6)
        return -jnp.abs(count - 4).astype(jnp.float32) * inband.astype(jnp.float32)

    col_idx = jnp.arange(G, dtype=jnp.int32)
    dest_bal = (imbalance(col_idx, colcount + 1)
                - imbalance(col_idx, colcount))               # [G]
    src_bal = (imbalance(px, colcount[px] - 1)
               - imbalance(px, colcount[px]))                 # [NCARD]
    balance = W_BALANCE * (dest_bal[None, :, None] + src_bal[:, None, None])

    move_cost = (4.0e3 * (~kn).astype(jnp.float32)
                 + 6.0e3 * ((~kn) & partner_seen).astype(jnp.float32))
    move_s = (quality - cur_q[:, None, None] + balance
              - move_cost[:, None, None])

    # ---- substep 3: reveal or place a hint --------------------------------
    reveal_s = (5.0e5 + W_REVEAL_SIZE * hint_size
                - jnp.arange(NHINT, dtype=jnp.float32))

    color_1h = (cgrid[None] == colors[:, None, None])
    col_1h = (colcolor[None] == colors[:, None, None]) & band[None]
    match_known = jnp.tensordot(hint_set, color_1h.astype(jnp.int32),
                                axes=([1], [0])) > 0          # [NHINT, G, G]
    match_col = jnp.tensordot(hint_set, col_1h.astype(jnp.int32),
                              axes=([1], [0])) > 0
    cell_known = cgrid >= 0
    cell_home = cell_known & (cgrid == (xg - 3))
    placeable = occb & (~lgrid)
    comp4 = jnp.zeros((G, G), bool).at[px, py].set(comp_size >= 4.0)
    seen_grid = jnp.zeros((G, G), bool).at[px, py].set(partner_seen)
    # Locking is irreversible, so only lock a card that is both in its home
    # column and part of a finished-looking run.
    anchored = cell_home & ((canon & vert_adj) | comp4)
    safe_place = match_known & anchored[None] & placeable[None]
    guess_place = (match_col & (~cell_known)[None] & placeable[None]
                   & (band & canon & comp4 & seen_grid)[None])

    place_s = (6.0e5 * safe_place + 2.0e4 * guess_place
               - 30.0 * hint_size[:, None, None]
               + (5.0e3 * comp4 + W_PLACE_SEEN * seen_grid + 150.0 * vert_adj
                  + 100.0 * canon
                  - 1.0e4 * (cell_known & (~cell_home)))[None])

    # ---- substep 0: end the game once the board must be solved ------------
    in_band = (px >= 3) & (px <= 6)
    hint_at_est = jnp.take(hint_set, jnp.clip(est, 0, NCOLOR - 1), axis=1)
    hint_ok = jnp.any(on_card & (hint_at_est > 0), axis=0)
    trusted = kn | partner_seen | hint_ok
    connected = jnp.all((~same) | (reach > 0))
    counts = jnp.sum(est[:, None] == colors[None, :], axis=0)
    end_ok = (jnp.all(in_band | kn) & jnp.all(counts == 4) & connected
              & jnp.all(trusted))
    end_s = jnp.where(end_ok, 1.0e9, -1.0e9)

    # ---- assemble ---------------------------------------------------------
    scores = jnp.full((NACT,), -1.0e9, dtype=jnp.float32)
    scores = scores.at[0].set(-1.0e8)
    scores = scores.at[1:17].set(obs_score)
    scores = scores.at[17:1617].set(move_s.reshape(-1))
    scores = scores.at[1617:1624].set(reveal_s)
    scores = scores.at[1624:2324].set(place_s.reshape(-1))
    scores = scores.at[2324].set(end_s)

    action = jnp.argmax(jnp.where(legal, scores, -1.0e10))
    action = jnp.where(active, action, 0).astype(jnp.int32)

    moved_card = jnp.clip((action - 17) // 100, 0, NCARD - 1)
    is_move = (action >= 17) & (action < 1617)
    legal_moves = legal[17:1617].reshape(NCARD, G, G)
    want = (kn & (px != home) & (~locked))[:, None, None] & (xg[None] == home[:, None, None])
    prod_legal = jnp.any(legal_moves & want)
    trace = {
        'known_count': jnp.sum(kn).astype(jnp.int32),
        'misplaced': jnp.sum(kn & (px != home)).astype(jnp.int32),
        'mis_free': jnp.sum(kn & (px != home) & (~locked)).astype(jnp.int32),
        'bad_lock': jnp.sum(kn & (px != home) & locked).astype(jnp.int32),
        'prod_legal': prod_legal.astype(jnp.int32),
        'defect': jnp.sum(comp_size < 4).astype(jnp.int32),
        'colbad': jnp.sum(jnp.abs(colcount[3:7] - 4)).astype(jnp.int32),
        'move_known': jnp.where(is_move, kn[moved_card], False).astype(jnp.int32),
        'move_prod': jnp.where(
            is_move, kn[moved_card] & (px[moved_card] != home[moved_card]),
            False).astype(jnp.int32),
        'trusted_count': jnp.sum(trusted).astype(jnp.int32),
        'in_band': jnp.sum(in_band).astype(jnp.int32),
        'counts_ok': jnp.all(counts == 4).astype(jnp.int32),
        'connected': connected.astype(jnp.int32),
        'end_ok': end_ok.astype(jnp.int32),
        'hints_placed': jnp.sum(hint_placed).astype(jnp.int32),
        'active': active.astype(jnp.int32),
    }
    return action, private, trace
