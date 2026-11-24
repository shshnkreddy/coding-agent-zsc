"""Joint policy for overcooked_v2 / grounded_coord_ring.

Layout facts that drive the design (see context/environment/README.md):

  * The board has two disjoint floor components.  agent_1 spawns in the inner
    8-cell ring around the pot at (4,4); agent_0 spawns in the outer 24-cell
    corridor.  Only the inner component touches the pot, so only the inner seat
    can cook.  The inner component also touches both ingredient supplies, both
    plate piles and both delivery stations, so the inner seat is self-sufficient.
  * No always-on recipe indicator `R` is ever inside the inner seat's view, so
    the inner seat learns the recipe from an active `L` button display (anyone
    may activate it for -5) or not at all.  The outer seat reads `R` for free
    from (1,4) or (7,4) and can activate the adjacent button from that same cell.

Both act functions run the same code and pick their role from the observation:
seeing the pot means being inside the ring.  Nothing below assumes that the
partner shares this policy, splits roles the same way, or signals in any
particular manner; the only channel used is the recipe display, which always
shows the true current recipe, and every adaptation is driven by observed
partner behaviour rather than by an agreed convention.

Inner seat - self-contained cook, 39 steps per dish (ingredient 0 side; the
ingredient 1 side is mirrored through row 4):

    (3,3) face LEFT  take ingredient -> (3,4) face DOWN add to pot   [x3]
    (3,4) face UP    activate the button when the pot timer reads 5, so the
                     display is still live 7 steps later at the delivery and
                     the freshly sampled recipe is read for free
    (3,5) face RIGHT take plate      -> (4,5) face LEFT  collect the dish
    (4,5) face RIGHT deliver         -> walk back to the supply

It skips its own activation once it has twice seen the partner light the
display, and in the final cycle where no further delivery can complete.  It
never delivers a dish whose contents do not match the recipe it knows.

Outer seat - information and logistics service that costs the team nothing
unless the partner demonstrably needs it:

  * activates the button only after 60 steps in which nobody lit the display
    (the cook's own cadence is about 39 steps), never while a display is live,
    and never again once the partner has been seen lighting it alone;
  * afterwards refreshes on every recipe change and about once per cook cycle,
    follows the cook to the other button when it stays out of view, and stops
    entirely after four activations that are never followed by a delivery;
  * carries a plated dish left on a shared counter to a delivery station when it
    matches the indicator and has been abandoned there for several steps, then
    re-broadcasts the recipe that its own delivery resampled.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "ring_cook",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}

# --------------------------------------------------------------------------
# Literal lookup tables over the 32 floor cells of the fixed layout.
# Floor cell order:
#   0:(1,1)  1:(1,2)  2:(1,3)  3:(1,4)  4:(1,5)  5:(1,6)  6:(1,7)  7:(2,1)
#   8:(2,7)  9:(3,1) 10:(3,3) 11:(3,4) 12:(3,5) 13:(3,7) 14:(4,1) 15:(4,3)
#  16:(4,5) 17:(4,7) 18:(5,1) 19:(5,3) 20:(5,4) 21:(5,5) 22:(5,7) 23:(6,1)
#  24:(6,7) 25:(7,1) 26:(7,2) 27:(7,3) 28:(7,4) 29:(7,5) 30:(7,6) 31:(7,7)
# --------------------------------------------------------------------------

# Object code per cell: 0 floor/off-board, otherwise 1 + static channel offset
# (1 wall, 2 goal, 3 pot, 4 recipe indicator, 5 button, 6 plate pile,
#  7 ingredient 0 supply, 8 ingredient 1 supply, 9 ingredient 2 supply).
# One 5x5 patch per floor cell; all 32 patches are distinct.
_PATCH = [
    [[0, 0, 0, 0, 0], [0, 1, 1, 1, 9], [0, 1, 0, 0, 0], [0, 1, 0, 1, 1], [0, 9, 0, 7, 0]],
    [[0, 0, 0, 0, 0], [1, 1, 1, 9, 4], [1, 0, 0, 0, 0], [1, 0, 1, 1, 5], [9, 0, 7, 0, 0]],
    [[0, 0, 0, 0, 0], [1, 1, 9, 4, 9], [0, 0, 0, 0, 0], [0, 1, 1, 5, 1], [0, 7, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [1, 9, 4, 9, 1], [0, 0, 0, 0, 0], [1, 1, 5, 1, 1], [7, 0, 0, 0, 6]],
    [[0, 0, 0, 0, 0], [9, 4, 9, 1, 1], [0, 0, 0, 0, 0], [1, 5, 1, 1, 0], [0, 0, 0, 6, 0]],
    [[0, 0, 0, 0, 0], [4, 9, 1, 1, 1], [0, 0, 0, 0, 1], [5, 1, 1, 0, 1], [0, 0, 6, 0, 9]],
    [[0, 0, 0, 0, 0], [9, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 0], [0, 6, 0, 9, 0]],
    [[0, 1, 1, 1, 9], [0, 1, 0, 0, 0], [0, 1, 0, 1, 1], [0, 9, 0, 7, 0], [0, 4, 0, 2, 0]],
    [[9, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 0], [0, 6, 0, 9, 0], [0, 2, 0, 4, 0]],
    [[0, 1, 0, 0, 0], [0, 1, 0, 1, 1], [0, 9, 0, 7, 0], [0, 4, 0, 2, 0], [0, 9, 0, 8, 0]],
    [[0, 0, 0, 0, 0], [0, 1, 1, 5, 1], [0, 7, 0, 0, 0], [0, 2, 0, 3, 0], [0, 8, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [1, 1, 5, 1, 1], [7, 0, 0, 0, 6], [2, 0, 3, 0, 2], [8, 0, 0, 0, 6]],
    [[0, 0, 0, 0, 0], [1, 5, 1, 1, 0], [0, 0, 0, 6, 0], [0, 3, 0, 2, 0], [0, 0, 0, 6, 0]],
    [[0, 0, 0, 1, 0], [1, 1, 0, 1, 0], [0, 6, 0, 9, 0], [0, 2, 0, 4, 0], [0, 6, 0, 9, 0]],
    [[0, 1, 0, 1, 1], [0, 9, 0, 7, 0], [0, 4, 0, 2, 0], [0, 9, 0, 8, 0], [0, 1, 0, 1, 1]],
    [[0, 1, 1, 5, 1], [0, 7, 0, 0, 0], [0, 2, 0, 3, 0], [0, 8, 0, 0, 0], [0, 1, 1, 5, 1]],
    [[1, 5, 1, 1, 0], [0, 0, 0, 6, 0], [0, 3, 0, 2, 0], [0, 0, 0, 6, 0], [1, 5, 1, 1, 0]],
    [[1, 1, 0, 1, 0], [0, 6, 0, 9, 0], [0, 2, 0, 4, 0], [0, 6, 0, 9, 0], [1, 1, 0, 1, 0]],
    [[0, 9, 0, 7, 0], [0, 4, 0, 2, 0], [0, 9, 0, 8, 0], [0, 1, 0, 1, 1], [0, 1, 0, 0, 0]],
    [[0, 7, 0, 0, 0], [0, 2, 0, 3, 0], [0, 8, 0, 0, 0], [0, 1, 1, 5, 1], [0, 0, 0, 0, 0]],
    [[7, 0, 0, 0, 6], [2, 0, 3, 0, 2], [8, 0, 0, 0, 6], [1, 1, 5, 1, 1], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 6, 0], [0, 3, 0, 2, 0], [0, 0, 0, 6, 0], [1, 5, 1, 1, 0], [0, 0, 0, 0, 0]],
    [[0, 6, 0, 9, 0], [0, 2, 0, 4, 0], [0, 6, 0, 9, 0], [1, 1, 0, 1, 0], [0, 0, 0, 1, 0]],
    [[0, 4, 0, 2, 0], [0, 9, 0, 8, 0], [0, 1, 0, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 1, 9]],
    [[0, 2, 0, 4, 0], [0, 6, 0, 9, 0], [1, 1, 0, 1, 0], [0, 0, 0, 1, 0], [9, 1, 1, 1, 0]],
    [[0, 9, 0, 8, 0], [0, 1, 0, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 1, 9], [0, 0, 0, 0, 0]],
    [[9, 0, 8, 0, 0], [1, 0, 1, 1, 5], [1, 0, 0, 0, 0], [1, 1, 1, 9, 4], [0, 0, 0, 0, 0]],
    [[0, 8, 0, 0, 0], [0, 1, 1, 5, 1], [0, 0, 0, 0, 0], [1, 1, 9, 4, 9], [0, 0, 0, 0, 0]],
    [[8, 0, 0, 0, 6], [1, 1, 5, 1, 1], [0, 0, 0, 0, 0], [1, 9, 4, 9, 1], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 6, 0], [1, 5, 1, 1, 0], [0, 0, 0, 0, 0], [9, 4, 9, 1, 1], [0, 0, 0, 0, 0]],
    [[0, 0, 6, 0, 9], [5, 1, 1, 0, 1], [0, 0, 0, 0, 1], [4, 9, 1, 1, 1], [0, 0, 0, 0, 0]],
    [[0, 6, 0, 9, 0], [1, 1, 0, 1, 0], [0, 0, 0, 1, 0], [9, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
]

# Navigation targets used by the two roles.
_SLOT_PICK0 = 0     # (3,3): LEFT -> ingredient 0 supply
_SLOT_PICK1 = 1     # (5,3): LEFT -> ingredient 1 supply
_SLOT_POT0 = 2      # (3,4): DOWN -> pot, UP -> button (2,4)
_SLOT_POT1 = 3      # (5,4): UP -> pot, DOWN -> button (6,4)
_SLOT_PLATE0 = 4    # (3,5): RIGHT -> plate pile (3,6), UP -> counter (2,5)
_SLOT_PLATE1 = 5    # (5,5): RIGHT -> plate pile (5,6), DOWN -> counter (6,5)
_SLOT_SERVE = 6     # (4,5): LEFT -> pot, RIGHT -> delivery (4,6)
_SLOT_POST = 7      # (1,4): DOWN -> button (2,4), sees R at (0,4)
_SLOT_HAND0 = 8     # (1,3): DOWN -> shared counter (2,3)
_SLOT_HAND1 = 9     # (1,5): DOWN -> shared counter (2,5)
_SLOT_SERVE0 = 10   # (4,1): RIGHT -> delivery (4,2)
_SLOT_SERVE1 = 11   # (4,7): LEFT -> delivery (4,6)
_SLOT_POST1 = 12    # (7,4): UP -> button (6,4), sees R at (8,4)
_SLOT_HAND2 = 13    # (7,3): UP -> shared counter (6,3)
_SLOT_HAND3 = 14    # (7,5): UP -> shared counter (6,5)

_TARGET_CELL = [10, 19, 11, 20, 12, 21, 16, 3, 2, 4, 14, 17, 28, 27, 29]

# First movement action of a shortest path from each floor cell to each target
# (RIGHT=0, DOWN=1, LEFT=2, UP=3, STAY=4 when already there or unreachable).
_NEXT = [
    [4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 0, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 2, 0, 2, 2, 2],
    [4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 2, 0, 2, 2, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 2, 0, 2, 0, 0, 2, 0],
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 2, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 1, 3, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 1, 3, 1, 1, 1],
    [4, 1, 0, 1, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 2, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 1, 2, 1, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 1, 1, 1, 1],
    [3, 1, 3, 1, 3, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 1, 3, 1, 3, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 1, 4, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1],
    [3, 4, 3, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 2, 0, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 2, 3, 2, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 1, 3, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 1, 3, 1, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 0, 2, 0, 0, 4, 0],
    [4, 4, 4, 4, 4, 4, 4, 0, 2, 0, 2, 0, 4, 2, 0],
    [4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 2, 0, 2, 2, 4],
    [4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 2, 0, 2, 2, 2],
    [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2, 3, 2, 2, 2],
]

# Shortest-path lengths to the same targets (99 = other floor component).
_DIST = [
    [99, 99, 99, 99, 99, 99, 99, 3, 2, 4, 3, 9, 9, 8, 10],
    [99, 99, 99, 99, 99, 99, 99, 2, 1, 3, 4, 8, 10, 9, 11],
    [99, 99, 99, 99, 99, 99, 99, 1, 0, 2, 5, 7, 11, 10, 12],
    [99, 99, 99, 99, 99, 99, 99, 0, 1, 1, 6, 6, 12, 11, 11],
    [99, 99, 99, 99, 99, 99, 99, 1, 2, 0, 7, 5, 11, 12, 10],
    [99, 99, 99, 99, 99, 99, 99, 2, 3, 1, 8, 4, 10, 11, 9],
    [99, 99, 99, 99, 99, 99, 99, 3, 4, 2, 9, 3, 9, 10, 8],
    [99, 99, 99, 99, 99, 99, 99, 4, 3, 5, 2, 10, 8, 7, 9],
    [99, 99, 99, 99, 99, 99, 99, 4, 5, 3, 10, 2, 8, 9, 7],
    [99, 99, 99, 99, 99, 99, 99, 5, 4, 6, 1, 11, 7, 6, 8],
    [0, 2, 1, 3, 2, 4, 3, 99, 99, 99, 99, 99, 99, 99, 99],
    [1, 3, 0, 4, 1, 3, 2, 99, 99, 99, 99, 99, 99, 99, 99],
    [2, 4, 1, 3, 0, 2, 1, 99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 5, 6, 4, 11, 1, 7, 8, 6],
    [99, 99, 99, 99, 99, 99, 99, 6, 5, 7, 0, 12, 6, 5, 7],
    [1, 1, 2, 2, 3, 3, 4, 99, 99, 99, 99, 99, 99, 99, 99],
    [3, 3, 2, 2, 1, 1, 0, 99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 6, 7, 5, 12, 0, 6, 7, 5],
    [99, 99, 99, 99, 99, 99, 99, 7, 6, 8, 1, 11, 5, 4, 6],
    [2, 0, 3, 1, 4, 2, 3, 99, 99, 99, 99, 99, 99, 99, 99],
    [3, 1, 4, 0, 3, 1, 2, 99, 99, 99, 99, 99, 99, 99, 99],
    [4, 2, 3, 1, 2, 0, 1, 99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 7, 8, 6, 11, 1, 5, 6, 4],
    [99, 99, 99, 99, 99, 99, 99, 8, 7, 9, 2, 10, 4, 3, 5],
    [99, 99, 99, 99, 99, 99, 99, 8, 9, 7, 10, 2, 4, 5, 3],
    [99, 99, 99, 99, 99, 99, 99, 9, 8, 10, 3, 9, 3, 2, 4],
    [99, 99, 99, 99, 99, 99, 99, 10, 9, 11, 4, 8, 2, 1, 3],
    [99, 99, 99, 99, 99, 99, 99, 11, 10, 12, 5, 7, 1, 0, 2],
    [99, 99, 99, 99, 99, 99, 99, 12, 11, 11, 6, 6, 0, 1, 1],
    [99, 99, 99, 99, 99, 99, 99, 11, 12, 10, 7, 5, 1, 2, 0],
    [99, 99, 99, 99, 99, 99, 99, 10, 11, 9, 8, 4, 2, 3, 1],
    [99, 99, 99, 99, 99, 99, 99, 9, 10, 8, 9, 3, 3, 4, 2],
]

# Actions: RIGHT=0 DOWN=1 LEFT=2 UP=3 STAY=4 INTERACT=5
_STAY = 4
_INTERACT = 5
# Facing ids: UP=0 DOWN=1 RIGHT=2 LEFT=3
_UP = 0
_DOWN = 1
_RIGHT = 2
_LEFT = 3
_TURN_ACTION = [3, 1, 0, 2]      # movement action that produces each facing
_FACE_DR = [-1, 1, 0, 0]
_FACE_DC = [0, 0, 1, -1]

_MODE_INTERACT = 0
_MODE_WAIT = 1

# memory slots
_M_REC = 0          # recipe ingredient id believed current, -1 when unknown
_M_PRESS = 1        # countdown crediting a live display to our own activation
_M_PARTNER = 2      # number of button activations credited to the partner
_M_STEP = 3         # steps since reset (saturating)
_M_BCAST = 4        # recipe last seen on a live button display
_M_INDICATOR = 5    # recipe last seen on an always-on indicator
_M_PREVDISP = 6     # whether a display was live on the previous step
_M_DARK = 7         # steps since a live display was last seen (saturating)
_M_SILENT = 8       # sticky: nobody but us has ever lit the display in time
_M_USEFUL = 9       # sticky: a delivery followed our broadcasts
_M_WASTED = 10      # broadcasts so far without any sign of a delivery
_M_FERRY = 11       # 0 none, 1 collecting from (2,3), 2 collecting from (2,5)
_M_OWE = 12         # we resampled the recipe ourselves and owe a broadcast
_M_POST = 13        # which end of the ring the outer seat broadcasts from
_M_UNSEEN = 14      # steps since the partner was last inside our view
_M_PARKED = 15      # steps a deliverable dish has been sitting on a counter

_INIT_MEMORY = [-1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# The partner must be seen lighting the display twice before the cook stops
# paying for its own well-timed activation: a single start-of-episode overlap
# is not evidence of a reliable broadcaster.
_PARTNER_EVIDENCE = 2
# The cook lights the display once per dish (~39 steps).  Only after clearly
# longer silence does the outer seat conclude that nobody is informing the cook
# and start paying for broadcasts itself.
_DARK_LIMIT = 60
# Once broadcasts are demonstrably followed by deliveries, keep the display
# available about once per cook cycle instead of once per recipe change.
_DARK_SERVING = 30
# Stop paying for broadcasts that never produce a delivery.
_WASTED_LIMIT = 4
# A cook waiting for the recipe at the other end of the ring is never inside
# our view; only then is it worth moving the broadcast post to the far button.
_UNSEEN_LIMIT = 25
# A cook has to put a finished dish down to free its hands for the button, so
# only a dish that stays put is treated as handed over rather than parked.
_PARKED_LIMIT = 8
# Pot timer value at which the cook activates the button: the remaining
# collect-and-deliver sequence then takes 7 steps and the display stays live
# for 9, so the recipe sampled by the delivery is read for free.
_PRESS_AT_TIMER = 5
# A well-timed activation only pays for itself if another full cook cycle
# (7 steps to this delivery plus 39 for the next one) still fits.
_HORIZON = 400
_CYCLE = 46
_START_CYCLE = 43


def _initial_memory():
    return jnp.asarray(_INIT_MEMORY, dtype=jnp.int32)


def _goto(pos, facing, slot, want_face, mode):
    """Walk to the target cell of ``slot``, turn to ``want_face``, then act."""
    target_cell = jnp.asarray(_TARGET_CELL, dtype=jnp.int32)[slot]
    at_cell = pos == target_cell
    aligned = at_cell & (facing == want_face)
    walk = jnp.asarray(_NEXT, dtype=jnp.int32)[pos, slot]
    turn = jnp.asarray(_TURN_ACTION, dtype=jnp.int32)[want_face]
    final = jnp.where(mode == _MODE_INTERACT, _INTERACT, _STAY)
    action = jnp.where(aligned, final, jnp.where(at_cell, turn, walk))
    return action.astype(jnp.int32), aligned


def _act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, jnp.asarray(_INIT_MEMORY, dtype=jnp.int32), memory)
    rec = memory[_M_REC]
    press_age = jnp.maximum(memory[_M_PRESS] - 1, 0)
    partner_presses = memory[_M_PARTNER]
    step = jnp.minimum(memory[_M_STEP] + 1, 10000)
    bcast = memory[_M_BCAST]
    indicator = memory[_M_INDICATOR]
    prev_disp = memory[_M_PREVDISP]
    dark = jnp.minimum(memory[_M_DARK] + 1, 10000)
    silent = memory[_M_SILENT]
    useful = memory[_M_USEFUL]
    wasted = memory[_M_WASTED]
    prev_indicator = memory[_M_INDICATOR]
    ferry = memory[_M_FERRY]
    owe = memory[_M_OWE]
    post_side = memory[_M_POST]
    unseen = jnp.minimum(memory[_M_UNSEEN] + 1, 10000)
    parked = memory[_M_PARKED]

    obs = obs.astype(jnp.int32)

    # ---- self localisation, facing, inventory ---------------------------
    weights = jnp.arange(1, 10, dtype=jnp.int32)
    code = jnp.sum(obs[:, :, 20:29] * weights[None, None, :], axis=2)
    patches = jnp.asarray(_PATCH, dtype=jnp.int32)
    pos = jnp.argmax(jnp.sum(patches == code[None, :, :], axis=(1, 2))).astype(jnp.int32)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ing = obs[2, 2, 7:10]
    has_ing = jnp.sum(held_ing) > 0
    holding_dish = held_plate & held_cooked
    holding_plate = held_plate & (~held_cooked)

    # ---- pot: visible from every inner cell, never from the outer ring ---
    unseen = jnp.where(jnp.sum(obs[:, :, 10]) > 0, 0, unseen)

    pot_mask = obs[:, :, 22]
    is_inner = jnp.sum(pot_mask) > 0
    pot_ing = jnp.max(obs[:, :, 31:34] * pot_mask[:, :, None], axis=(0, 1))
    pot_count = jnp.sum(pot_ing)
    pot_cooked = jnp.max(obs[:, :, 30] * pot_mask) > 0
    pot_timer = jnp.max(obs[:, :, 39] * pot_mask)

    # ---- recipe displays -------------------------------------------------
    button_mask = obs[:, :, 24]
    disp = jnp.max(obs[:, :, 36:39] * button_mask[:, :, None], axis=(0, 1))
    disp_live = jnp.sum(disp) > 0
    disp_rec = jnp.argmax(disp).astype(jnp.int32)
    ind_mask = obs[:, :, 23]
    ind = jnp.max(obs[:, :, 36:39] * ind_mask[:, :, None], axis=(0, 1))
    ind_seen = jnp.sum(ind) > 0
    ind_rec = jnp.argmax(ind).astype(jnp.int32)

    # A display that lights up while none of our own activations can still be
    # live is evidence that the partner broadcasts the recipe on its own.
    partner_event = disp_live & (prev_disp == 0) & (press_age == 0)
    partner_presses = jnp.minimum(partner_presses + partner_event.astype(jnp.int32), 100)
    timed_mode = partner_presses < _PARTNER_EVIDENCE
    dark = jnp.where(disp_live, 0, dark)

    # Any live display and any visible indicator show the *current* recipe.
    indicator = jnp.where(ind_seen, ind_rec, indicator)
    bcast = jnp.where(disp_live, disp_rec, bcast)
    rec = jnp.where(disp_live, disp_rec, rec)
    rec = jnp.where(ind_seen, ind_rec, rec)
    rec_known = rec >= 0
    side = jnp.where(rec == 1, 1, 0)

    # ---- inner cook: stations -------------------------------------------
    pick_slot = jnp.where(side == 0, _SLOT_PICK0, _SLOT_PICK1)
    pick_face = jnp.asarray(_LEFT, dtype=jnp.int32)
    pot_slot = jnp.where(side == 0, _SLOT_POT0, _SLOT_POT1)
    pot_face = jnp.where(side == 0, _DOWN, _UP)
    # The well-timed activation and the plate always use the top of the ring:
    # the route (3,4)->(3,5)->(4,5) costs the same from either filling side and
    # keeps every activation inside the outer seat's view from (1,4), so the
    # partner can tell that the cook informs itself.
    btn_slot = jnp.asarray(_SLOT_POT0, dtype=jnp.int32)
    btn_face = jnp.asarray(_UP, dtype=jnp.int32)
    plate_slot = jnp.asarray(_SLOT_PLATE0, dtype=jnp.int32)
    plate_face = jnp.asarray(_RIGHT, dtype=jnp.int32)

    dist = jnp.asarray(_DIST, dtype=jnp.int32)
    near_bottom = dist[pos, _SLOT_POT1] < dist[pos, _SLOT_POT0]
    any_btn_slot = jnp.where(near_bottom, _SLOT_POT1, _SLOT_POT0)
    any_btn_face = jnp.where(near_bottom, _DOWN, _UP)

    # Spare counters for discarding an unusable item; switch side if occupied.
    face_dr = jnp.asarray(_FACE_DR, dtype=jnp.int32)
    face_dc = jnp.asarray(_FACE_DC, dtype=jnp.int32)
    front_item = obs[2 + face_dr[facing], 2 + face_dc[facing], 29:34]
    front_busy = jnp.sum(front_item) > 0
    drop_slot = jnp.asarray(_SLOT_PLATE0, dtype=jnp.int32)
    drop_face = jnp.asarray(_UP, dtype=jnp.int32)
    at_drop = (pos == jnp.asarray(_TARGET_CELL, dtype=jnp.int32)[drop_slot]) & (facing == drop_face)
    switch = at_drop & front_busy
    drop_slot = jnp.where(switch, _SLOT_PLATE1, drop_slot)
    drop_face = jnp.where(switch, _DOWN, drop_face)

    # ---- inner cook: what to do next -------------------------------------
    pot_fillable = (pot_timer == 0) & (~pot_cooked) & (pot_count < 3)

    # empty hands
    worth_pressing = step + _CYCLE <= _HORIZON
    hold_button = timed_mode & worth_pressing & (pot_timer >= _PRESS_AT_TIMER)
    cook_slot = jnp.where(hold_button, btn_slot, plate_slot)
    cook_face = jnp.where(hold_button, btn_face, plate_face)
    cook_mode = jnp.where(hold_button & (pot_timer > _PRESS_AT_TIMER),
                          _MODE_WAIT, _MODE_INTERACT)

    idle_slot = jnp.where(pot_cooked, plate_slot,
                          jnp.where(pot_timer > 0, cook_slot, pick_slot))
    idle_face = jnp.where(pot_cooked, plate_face,
                          jnp.where(pot_timer > 0, cook_face, pick_face))
    idle_mode = jnp.where(pot_cooked, _MODE_INTERACT,
                          jnp.where(pot_timer > 0, cook_mode, _MODE_INTERACT))
    # Learning the recipe from scratch only pays for itself if a whole dish
    # still fits in the remaining horizon.
    worth_starting = step + _START_CYCLE <= _HORIZON
    idle_slot = jnp.where(rec_known, idle_slot, any_btn_slot)
    idle_face = jnp.where(rec_known, idle_face, any_btn_face)
    idle_mode = jnp.where(rec_known, idle_mode,
                          jnp.where(worth_starting, _MODE_INTERACT, _MODE_WAIT))

    # holding a raw ingredient
    onehot = jnp.eye(3, dtype=jnp.int32)[jnp.clip(rec, 0, 2)]
    held_target = jnp.sum(held_ing * onehot)
    usable = rec_known & (held_target > 0) & pot_fillable
    ing_slot = jnp.where(usable, pot_slot, drop_slot)
    ing_face = jnp.where(usable, pot_face, drop_face)

    # holding an empty plate
    pot_busy = pot_cooked | (pot_timer > 0)
    wait_slot = jnp.where(pot_busy, _SLOT_SERVE, drop_slot)
    wait_face = jnp.where(pot_busy, _LEFT, drop_face)
    wait_mode = jnp.where(pot_busy & (~pot_cooked), _MODE_WAIT, _MODE_INTERACT)

    # holding a plated cooked dish
    dish_ok = (~rec_known) | (held_target >= 3)
    dish_slot = jnp.where(dish_ok, _SLOT_SERVE, drop_slot)
    dish_face = jnp.where(dish_ok, _RIGHT, drop_face)

    in_slot = jnp.where(holding_dish, dish_slot,
                        jnp.where(holding_plate, wait_slot,
                                  jnp.where(has_ing, ing_slot, idle_slot)))
    in_face = jnp.where(holding_dish, dish_face,
                        jnp.where(holding_plate, wait_face,
                                  jnp.where(has_ing, ing_face, idle_face)))
    in_mode = jnp.where(holding_dish, _MODE_INTERACT,
                        jnp.where(holding_plate, wait_mode,
                                  jnp.where(has_ing, _MODE_INTERACT, idle_mode)))

    # ---- outer information service ---------------------------------------
    # Broadcast only once it is clear that nobody else lights the display: a
    # cook that informs itself refreshes it about once per dish.  Once that
    # silence is established, refresh whenever the recipe changes, and - as soon
    # as a delivery proves that somebody is acting on the broadcasts - roughly
    # once per cook cycle so a partner without recipe memory is never stuck.
    # Broadcasts that never lead to a delivery are abandoned after a few tries.
    silent = jnp.maximum(silent, (dark >= _DARK_LIMIT).astype(jnp.int32))
    recipe_changed = ind_seen & (prev_indicator >= 0) & (ind_rec != prev_indicator)
    useful = jnp.maximum(useful, ((silent > 0) & recipe_changed).astype(jnp.int32))
    wasted = jnp.where(recipe_changed, 0, wasted)
    dark_limit = jnp.where(useful > 0, _DARK_SERVING, _DARK_LIMIT)
    # A partner carrying a finished dish is about to deliver and to resample the
    # recipe, so that is the ideal moment to have the display live for it.
    partner_dish = jnp.max(obs[:, :, 15] * obs[:, :, 16]) > 0
    stale = ((dark >= dark_limit) | ((silent > 0) & (indicator != bcast))
             | ((silent > 0) & partner_dish))
    give_up = (useful == 0) & (wasted >= _WASTED_LIMIT)
    # Delivering a handed-over dish ourselves resamples the recipe, so we owe
    # the cook that information even when it normally informs itself.
    owe = jnp.where(disp_live, 0, owe)
    # If the partner ever lights the display without us, it can inform itself
    # and paying for further broadcasts would only duplicate its cost.
    serves_itself = partner_presses > 0
    needs_broadcast = ((indicator >= 0) & (~disp_live)
                       & ((stale & (~give_up) & (~serves_itself)) | (owe > 0)))

    # Anything plated, cooked and matching the recipe that is left on a shared
    # counter is worth carrying to a delivery station; the outer seat can check
    # the match against the always-on indicator, so this can never lose points.
    counter_item = (obs[:, :, 20] > 0) & (obs[:, :, 29] > 0) & (obs[:, :, 30] > 0)
    ind_onehot = jnp.eye(3, dtype=jnp.int32)[jnp.clip(indicator, 0, 2)]
    item_target = jnp.sum(obs[:, :, 31:34] * ind_onehot[None, None, :], axis=2)
    item_total = jnp.sum(obs[:, :, 31:34], axis=2)
    good_dish = counter_item & (item_target >= 3) & (item_total <= 3) & (indicator >= 0)
    # A cook waiting for the recipe may be waiting at either button, so an
    # unanswered broadcast is retried from the opposite side of the ring.
    post_slot = jnp.where(post_side == 0, _SLOT_POST, _SLOT_POST1)
    post_face = jnp.where(post_side == 0, _DOWN, _UP)
    at_post = pos == jnp.asarray(_TARGET_CELL, dtype=jnp.int32)[post_slot]
    left_cell = jnp.where(post_side == 0, good_dish[3, 1], good_dish[1, 1])
    right_cell = jnp.where(post_side == 0, good_dish[3, 3], good_dish[1, 3])
    first = jnp.where(post_side == 0, 1, 3)
    parked = jnp.where(at_post & (left_cell | right_cell),
                       jnp.minimum(parked + 1, 1000), 0)
    abandoned = parked >= _PARKED_LIMIT
    ferry = jnp.where(at_post & (ferry == 0) & abandoned & left_cell, first, ferry)
    ferry = jnp.where(at_post & (ferry == 0) & abandoned & right_cell, first + 1, ferry)

    hands = jnp.asarray([_SLOT_HAND0, _SLOT_HAND1, _SLOT_HAND2, _SLOT_HAND3], dtype=jnp.int32)
    faces = jnp.asarray([_DOWN, _DOWN, _UP, _UP], dtype=jnp.int32)
    serves = jnp.asarray([_SLOT_SERVE0, _SLOT_SERVE1, _SLOT_SERVE0, _SLOT_SERVE1],
                         dtype=jnp.int32)
    serve_faces = jnp.asarray([_RIGHT, _LEFT, _RIGHT, _LEFT], dtype=jnp.int32)
    which = jnp.clip(ferry - 1, 0, 3)
    take_slot = hands[which]
    take_face = faces[which]
    give_slot = serves[which]
    give_face = serve_faces[which]
    carrying = held_plate & held_cooked
    carry_ok = jnp.sum(held_ing * ind_onehot) >= 3
    ferrying = (ferry > 0) | carrying
    ferry_slot = jnp.where(carrying, jnp.where(carry_ok, give_slot, take_slot), take_slot)
    ferry_face = jnp.where(carrying, jnp.where(carry_ok, give_face, take_face), take_face)

    # Once we know nobody else lights the display, follow the cook: if it has
    # not been inside our view for a while it is waiting at the far button.
    move_post = ((~is_inner) & (silent > 0) & at_post & (ferry == 0)
                 & (~carrying) & (unseen >= _UNSEEN_LIMIT))

    out_slot = jnp.where(ferrying, ferry_slot, post_slot)
    out_face = jnp.where(ferrying, ferry_face, post_face)
    out_mode = jnp.where(ferrying, _MODE_INTERACT,
                         jnp.where(needs_broadcast, _MODE_INTERACT, _MODE_WAIT))

    slot = jnp.where(is_inner, in_slot, out_slot)
    want_face = jnp.where(is_inner, in_face, out_face)
    mode = jnp.where(is_inner, in_mode, out_mode)

    action, aligned = _goto(pos, facing, slot, want_face, mode)
    acted = aligned & (mode == _MODE_INTERACT)

    # What the interaction actually hits, read straight off the observation.
    front_row = 2 + face_dr[want_face]
    front_col = 2 + face_dc[want_face]
    front_button = obs[front_row, front_col, 24] > 0
    front_goal = obs[front_row, front_col, 21] > 0
    front_empty = jnp.sum(obs[front_row, front_col, 29:34]) == 0
    empty_hands = (~held_plate) & (~has_ing)
    pressed = acted & front_button & empty_hands
    delivered = acted & front_goal & holding_dish

    # Ferry bookkeeping for the outer seat.
    gone = aligned & (~carrying) & (ferry > 0) & front_empty
    blocked = aligned & carrying & (~carry_ok) & (~front_empty)
    ferry = jnp.where(blocked, (((ferry - 1) ^ 1) + 1), ferry)
    ferry = jnp.where(is_inner | (acted & carrying & (front_goal | front_empty)) | gone,
                      0, ferry)

    next_memory = jnp.stack([
        jnp.where(delivered, -1, rec).astype(jnp.int32),
        jnp.where(pressed, 10, press_age).astype(jnp.int32),
        partner_presses.astype(jnp.int32),
        step.astype(jnp.int32),
        jnp.where(pressed & (indicator >= 0), indicator, bcast).astype(jnp.int32),
        indicator.astype(jnp.int32),
        disp_live.astype(jnp.int32),
        dark.astype(jnp.int32),
        silent.astype(jnp.int32),
        useful.astype(jnp.int32),
        jnp.minimum(wasted + (pressed & (~is_inner)).astype(jnp.int32), 100).astype(jnp.int32),
        ferry.astype(jnp.int32),
        jnp.where(delivered & (~is_inner), 1, owe).astype(jnp.int32),
        jnp.where(move_post, 1 - post_side, post_side).astype(jnp.int32),
        jnp.where(move_post, 0, unseen).astype(jnp.int32),
        parked.astype(jnp.int32),
    ])

    trace = {
        "pos": pos,
        "recipe": rec.astype(jnp.int32),
        "pot_timer": pot_timer.astype(jnp.int32),
        "pot_count": pot_count.astype(jnp.int32),
        "inner": is_inner.astype(jnp.int32),
        "action": action.astype(jnp.int32),
        "pressed": pressed.astype(jnp.int32),
        "delivered": delivered.astype(jnp.int32),
        "dark": dark.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_0():
    return _initial_memory()


def agent_0_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)


def init_agent_1():
    return _initial_memory()


def agent_1_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)
