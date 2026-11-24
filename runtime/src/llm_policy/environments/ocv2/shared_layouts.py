"""Shared restaurant floor plans using only standard OvercookedV2 tiles.

The original three layouts retain the named V2.5 geometry and supplies.
New variants have distinct names so prior runs remain comparable.
Completion displays C/Q become counters W; L is the native recipe button.
There is no V2.5 dependency. Agent markers use ASCII scan order.
"""

from llm_policy.environments.ocv2.restaurant_layouts import RESTAURANT_RECIPES

_SHARED_PASS = """
WWW02WBWW1WWWWW
1             B
W PW  A    A  2
W LW          0
W    WPWWW    W
B             R
W    WWWBW    X
2             W
W WW WWWPW    W
0 PW          B
1          A  W
W             1
WWWWWBWWW02WWWW
"""

# Reflective about the center column. Ingredient, plate, and pot counts match
# shared_pass; the board, button, and delivery station sit on the shared axis.
_SHARED_PASS_SYMMETRIC = """
WW0B2WWRWW2B0WW
W             W
W A         A W
W  WW     WW  W
1  PW     WP  1
W  WW  L  WW  W
B      X      B
W  WW     WW  W
2  PW     WP  2
W  WW     WW  W
W      A      W
W             W
WW1B0WWWWW0B1WW
"""

_PARALLEL_LANES = """
WB02WW1WWBWW0WW2W1W
W                 W
W A            A BW
W  WW WW    W     W
1  WW WW    P     0
W  WP PW    1     W
W  WW W2    W     W
B           W     W
L  WW WB  A       W
2  WW WW     WWWBWW
0  WP PW    WWWWWWW
W           2WWWWWW
WWWBWWXWR01BWWWWWWW
"""

# A square street grid: five horizontal and five vertical one-cell-wide roads
# enclose sixteen 3x3 counter blocks. Stations are mirrored left/right, while
# the four center buttons make the current recipe observable from the middle.
_PARALLEL_LANES_BLOCK_GRID = """
WWWWWWWWWRWWWWWWWWW
W                 W
W W0W W1W W1W W0W W
W WWW WWW WWW WWW W
W WPW W2W W2W WPW W
W                 W
W WBW WBW WBW WBW W
W WWW WWW WWW WWW W
W WWW WLW WLW WWW W
W    A   A   A    W
W WWW WLW WLW WWW W
W WWW WWW WWW WWW W
W WBW WBW WBW WBW W
W                 W
W WPW W2W W2W WPW W
W WWW WWW WWW WWW W
W W0W W1W W1W W0W W
W                 W
WWWWWWWWWXWWWWWWWWW
"""

# Two-player counterpart of block_grid. Geometry and stations are unchanged;
# the center spawn is removed so the two authored starts remain symmetric.
_PARALLEL_LANES_BLOCK_GRID_2P = """
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
"""

_KITCHEN_CORRIDOR = """
WWWR1W0BWWWWWWWWWWWWWWWWWWW
W       WWW0WW1B2WWWWLWWWWW
W A W   W A       W     B W
B       W  P  P   W       W
2 W  W  W  W  W           2
X       W WPWWPW  W   A   W
W   W             W       1
0       W         WWBWWWW0W
W1WWWWWWWWWWBWWW2WWWWWWWWWW
"""

# Three equal 7x7 rooms (40 walkable cells apiece), each with a 3x3 counter
# island. The two inter-room passages are one cell wide. Ingredient piles and
# recipe boards live on the left, pots in the middle, and delivery on the right.
# One authored spawn lies in each room; its preset disables random starts.
_KITCHEN_CORRIDOR_THREE_BOXES = """
WW0W1W2WWWWWWPWWWWWWWWWWWWW
W       WW       WW       W
R       WL       WL       W
0  0R1  WW  PBP  WW  BWB  W
1 AWWW     AWWW      WWWA X
2  2WW  WW  PBP  WW  BWB  W
R       WW       WW       W
W       WW       WW       W
WW0W1W2WWWWWWPWWWWWWWWWWWWW
"""

# Two-player counterpart of pipeline_cook. The left- and right-room starts are
# retained and the middle-room spawn is removed; either player can enter the
# shared middle cooking room through the existing passages.
_KITCHEN_CORRIDOR_THREE_BOXES_2P = """
WW0W1W2WWWWWWPWWWWWWWWWWWWW
W       WW       WW       W
R       WW       WW       W
0  0R1  WL  PBP  WL  BWB  W
1 AWWW      WWW      WWWA X
2  2WW  WW  PBP  WW  BWB  W
R       WW       WW       W
W       WW       WW       W
WW0W1W2WWWWWWPWWWWWWWWWWWWW
"""

# Two isolated, vertically mirrored work areas exchange items across five
# shared counters. The left player has all supplies and delivery; the right
# player has all pots and four always-on recipe indicators, one per island.
# With a radius-two view, the left player cannot see an indicator from any
# reachable floor cell. Fixed starts preserve the intended player roles.
_MIRROR_WORKSHOP = """
WWWBBBWWWWWWWWWWWWW
W       WWW       W
W WWW0  WWW  WRWW W
W WWW1   W   PWWW W
W WWW2  WWW  WWWW W
W       WWW       W
W WWW0   W   WRWW W
W WWW1  WWW  PWWW W
W WWW2  WWW  WWWW W
X      A W A      W
W WWW2  WWW  WWWW W
W WWW1  WWW  PWWW W
W WWW0   W   WRWW W
W       WWW       W
W WWW2  WWW  WWWW W
W WWW1   W   PWWW W
W WWW0  WWW  WRWW W
W       WWW       W
WWWBBBWWWWWWWWWWWWW
"""

SHARED_LAYOUTS = {
    "shared_pass": (_SHARED_PASS, RESTAURANT_RECIPES),
    "mirror_pass": (_SHARED_PASS_SYMMETRIC, RESTAURANT_RECIPES),
    "parallel_lanes": (_PARALLEL_LANES, RESTAURANT_RECIPES),
    "block_grid": (_PARALLEL_LANES_BLOCK_GRID, RESTAURANT_RECIPES),
    "block_grid_2p": (_PARALLEL_LANES_BLOCK_GRID_2P, RESTAURANT_RECIPES),
    "kitchen_corridor": (_KITCHEN_CORRIDOR, RESTAURANT_RECIPES),
    "pipeline_cook": (_KITCHEN_CORRIDOR_THREE_BOXES, RESTAURANT_RECIPES),
    "pipeline_cook_2p": (_KITCHEN_CORRIDOR_THREE_BOXES_2P, RESTAURANT_RECIPES),
    "mirror_workshop": (_MIRROR_WORKSHOP, RESTAURANT_RECIPES),
}

# Preserve previously published draft preset IDs and their saved configs.
SHARED_LAYOUTS.update({
    "shared_pass_symmetric": SHARED_LAYOUTS["mirror_pass"],
    "parallel_lanes_block_grid": SHARED_LAYOUTS["block_grid"],
    "kitchen_corridor_three_boxes": SHARED_LAYOUTS["pipeline_cook"],
})
