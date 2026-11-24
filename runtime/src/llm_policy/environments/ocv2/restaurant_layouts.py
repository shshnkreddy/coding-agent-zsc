"""Original grid abstractions of restaurant circulation, not scale replicas.

Source drawings and design choices live in docs/restaurant_layouts/. Only the
neutral preset identity, geometry, and mechanics enter generated contexts.
"""

RESTAURANT_RECIPES = [[0, 0, 1], [1, 1, 2], [0, 2, 2], [0, 1, 2]]

_PASS = """
WWW0WWBWW1WWWWW
W             W
W WW  A       2
W WW          W
W    WPWWW    W
W             R
W    WWWWW    X
2             W
W WW WWWPW    W
W WW          B
1          A  W
W             W
WWWWWBWWW0WWWWW
"""

_LANES = """
WW0WWWWWWBWWWWW2WWW
W                 W
W A              BW
W  WW WW    W     W
1  WW WW    W     0
W  WP PW    W     W
W  WW WW    W     W
B           W     W
W  WW WW  A       W
2  WW WW     WWWWWW
W  WW WW    WWWWWWW
W           WWWWWWW
WWWWWWXWRW1WWWWWWWW
"""

_LINK = """
WWWRWW0WWWWWWWWWWWWWWWWWWWW
W       WWW0WW1W2WWWWWWWWWW
W A W   W         W     B W
B       W  P  P   W       W
W W  W  W  W  W           W
X       W WWWWWW  W   A   W
W   W             W       1
W       W         WWWWWWWWW
WWWWWWWWWWWWBWWW2WWWWWWWWWW
"""

RESTAURANT_LAYOUTS = {
    "restaurant_pass_v1": (_PASS, RESTAURANT_RECIPES),
    "restaurant_lanes_v1": (_LANES, RESTAURANT_RECIPES),
    "restaurant_link_v1": (_LINK, RESTAURANT_RECIPES),
}
