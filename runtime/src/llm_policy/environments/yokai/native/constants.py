import jax.numpy as jnp

# 16 x 16 adjecency matrix for the 4x4 initial grid
INITIAL_neighbors_4x4 = jnp.array([
    # ROW 0
    [0, 1, 0, 0, 
     1, 0, 0, 0, 
     0, 0, 0, 0, 
     0, 0, 0, 0],
    [1, 0, 1, 0, 
     0, 1, 0, 0, 
     0, 0, 0, 0, 
     0, 0, 0, 0],
    [0, 1, 0, 1, 
     0, 0, 1, 0, 
     0, 0, 0, 0, 
     0, 0, 0, 0],
    [0, 0, 1, 0, 
     0, 0, 0, 1, 
     0, 0, 0, 0,
     0, 0, 0, 0],
    # ROW 1
    [1, 0, 0, 0, 
     0, 1, 0, 0, 
     1, 0, 0, 0, 
     0, 0, 0, 0],
    [0, 1, 0, 0,
     1, 0, 1, 0, 
     0, 1, 0, 0, 
     0, 0, 0, 0],
    [0, 0, 1, 0, 
     0, 1, 0, 1, 
     0, 0, 1, 0, 
     0, 0, 0, 0],
    [0, 0, 0, 1, 
     0, 0, 1, 0, 
     0, 0, 0, 1,
     0, 0, 0, 0],
    # ROW 2
    [0, 0, 0, 0, 
     1, 0, 0, 0, 
     0, 1, 0, 0, 
     1, 0, 0, 0],
    [0, 0, 0, 0, 
     0, 1, 0, 0, 
     1, 0, 1, 0, 
     0, 1, 0, 0],
    [0, 0, 0, 0, 
     0, 0, 1, 0, 
     0, 1, 0, 1, 
     0, 0, 1, 0],
    [0, 0, 0, 0, 
     0, 0, 0, 1, 
     0, 0, 1, 0, 
     0, 0, 0, 1],
    # ROW 3
    [0, 0, 0, 0, 
     0, 0, 0, 0, 
     1, 0, 0, 0, 
     0, 1, 0, 0],
    [0, 0, 0, 0, 
     0, 0, 0, 0, 
     0, 1, 0, 0, 
     1, 0, 1, 0],
    [0, 0, 0, 0, 
     0, 0, 0, 0, 
     0, 0, 1, 0, 
     0, 1, 0, 1],
    [0, 0, 0, 0, 
     0, 0, 0, 0,
     0, 0, 0, 1, 
     0, 0, 1, 0],
], dtype=jnp.uint8)

INITIAL_neighbors_3x3 = jnp.array([
    # ROW 1
    [0, 1, 0, 
     1, 0, 0, 
     0, 0, 0],
    [1, 0, 1, 
     0, 1, 0, 
     0, 0, 0],
    [0, 1, 0, 
     0, 0, 1,
     0, 0, 0],
    # ROW 2
    [1, 0, 0, 
     0, 1, 0, 
     1, 0, 0],
    [0, 1, 0, 
     1, 0, 1, 
     0, 1, 0],
    [0, 0, 1, 
     0, 1, 0, 
     0, 0, 1],
    # ROW 3
    [0, 0, 0, 
     1, 0, 0, 
     0, 1, 0],
    [0, 0, 0, 
     0, 1, 0, 
     1, 0, 1],
    [0, 0, 0, 
     0, 0, 1, 
     0, 1, 0],
], dtype=jnp.uint8)


YOKAI_COLORS = {
    0: 0, # jnp.array([0, 0, 0, 1]),  # red
    1: 1, # jnp.array([0, 0, 1, 0]),  # green
    2: 2, #jnp.array([0, 1, 0, 0]),  # blue
    3: 3, # jnp.array([1, 0, 0, 0]),  # purple
}

REL_MOVE_INDEX_TO_OFFSET = jnp.array([
    [-1, 0], # OBEN
    [1, 0], # UNTEN
    [0, -1], # LINKS
    [0, 1], # RECHTS
])

ALL_ONE_COLORS_4x4 = jnp.array(
    [jnp.array([0, 0, 0, 1]), jnp.array([0, 0, 1, 0]), jnp.array([0, 1, 0, 0]), jnp.array([1, 0, 0, 0])], dtype=jnp.uint8)

ALL_TWO_COLORS_4x4 = jnp.array(
    [jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 0, 1]), jnp.array([1, 0, 0, 1]),
     jnp.array([0, 1, 1, 0]), jnp.array([1, 0, 1, 0]), jnp.array([1, 1, 0, 0])], dtype=jnp.uint8)

ALL_THREE_COLORS_4x4 = jnp.array(
    [jnp.array([0, 1, 1, 1]), jnp.array([1, 0, 1, 1]), jnp.array([1, 1, 0, 1]), jnp.array([1, 1, 1, 0])], dtype=jnp.uint8)


ALL_ONE_COLORS_3x3 = jnp.array(
    [jnp.array([0, 0, 1]), jnp.array([0, 1, 0]), jnp.array([1, 0, 0])], dtype=jnp.uint8)

ALL_TWO_COLORS_3x3 = jnp.array(
    [jnp.array([0, 1, 1]), jnp.array([1, 0, 1]), jnp.array([1, 1, 0])], dtype=jnp.uint8)

INITIAL_POSITIONS_4x4 = jnp.array([
    # 0      1      2      3
    [0, 0], [0, 1], [0, 2], [0, 3],
    # 4      5      6      7
    [1, 0], [1, 1], [1, 2], [1, 3],
    # 8      9      10     11
    [2, 0], [2, 1], [2, 2], [2, 3],
    # 12     13     14     15
    [3, 0], [3, 1], [3, 2], [3, 3]
], dtype=jnp.int32)


INITIAL_POSITIONS_3x3 = jnp.array([
    [0, 0], [0, 1], [0, 2], 
    [1, 0], [1, 1], [1, 2],
    [2, 0], [2, 1], [2, 2], 
], dtype=jnp.int32)
