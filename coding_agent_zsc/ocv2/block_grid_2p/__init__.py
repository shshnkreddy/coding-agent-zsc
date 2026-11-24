"""Six wording variants of the block_grid_2p coordination task."""
from coding_agent_zsc.common import make_variants, self_play


def evaluate(path, *, split, seed):
    return self_play("block_grid_2p", path, split=split, seed=seed)


p0, p1, p2, p3, p4, p5 = make_variants("block_grid_2p", evaluate)
task = p0
