"""Six wording variants of the hanabi_2p_h_group_level_1 coordination task."""
from coding_agent_zsc.common import make_variants, self_play


def evaluate(path, *, split, seed):
    return self_play("hanabi_2p_h_group_level_1", path, split=split, seed=seed)


p0, p1, p2, p3, p4, p5 = make_variants("hanabi_2p_h_group_level_1", evaluate)
task = p0
