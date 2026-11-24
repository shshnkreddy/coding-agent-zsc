"""Six wording variants of the yokai_2p_9c_standard coordination task."""
from coding_agent_zsc.common import make_variants, self_play


def evaluate(path, *, split, seed):
    return self_play("yokai_2p_9c_standard", path, split=split, seed=seed)


p0, p1, p2, p3, p4, p5 = make_variants("yokai_2p_9c_standard", evaluate)
task = p0
