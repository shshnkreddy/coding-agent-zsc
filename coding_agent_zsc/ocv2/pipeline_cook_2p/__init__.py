"""Six wording variants of the pipeline_cook_2p coordination task."""
from coding_agent_zsc.common import make_variants, self_play


def evaluate(path, *, split, seed):
    return self_play("pipeline_cook_2p", path, split=split, seed=seed)


p0, p1, p2, p3, p4, p5 = make_variants("pipeline_cook_2p", evaluate)
task = p0
