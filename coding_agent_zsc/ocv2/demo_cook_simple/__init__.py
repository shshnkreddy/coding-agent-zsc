"""Six wording variants of the demo_cook_simple coordination task."""
from coding_agent_zsc.common import make_variants, self_play


def evaluate(path, *, split, seed):
    return self_play("demo_cook_simple", path, split=split, seed=seed)


p0, p1, p2, p3, p4, p5 = make_variants("demo_cook_simple", evaluate)
task = p0
