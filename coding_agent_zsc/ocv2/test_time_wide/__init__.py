"""Six wording variants of the test_time_wide coordination task."""
from coding_agent_zsc.common import make_variants, self_play


def evaluate(path, *, split, seed):
    return self_play("test_time_wide", path, split=split, seed=seed)


p0, p1, p2, p3, p4, p5 = make_variants("test_time_wide", evaluate)
task = p0
