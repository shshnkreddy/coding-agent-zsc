"""Six wording variants of the cat_dog coordination task."""
from coding_agent_zsc.common import make_variants, self_play


def evaluate(path, *, split, seed):
    return self_play("cat_dog", path, split=split, seed=seed)


p0, p1, p2, p3, p4, p5 = make_variants("cat_dog", evaluate)
task = p0
