"""Static native encoding for standard 2–5 player Hanabi; no simulator imports."""


def hanabi_layout(num_agents=2, hand_size=None):
    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents not in (2, 3, 4, 5):
        raise ValueError("Hanabi requires num_agents in 2..5")
    expected_hand = 5 if num_agents < 4 else 4
    if hand_size is None:
        hand_size = expected_hand
    if isinstance(hand_size, bool) or not isinstance(hand_size, int) or hand_size != expected_hand:
        raise ValueError(f"standard {num_agents}-player Hanabi requires hand_size={expected_hand}")
    n, h = num_agents, hand_size
    blocks, offset = {}, 0
    for name, size in (("other_hands", (n-1)*h*25), ("missing_cards", n),
                       ("deck", 50-n*h), ("fireworks", 25), ("information_tokens", 8),
                       ("lives", 3), ("discards", 50), ("last_action", 2*n+41+2*h),
                       ("native_knowledge", n*h*35)):
        blocks[name] = [offset, offset+size]
        offset += size
    last, cursor = {}, 0
    for name, size in (("actor", n), ("kind", 4), ("target", n), ("color", 5),
                       ("rank", 5), ("matches", h), ("slot", h), ("card", 25),
                       ("scored", 1), ("added_token", 1)):
        last[name] = [cursor, cursor+size]
        cursor += size
    color_start, rank_start = 2*h, 2*h+(n-1)*5
    noop = 2*h+(n-1)*10
    return {"num_agents": n, "hand_size": h, "num_colors": 5, "num_ranks": 5,
            "observation_size": offset, "num_actions": noop+1,
            "initial_deck_size": 50-n*h, "observation_blocks": blocks,
            "last_action_blocks": last,
            "actions": {"discard": [0,h], "play": [h,2*h],
                        "hint_color": [color_start,rank_start],
                        "hint_rank": [rank_start,noop], "inactive_noop": noop}}


def observation_layout(observation_size):
    for n in (2, 3, 4, 5):
        layout = hanabi_layout(n)
        if layout["observation_size"] == observation_size:
            return layout
    raise ValueError(f"unsupported Hanabi observation size: {observation_size}")
