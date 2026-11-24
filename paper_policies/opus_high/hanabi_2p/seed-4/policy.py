"""Convention-light rule-based two-player Hanabi agent.

Design goals, in order of importance for play with an *unknown* partner:

1. Never rely on a signal that only this code base understands.  Every hint is
   chosen so that it is useful under the two readings that essentially all
   Hanabi agents share: the literal reading (the hint restricts the possible
   identities of the touched cards) and the "a touched card is worth playing"
   reading.  Preference goes to hints after which a genuinely playable card is
   *literally certain* to be playable, and hints that touch a non-playable card
   are penalised so that an eager partner is not walked into a bomb.
2. Act on locally verifiable evidence.  The agent rebuilds an exact literal
   posterior for each of its own slots from the native hint mask, the discard
   pile, the fireworks and the partner hand it can see, and plays only when
   that posterior clears a threshold that tightens as lives run out.
3. Treat "the partner just hinted this slot" as evidence, not as a protocol.
   It enters as a bounded Bayesian odds boost on the literal posterior, so a
   slot that literally cannot be playable is never played no matter how it was
   hinted, and a save hint on a dangerous card cannot force a play.
4. Spend discards on the cheapest card available, measuring cost in points that
   the discard would make unreachable rather than in a flat "useless" flag.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'literal_hanabi_v2',
    'adapter': 'hanabi',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 5,
    'num_ranks': 5,
    'hand_size': 5,
    'observation_size': 658,
    'num_actions': 21,
    'policy_structure': 'shared_local',
}

NUM_COLORS = 5
NUM_RANKS = 5
HAND_SIZE = 5
NOOP = 20

RANK_COPIES = (3.0, 2.0, 2.0, 2.0, 1.0)

# --- play thresholds by remaining lives (bomb-zero scoring) ----------------
PLAY_THR_LIVES3 = 0.75
PLAY_THR_LIVES2 = 0.85
PLAY_THR_LIVES1 = 0.97
ENDGAME_RELIEF = 0.12

# --- how much a fresh partner hint shifts the odds of "this is playable" ---
HINT_ODDS = 6.0
HINT_ODDS_TWO = 4.5
HINT_DECAY = 0.75
HINT_FLOOR = 0.05

# --- hint scoring weights, in approximate points -------------------------
W_INDUCE_PLAY = 1.2
W_INDUCE_BAD = 2.0
W_CERTAIN_PLAY = 1.2
W_CERTAIN_TRASH = 0.3
W_TOUCH_PLAYABLE = 1.0
W_TOUCH_BAD = 0.3
W_TOUCH_TRASH = 0.05
W_SAVE = 0.6
W_TOUCH_ANY = 0.02
# flat penalty for a hint that touches any non-playable card: such a hint is
# only safe for partners that reason literally, and is a bomb for partners that
# simply play a card the hint touched
W_UNSAFE = 0.8

HINT_THR_LOW = 1.2
HINT_THR_MID = 0.4
HINT_THR_HIGH = 0.15
HINT_THR_RICH = 0.15

# --- discard scoring -------------------------------------------------------
LOSS_PAIR = 0.6
LOSS_TRIPLE = 0.15
DISCARD_HINTED = 0.6
DISCARD_AGE = 0.04


def _one(x):
    return x.astype(jnp.float32)


def _idx():
    return jnp.arange(5, dtype=jnp.int32)


def _copies():
    return jnp.asarray(RANK_COPIES, dtype=jnp.float32)


def _decode(features):
    """Split the 658-float observation into the quantities the rules need."""
    partner_hand = features[0:125].reshape(HAND_SIZE, NUM_COLORS, NUM_RANKS)
    deck = jnp.sum(features[127:167])
    fireworks = features[167:192].reshape(NUM_COLORS, NUM_RANKS)
    height = jnp.sum(fireworks * jnp.arange(1, 6, dtype=jnp.float32)[None, :], axis=1)
    tokens = jnp.sum(features[192:200])
    lives = jnp.sum(features[200:203])
    pile = features[203:253].reshape(NUM_COLORS, 10)
    discards = jnp.stack(
        [
            jnp.sum(pile[:, 0:3], axis=1),
            jnp.sum(pile[:, 3:5], axis=1),
            jnp.sum(pile[:, 5:7], axis=1),
            jnp.sum(pile[:, 7:9], axis=1),
            jnp.sum(pile[:, 9:10], axis=1),
        ],
        axis=1,
    )
    knowledge = features[308:658].reshape(2, HAND_SIZE, 35)
    return partner_hand, deck, height, tokens, lives, discards, knowledge


def _card_tables(height, discards):
    """Board-derived per-card-type facts shared by every rule."""
    ranks = _idx()[None, :]
    playable = ranks == height[:, None].astype(jnp.int32)
    played = ranks < height[:, None].astype(jnp.int32)
    total = jnp.broadcast_to(_copies()[None, :], (NUM_COLORS, NUM_RANKS))
    existing = total - discards          # copies not in the discard pile
    in_play = existing - _one(played)    # copies still in hands or the deck
    # a rank is reachable only if every rank from the current height up to it
    # still has at least one surviving copy
    reachable = jnp.cumprod(_one((existing > 0.5) | played), axis=1)
    useless = played | (reachable < 0.5)
    # losing the last copy of rank r caps that colour at r, so 5 - r points go
    loss = jnp.where(
        useless,
        0.0,
        jnp.where(
            existing < 1.5,
            5.0 - _one(ranks),
            jnp.where(existing < 2.5, LOSS_PAIR, LOSS_TRIPLE),
        ),
    )
    return playable, useless, loss, jnp.maximum(in_play, 0.0)


def _own_belief(knowledge, in_play, partner_hand):
    """Literal posterior over each own slot, refined with the visible hand."""
    v0 = knowledge[0][:, 0:25].reshape(HAND_SIZE, NUM_COLORS, NUM_RANKS)
    mask = v0 > 0
    mine = jnp.maximum(in_play - jnp.sum(partner_hand, axis=0), 0.0)
    weights = jnp.where(mask, mine[None, :, :], 0.0)
    total = jnp.sum(weights, axis=(1, 2), keepdims=True)
    posterior = jnp.where(total > 0, weights / jnp.maximum(total, 1e-6), v0)
    occupied = jnp.sum(v0, axis=(1, 2)) > 0
    hinted = jnp.sum(knowledge[0][:, 25:35], axis=1) > 0
    return posterior, occupied, hinted


def _hint_tables(partner_hand, knowledge):
    """Per-hint partner masks: index h < 5 is colour h, h >= 5 is rank h - 5."""
    idx = _idx()
    same = idx[:, None] == idx[None, :]
    color_sel = jnp.broadcast_to(same[:, :, None], (5, NUM_COLORS, NUM_RANKS))
    rank_sel = jnp.broadcast_to(same[:, None, :], (5, NUM_COLORS, NUM_RANKS))
    sel = jnp.concatenate([color_sel, rank_sel], axis=0)          # [10, c, r]
    by_color = jnp.sum(partner_hand, axis=2)                      # [slot, colour]
    by_rank = jnp.sum(partner_hand, axis=1)                       # [slot, rank]
    touched = jnp.concatenate([by_color.T, by_rank.T], axis=0) > 0  # [10, slot]
    base = knowledge[1][:, 0:25].reshape(HAND_SIZE, NUM_COLORS, NUM_RANKS) > 0
    keep = jnp.where(touched[:, :, None, None], sel[:, None, :, :], ~sel[:, None, :, :])
    return base, touched, base[None] & keep


def _certain(mask, prop, occupied):
    """Slots whose whole literal possibility set satisfies ``prop``."""
    good = jnp.sum(jnp.where(mask & ~prop, 1.0, 0.0), axis=(-1, -2)) < 0.5
    nonempty = jnp.sum(_one(mask), axis=(-1, -2)) > 0.5
    return good & nonempty & occupied


def _act(features, legal, memory):
    partner_hand, deck, height, tokens, lives, discards, knowledge = _decode(features)
    playable, useless, loss, in_play = _card_tables(height, discards)
    posterior, occupied, hinted = _own_belief(knowledge, in_play, partner_hand)

    # ---- own play candidates -------------------------------------------
    p_play = jnp.sum(posterior * playable[None], axis=(1, 2))
    exp_loss = jnp.sum(posterior * loss[None], axis=(1, 2))
    # A fresh partner hint is evidence, applied as a bounded odds multiplier:
    # it cannot make a literally impossible play look attractive.
    # trust shrinks as the bomb budget shrinks: on the last life the agent
    # falls back to purely literal evidence
    hint_odds = jnp.where(lives > 2.5, HINT_ODDS, jnp.where(lives > 1.5, HINT_ODDS_TWO, 1.0))
    odds = 1.0 + (hint_odds - 1.0) * memory
    trusted = p_play * odds / jnp.maximum(p_play * odds + (1.0 - p_play), 1e-6)
    play_value = jnp.where(occupied, trusted, -1.0)
    play_slot = jnp.argmax(play_value)
    best_play = play_value[play_slot]

    threshold = jnp.where(
        lives > 2.5, PLAY_THR_LIVES3,
        jnp.where(lives > 1.5, PLAY_THR_LIVES2, PLAY_THR_LIVES1),
    )
    # once the deck is empty an unplayed card is worth nothing, so risk more
    threshold = jnp.where((deck < 0.5) & (lives > 1.5), threshold - ENDGAME_RELIEF, threshold)
    can_play = best_play >= threshold
    play_threshold = threshold

    # ---- hint candidates ------------------------------------------------
    p_occupied = jnp.sum(partner_hand, axis=(1, 2)) > 0
    base_mask, touched, new_mask = _hint_tables(partner_hand, knowledge)
    certain_play_now = _certain(base_mask, playable, p_occupied)
    certain_play_new = _certain(new_mask, playable[None, None], p_occupied[None])
    certain_trash_now = _certain(base_mask, useless, p_occupied)
    certain_trash_new = _certain(new_mask, useless[None, None], p_occupied[None])

    is_playable = jnp.sum(partner_hand * playable[None], axis=(1, 2)) > 0
    is_useless = jnp.sum(partner_hand * useless[None], axis=(1, 2)) > 0
    card_loss = jnp.sum(partner_hand * loss[None], axis=(1, 2))

    # Model the partner's next move from what it can literally see: the slot it
    # would judge cheapest to throw away, and whether it has a play instead.
    p_hinted = jnp.sum(knowledge[1][:, 25:35], axis=1) > 0
    p_weights = _one(base_mask) * in_play[None, :, :]
    p_mass = jnp.sum(p_weights, axis=(1, 2))
    p_norm = p_weights / jnp.maximum(p_mass, 1e-6)[:, None, None]
    p_their_loss = jnp.sum(p_norm * loss[None], axis=(1, 2))
    p_their_play = jnp.sum(p_norm * playable[None], axis=(1, 2))
    their_odds = jnp.where(p_hinted, hint_odds, 1.0)
    their_trust = (p_their_play * their_odds
                   / jnp.maximum(p_their_play * their_odds + (1.0 - p_their_play), 1e-6))

    # What the partner can literally conclude after each candidate hint: the
    # public-count posterior of the touched slot being playable, with the same
    # bounded trust boost this policy applies to a freshly hinted slot.
    weights = _one(new_mask) * in_play[None, None, :, :]
    mass = jnp.sum(weights, axis=(2, 3))
    p_after = jnp.sum(weights * playable[None, None], axis=(2, 3)) / jnp.maximum(mass, 1e-6)
    odds_after = jnp.where(touched, hint_odds, 1.0)
    boosted = p_after * odds_after / jnp.maximum(p_after * odds_after + (1.0 - p_after), 1e-6)
    induced = (boosted >= play_threshold) & p_occupied[None] & (mass > 0.5)
    # Do not pay a token to re-announce a card the partner can already prove is
    # playable.  The test is literal certainty, not a model of how the partner
    # weighs evidence, so it stays correct for any partner.
    induced = induced & ~certain_play_now[None]
    induce_play = jnp.sum(_one(induced & is_playable[None]), axis=1)
    induce_bad = jnp.sum(_one(induced & ~is_playable[None]), axis=1)

    informative = jnp.sum(_one(new_mask), axis=(2, 3)) < jnp.sum(_one(base_mask), axis=(1, 2))[None]
    fresh = touched & informative & p_occupied[None]

    chop_value = jnp.where(
        p_occupied,
        -p_their_loss - DISCARD_HINTED * _one(p_hinted) - DISCARD_AGE * _one(_idx()),
        -1e9,
    )
    chop = jnp.argmax(chop_value)
    will_play = jnp.any(p_occupied & (certain_play_now | (their_trust >= play_threshold)))
    at_risk = ~will_play & (tokens < 7.5) & ~certain_play_now[chop]
    chop_loss = jnp.where(at_risk, card_loss[chop], 0.0)

    gain_play = jnp.sum(_one(certain_play_new & ~certain_play_now[None]), axis=1)
    gain_trash = jnp.sum(_one(certain_trash_new & ~certain_trash_now[None]), axis=1)
    touch_play = jnp.sum(_one(fresh & is_playable[None]), axis=1)
    touch_bad = jnp.sum(_one(fresh & ~is_playable[None] & ~is_useless[None]), axis=1)
    touch_trash = jnp.sum(_one(fresh & is_useless[None]), axis=1)
    touched_occ = touched & p_occupied[None]
    unsafe = jnp.any(touched_occ & ~is_playable[None], axis=1)
    saves = _one(touched[:, chop]) * chop_loss

    hint_score = (
        W_INDUCE_PLAY * induce_play
        - W_INDUCE_BAD * induce_bad
        + W_CERTAIN_PLAY * gain_play
        + W_CERTAIN_TRASH * gain_trash
        + W_TOUCH_PLAYABLE * touch_play
        - W_TOUCH_BAD * touch_bad
        - W_TOUCH_TRASH * touch_trash
        - W_UNSAFE * _one(unsafe)
        + W_SAVE * saves
        + W_TOUCH_ANY * jnp.sum(_one(touched), axis=1)
    )
    hint_legal = legal[10:20]
    hint_score = jnp.where(hint_legal, hint_score, -1e9)
    hint_index = jnp.argmax(hint_score)
    best_hint = hint_score[hint_index]
    hint_threshold = jnp.where(
        tokens < 1.5, HINT_THR_LOW,
        jnp.where(tokens < 3.5, HINT_THR_MID,
                  jnp.where(tokens < 5.5, HINT_THR_HIGH, HINT_THR_RICH)),
    )
    can_hint = jnp.any(hint_legal) & (best_hint >= hint_threshold)

    # ---- discard candidates ---------------------------------------------
    discard_value = (
        -exp_loss
        - DISCARD_HINTED * _one(hinted)
        - DISCARD_AGE * _one(_idx())
    )
    discard_value = jnp.where(occupied, discard_value, -1e9)
    discard_slot = jnp.argmax(discard_value)
    can_discard = jnp.any(legal[0:5])

    # ---- arbitration ------------------------------------------------------
    fallback = jnp.argmax(legal).astype(jnp.int32)
    action = jnp.where(
        can_play, 5 + play_slot,
        jnp.where(
            can_hint, 10 + hint_index,
            jnp.where(
                can_discard, discard_slot,
                jnp.where(jnp.any(hint_legal), 10 + hint_index,
                          jnp.where(jnp.any(legal[5:10]), 5 + play_slot, fallback)),
            ),
        ),
    ).astype(jnp.int32)
    action = jnp.where(legal[action], action, fallback)
    mode = jnp.where(can_play, 0, jnp.where(can_hint, 1, jnp.where(can_discard, 2, 3)))
    return action, best_play, best_hint, mode.astype(jnp.int32)


def _update_memory(features, memory):
    """Track which own slots were recently touched by a partner hint."""
    block = features[253:308]
    actor_partner = block[1] > 0.5
    is_color = block[4] > 0.5
    is_rank = block[5] > 0.5
    to_me = block[6] > 0.5
    matched = block[18:23] > 0.5
    hint_to_me = actor_partner & (is_color | is_rank) & to_me
    faded = jnp.where(memory * HINT_DECAY > HINT_FLOOR, memory * HINT_DECAY, 0.0)
    refreshed = jnp.where(hint_to_me, jnp.where(matched, 1.0, faded), memory)

    # our own play or discard compacts the hand, so the tracker shifts with it
    self_removed = (block[0] > 0.5) & ((block[2] > 0.5) | (block[3] > 0.5))
    slot = jnp.argmax(block[23:28])
    shifted = jnp.concatenate([refreshed[1:], jnp.zeros((1,), dtype=refreshed.dtype)])
    compacted = jnp.where(_idx() < slot, refreshed, shifted)
    return jnp.where(self_removed, compacted, refreshed)


def init_agent():
    return jnp.zeros((HAND_SIZE,), dtype=jnp.float32)


def agent_act(obs, reset, memory, rng):
    features = obs.features
    legal = obs.legal_actions
    memory = jnp.where(reset, jnp.zeros((HAND_SIZE,), dtype=jnp.float32), memory)
    memory = jnp.where(reset, memory, _update_memory(features, memory))

    action, best_play, best_hint, mode = _act(features, legal, memory)
    inactive = legal[NOOP]
    action = jnp.where(inactive, jnp.asarray(NOOP, dtype=jnp.int32), action)

    trace = {
        'mode': jnp.where(inactive, jnp.asarray(4, dtype=jnp.int32), mode),
        'best_play': best_play.astype(jnp.float32),
        'best_hint': best_hint.astype(jnp.float32),
    }
    return action.astype(jnp.int32), memory, trace
