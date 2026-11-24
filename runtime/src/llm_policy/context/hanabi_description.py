"""Dimension-derived factual descriptions for multiplayer Hanabi."""

from llm_policy.environments.hanabi.layout import hanabi_layout


def rules(contract):
    c = contract
    n, h = c['num_agents'], c['hand_size']
    layout = hanabi_layout(n, h)
    blocks, last = layout['observation_blocks'], layout['last_action_blocks']
    def interval(bounds):
        return f'[{bounds[0]}:{bounds[1]}]'
    descriptions = {
        'other_hands': f'Visible hands [{n-1},{h},5,5], relative players 1..{n-1}; one-hot identities in color-major order.',
        'missing_cards': f'{n} relative-player flags: 1 means fewer than {h} cards, not a missing-card count.',
        'deck': f'{layout["initial_deck_size"]}-entry remaining-deck thermometer.',
        'fireworks': 'Five rows of five: one-hot highest successful rank; zero row means height 0.',
        'information_tokens': 'Eight-entry information-token thermometer.',
        'lives': 'Three-entry life-token thermometer.',
        'discards': 'Five rows of ten: per-rank thermometer slices [0:3], [3:5], [5:7], [7:9], [9:10]. Includes failed plays, excludes successful plays.',
        'last_action': 'Most recent public event; all zero on reset. Offsets below are relative to this block.',
        'native_knowledge': f'[{n},{h},35], relative-player order: 25 V0 weights, five positive color flags, five positive rank flags per slot.',
    }
    table = '\n'.join(f'| {interval(bounds)} | {descriptions[name]} |' for name, bounds in blocks.items())
    meanings = {'actor': 'Acting relative player, one-hot.', 'kind': 'One-hot: play, discard, color hint, rank hint.',
                'target': 'Hint recipient, relative-player one-hot; zero for play/discard.',
                'color': 'Named color, one-hot; zero unless color hint.', 'rank': 'Named zero-based rank, one-hot; zero unless rank hint.',
                'matches': f'{h} recipient-slot match flags; zero for play/discard.',
                'slot': 'Removed slot before removal, one-hot; zero unless play/discard.',
                'card': 'Revealed played/discarded identity [5,5]; zero unless play/discard.',
                'scored': 'Boolean: play advanced a firework.',
                'added_token': 'Boolean: completing a firework actually increased tokens; zero for discards.'}
    events = '\n'.join(f'| {interval(bounds)} | {meanings[name]} |' for name, bounds in last.items())
    return f'''# {n}-player Hanabi: environment specification

## Rules and information boundary

All {n} players cooperate to build five fireworks in rank order 1..5. The 50-card
deck has per-color multiplicities [3,2,2,2,1]. Colors and array ranks are 0..4;
human ranks are 1..5. Reset shuffles the multiset, deals {h} cards to each player,
and leaves {layout['initial_deck_size']} cards. There are initially eight information
tokens, three lives, no hints, empty fireworks and no discards. Absolute player 0
moves first; turns advance cyclically through all {n} players.

A player sees every other hand, public board/counts, and the most recent public
action, but never their own card identities, deck order, other policies' private
observations or memory. Player indices in observations are relative: self=0,
next seat=1, then cyclically up to {n-1}. There is no absolute-seat feature.
Every policy instance processes every transition; only the active seat acts.

## Actions and legality

A scalar integer indexes a legal-action mask of length {layout['num_actions']}.
For own slot s in 0..{h-1}, color c or rank r in 0..4, and relative recipient
t in 1..{n-1}:

| Action | ID | Rule |
|---|---|---|
| Discard | s | Occupied slot and fewer than eight tokens; reveal/discard, restore one token. |
| Play | {h}+s | Any occupied slot. Success iff rank ID equals that color's firework height; otherwise lose a life and discard. |
| Color hint | {2*h}+5*(t-1)+c | Requires a token and at least one matching card in recipient t's hand. |
| Rank hint | {2*h+5*(n-1)}+5*(t-1)+r | Requires a token and at least one matching card in recipient t's hand. |
| Inactive no-op | {layout['actions']['inactive_noop']} | Only legal for an inactive seat; active seats cannot pass. |

A hint costs one token and reveals ALL matching slots, including negative
information for unmatched occupied slots. Repeated hints are legal; empty hints
are not. Successful rank-5 plays restore a token up to eight. Plays and discards
remove the chosen slot, shift later slots left, and append a fresh draw at slot
{h-1}, or an empty slot if the deck was empty. Hints follow surviving cards; new
cards have no hints. Card slots are positions, not permanent identities.
Illegal active actions invalidate evaluation; inactive actions are forced to no-op.

## Observations

HanabiObservation contains features[{layout['observation_size']}] and
legal_actions[{layout['num_actions']}]. All intervals are half-open. Thermometer
counts are sums. One-hot indices are argmax only when an event is present.

| Feature interval | Meaning |
|---|---|
{table}

### Public last action: local offsets

| Local interval | Meaning |
|---|---|
{events}

Hands, board, discards and knowledge describe the post-action state. The removed
slot describes the pre-action hand. The event is replaced after each transition;
process it once, including on inactive turns. Do not interpret inapplicable fields.

Each occupied card's native V0 weights multiply literal possibilities by public
remaining card counts and normalize per slot. They do not subtract visible hands
or enforce joint counts across slots. Positive hint flags alone omit negative
information. Empty slots have zero V0 and hint rows. These are native observation
features, not exact posteriors, strategic recommendations, or convention inferences.
All players' native knowledge is based on public hints, not private policy memory.

## Rewards, completion and resets

Each successful play gives shared +1. Losing the final life subtracts the current
firework score, making episode return zero. Other actions give zero reward.
Completing all fireworks ends with 25. Final-life termination correction is
{c['fix_final_life_termination']}; when enabled the last-life loss ends immediately.
Drawing the final card starts {n} further player turns unless lives or completion
end the game first. The deck becoming empty alone does not end play. Final-round
return is the sum of firework heights unless all lives were lost.

The evaluation cap is {c['max_steps']} transitions; incomplete episodes are rejected.
The adapter resets automatically after termination. At the next policy call,
reset=True and private memory returns to its initializer. Policy inputs contain
no runtime state or reward. No rule here prescribes a signaling convention.
'''
