"""Factual card-game descriptions for contexts without simulator source."""


def hanabi(record):
    c = record['runtime_contract']
    if c['num_agents'] > 2:
        from llm_policy.context.hanabi_description import rules
        return rules(c)
    obs = record['observation']
    return f"""# Two-player Hanabi: environment specification

## Goal, objects and initial state

The two players cooperate to build five public fireworks, one per color. Each
firework must contain ranks 1, 2, 3, 4, 5 in that order, without duplicates or
skipping ranks. The episode return is the team's score, shared by both players.
The maximum is 25. This specification describes game mechanics and data formats;
no extra meanings are assigned to a player's choice of action.

The game contains 50 cards: five colors, with 3 copies of rank 1, 2 of each of
ranks 2, 3 and 4, and 1 of rank 5 per color. Color IDs are 0..4. Array rank IDs
are also 0..4, corresponding to human ranks 1..5. There is no special color.
A fresh episode uniformly shuffles this fixed multiset, deals five cards to
player 0 and five to player 1, and leaves 40 cards in the draw pile. Player 0
acts first. All fireworks and the discard pile start empty, with eight
information tokens and three lives. There are no hints on the initial hands.

Each player sees the identities and ordering of the other player's cards but
not the identities of their own cards or the order of the draw pile. Both see
public firework heights, discard counts, information tokens, lives, deck size,
and the most recent action and its public outcome. Hints and the identities
of played/discarded cards are public. The partner's private memory is never an
observation. The player's own private memory may persist within an episode.

## Turn sequence and action space

One environment transition executes one active player's action, then changes
the active player. Both policy instances are called on every transition, so
both receive the public result of the previous turn. Only one seat is active.
The action is a scalar integer in 0..20; the action mask has 21 booleans.
A true mask entry means the action is allowed by the game rules, not that it
will score points. In particular, legality of playing a card does not reveal
whether that card will succeed.

| IDs | Action | Exact legality and effect |
|---|---|---|
| 0..4 | Discard own slot s: action = s | The slot must be occupied and fewer than 8 information tokens remain. Reveal and discard that card, restore one information token, then replace the card as described below. No life is lost. |
| 5..9 | Play own slot s: action = 5+s | Any occupied slot may be played. Reveal the card. It succeeds exactly when its rank ID equals that color's current firework height. Success increases the height by one. Otherwise discard it and lose one life. Replace the removed card. |
| 10..14 | Hint color c to the partner: action = 10+c | At least one information token remains and at least one partner card has color c. Spend one token and identify every matching slot. |
| 15..19 | Hint rank r to the partner: action = 15+r | At least one information token remains and at least one partner card has rank ID r. Spend one token and identify every matching slot. |
| 20 | Inactive no-op | This is the only legal action for the inactive seat; the active seat cannot pass. |

A successful rank-5 play also restores one information token, capped at eight.
An unsuccessful play does not restore a token. Hints may be repeated, even if
all their literal information was already known. An empty hint is illegal.
A hint refers to all matching cards; the sender cannot select only some of
them. For the named attribute, matched slots have that value and unmatched
occupied slots do not. Hinting changes no card positions and draws no cards.

Whenever a card is played or discarded, remove that slot, shift all later
slots one position left, and append the next draw at slot 4 if the deck was
nonempty. Otherwise append an empty slot. This happens regardless of whether
a play succeeds. Public hint records move with the shifted cards. A new draw
has no previous hints. Slots are current positions, not stable card IDs.
The newly drawn own card remains hidden to its owner and becomes visible to
the partner. Discarded cards and failed plays cannot be recovered.

## Observation arrays

The input is HanabiObservation with two fields:

- features: shape {obs['features']['shape']}, dtype {obs['features']['dtype']}.
- legal_actions: shape {obs['legal_actions']['shape']}, dtype {obs['legal_actions']['dtype']}.

Features are a flat vector of length 658. All intervals below are half-open.
A one-hot vector contains a single 1 at its indexed category; an all-zero vector indicates
absence where specified. A thermometer with value k contains k leading ones
and zeros thereafter. Counts are obtained by summing the indicated thermometer.
Indices for players are relative to the observer: 0 means self and 1 partner.
There is no separate absolute-seat or turn-counter feature. The no-op mask
entry identifies inactivity; it is false for the active seat.

| Feature interval | Shape / meaning |
|---|---|
| [0:125] | Partner hand, reshaped [5 slots, 5 colors, 5 ranks]. Entry 25*s+5*c+r is 1 for the card in partner slot s. An empty slot has 25 zeros. |
| [125:127] | Two missing-card flags in relative-player order. A flag is 1 when that hand has fewer than five cards; it is not a missing-card count. |
| [127:167] | Remaining draw-pile size, a 40-entry thermometer. Initially all 40 entries are 1. |
| [167:192] | Fireworks, reshaped [5 colors, 5 ranks]. Each color is one-hot at its highest successfully played rank ID. An all-zero vector indicates height 0; a one at rank ID r means height r+1. These rows are not thermometers. |
| [192:200] | Information tokens, an 8-entry thermometer. |
| [200:203] | Lives, a 3-entry thermometer. |
| [203:253] | Discard counts, reshaped [5 colors, 10 entries]. Within each color the five rank subblocks are [0:3], [3:5], [5:7], [7:9], [9:10]. Each subblock is a thermometer for copies of that rank discarded, including failed plays. Successful firework cards are not in this pile. |
| [253:308] | Most recent public action, detailed next. Entirely zero on reset. |
| [308:658] | Native per-card knowledge features, reshaped [2 relative players, 5 slots, 35 entries], detailed below. |

### Most recent action: offsets relative to 253

| Local interval | Meaning |
|---|---|
| [0:2] | Acting player, one-hot in observer-relative order. |
| [2:6] | Action type, one-hot in order play, discard, color hint, rank hint. |
| [6:8] | Hint recipient, observer-relative one-hot; zero for play/discard. |
| [8:13] | Named color, one-hot for a color hint; otherwise zero. |
| [13:18] | Named rank ID, one-hot for a rank hint; otherwise zero. |
| [18:23] | Five hint-match flags, indexed by recipient slot. Unmatched slots are 0. Zero for play/discard. |
| [23:28] | Removed slot, one-hot for play/discard, referring to the actor's hand before removal. Otherwise zero. |
| [28:53] | Revealed played/discarded identity, [5 colors, 5 ranks] one-hot. Otherwise zero. |
| [53] | Boolean: this action added a card to a firework. |
| [54] | Boolean: a play actually increased information tokens by completing a color. Zero for discards, even though a discard restores a token. |

The hand, board, discard and knowledge blocks describe the state after this
action; the removed-slot field refers to the state before it. The last-action
block is replaced on every transition and is not an accumulated history.
The active actor therefore alternates, but both seats can read each result.

### Native knowledge block: 35 entries per current slot

Entries [0:25] are the environment's native V0 weights, flattened color-major
as 5*c+r. Entries [25:30] are positive color-hint flags and [30:35] positive
rank-hint flags. Positive flags accumulate on matching cards and shift on
removal; absence of a positive flag does not by itself encode a negative hint.

For an occupied slot, V0 multiplies the native literal-possibility mask by the
remaining public counts for each card type, then normalizes this slot's 25
values to sum to one. Remaining public count is the original multiplicity
minus discarded copies minus the copy already in that color/rank firework
position. It does not subtract cards visible in the partner's hand or enforce
joint card-count consistency across slots. V0 is thus an observation feature,
not an exact joint distribution over hidden hands and not a prediction of
what another policy intends. The native possibility mask starts unrestricted,
is reduced by matching and nonmatching hint evidence, shifts with card removal,
and is reset for a new draw. For an empty slot, all V0 entries and positive
hint flags are zero. Both players' native knowledge records appear in relative order;
they are based on public hints, not private policy memory.

## Rewards and episode termination

Every successful play gives shared reward +1. Hints, discards and unsuccessful
plays that leave lives remaining give 0. On loss of the final life, the current
firework score is subtracted, making total episode return 0 (bomb-zero scoring).
The configured correction ends the episode immediately on final-life loss:
{c['fix_final_life_termination']}. Completing all five fireworks ends the episode
with total return 25.

When the last draw is taken, the native final-round counter begins. There are
two further active-player turns: one for the other player and one for the
player who drew the last card, unless another termination condition occurs
first. There are no further draws. Hints still consume these turns. At final
round completion, the return is the sum of firework heights, unless all lives
were lost. The end of the deck alone does not immediately end the episode.
The safety cap is {c['max_steps']} transitions per evaluation episode; incomplete
episodes are rejected, not scored as ordinary successes or failures.

The adapter automatically resets after a terminal transition. At the following
policy call reset is true, the observation belongs to the next shuffled deal,
and private memory is reset to the value returned by its initializer. Policy inputs never include
hidden environment state or the reward. Evaluation tools report returns and
public observations separately. Illegal active actions invalidate evaluation;
the inactive seat is forced to no-op by the adapter.
"""


def yokai(record):
    c = record['runtime_contract']
    n, k, g, h = (c[key] for key in ('deck_size','num_colors','grid_size','num_hints'))
    side = 3 if n == 9 else 4
    origin = g//2-side//2
    move = 1+n
    reveal = move+n*g*g
    place = reveal+h
    end = c['num_actions']-1
    obs = record['observation']
    schema = '\n'.join(f"| {name} | {item['shape']} | {item['dtype']} |" for name,item in obs.items())
    hints = ('Hint 0 has one color; hints 1, 2, 3 each have two colors. The single\ncolor is sampled uniformly from the three colors. The three two-color hints\nare the three distinct unordered color pairs, in random order.' if n==9 else
             'Hints 0, 1 each have one color, hints 2, 3, 4 each have two colors, and\nhints 5, 6 each have three colors. Within each size, distinct color subsets\nare sampled without replacement from all subsets of the four colors.')
    memory = ('With perfect memory enabled, during your active turns the own-color\nfeatures expose every card you have personally observed at any earlier time\nin this episode, including moved or subsequently locked cards. Identity is\nattached to stable card ID. While your seat is inactive, all own-color features\nare zero, including previously observed colors. They reappear when your seat\nis active again. The other player’s private color observations are never\nadded to your own-color features.' if c['past_colors_revealed'] else
              'With perfect memory disabled, own-color features expose only cards\nobserved during the current active turn. Those colors disappear at turn end.\nDuring inactivity all own-color features are zero. The policy may retain\nits own past observations in its private memory; the environment does not\nrestore them in later observations.')
    failures = ('If any color is not clustered, the terminal reward is 0 regardless of\nhint correctness or the explicit end action.' if c['reward_mode']=='upstream_sparse' else
                f'If any color is not clustered, the terminal reward is\n- E - ({k} - K) - I, where E is 1 for an explicit end-game action and 0 otherwise,\nK is the number of connected color groups, and I is the number of incorrect\nplaced hints. These are terminal penalties only.')
    return f"""# Two-player {n}-card Yōkai: environment specification

## Goal, board and objects

The players cooperate to rearrange {n} cards so that all cards of each color
form one orthogonally connected group. There are {k} colors (IDs 0..{k-1}),
with {side} cards of each color. A connection uses horizontal or vertical
neighbors, never diagonals. For a color to count as connected, paths between
its cards must use only cards of that color. Different colors may touch; there
is no requirement to separate them by empty space. Colors need not form rows,
columns or any prescribed shape. The entire board must remain connected by
orthogonal card adjacency after each legal move, regardless of colors.

Cards occupy distinct cells of a {g}×{g} finite grid. Coordinates (x,y) use
x for the first array coordinate and y for the second; each is in 0..{g-1}.
There is no wraparound. All card positions, stable card IDs and locks are
public. Cards begin face down, and private observations reveal their colors only
to the observing player. There are no walls, draw pile, replacement cards,
color changes or removed cards.

There are also {h} hint tiles. Each shows a set of colors when revealed. A tile
can later be placed on a card, locking it for the rest of the episode. Hints
are separate objects from cards and do not participate in color connectivity.
A hint may legally be placed on a card whose true color is absent from its
set. This is an incorrect hint for scoring, but the move is still legal and
does not reveal the hidden card color. No convention for interpreting a
player's choice of hint, observation or movement is supplied.

## Action space and turn sequence

Every call returns one scalar integer in 0..{end}; obs.legal_actions is a
boolean vector of length {c['num_actions']}. Player 0 starts. Each full turn
contains four substeps by the same active player, in order 0, 1, 2, 3. After
substep 3 the other player becomes active at substep 0. Both policy instances
receive observations and are called at every substep; the inactive seat's
only legal action is no-op 0. The adapter forces that seat to no-op. The active
seat cannot pass. A true legal-action entry reflects public rule constraints,
not whether a move improves the hidden-color arrangement.

| Substep | Action and ID formula | Legal choices and effect |
|---|---|---|
| 0 | Observe card i: 1+i; or end game: {end} | Observe any unlocked card, revealing its color privately for this turn, or terminate immediately and score the current arrangement. |
| 1 | Observe card i: 1+i | Observe a different unlocked card, not the one already observed this turn. No end-game action is allowed here. |
| 2 | Move card i to (x,y): {move} + i*{g*g} + x*{g} + y | Move any unlocked card to an empty in-grid cell, different from its current cell, such that all {n} cards are connected afterward. The moved card need not be one of the two observed cards. |
| 3 | Reveal hint h: {reveal}+h; or place hint h at (x,y): {place} + h*{g*g} + x*{g} + y | Reveal one unrevealed hint, making its color set public; alternatively place one already revealed, unplaced hint on an unlocked card occupying (x,y). |

The action intervals are: no-op [0:1], observe [1:{move}], move
[{move}:{reveal}], reveal-hint [{reveal}:{place}], place-hint [{place}:{end}],
and end-game [{end}:{end+1}]. IDs i and h are zero-based. Place actions address
a grid cell, not a card ID; that cell must currently contain an unlocked card.

A move is a relocation, not a path traversal: only the destination and the
resulting whole-board connectivity are tested. There is no movement distance
cost or requirement that the remaining board stay connected during an imagined
intermediate removal. The moved card retains its ID, color and both players'
observation history. Locked cards cannot move or be observed again. There is
no swap action, no staying in the same cell, and no moving a hint separately.

Revealing and placing a hint are separate actions on separate turns. A revealed
hint cannot be hidden again. A placed hint cannot be moved, replaced or removed.
Only one hint can occupy a card because placement locks it. Revealing or placing
one hint completes substep 3. The game terminates immediately when all {h} hints
have been placed. Otherwise play continues even if all color groups happen to
be connected: connectivity alone does not automatically end the episode.

## Observation arrays and exact fields

The input is YokaiObservation with these fields:

| Field | Shape | Dtype |
|---|---|---|
{schema}

There are {n+h} graph nodes: rows 0..{n-1} are stable card IDs, and rows
{n}..{n+h-1} are hint IDs 0..{h-1}. Node rows do not reorder when objects move.
The graph has {2*k+7} features per node. Small integer values and binary
indicators are stored directly; coordinates and IDs are not normalized.

| Feature columns | Card row | Hint row |
|---|---|---|
| [0:{k}] | Privately visible true color, one-hot; all zero if unknown/masked. | Always zero. |
| [{k}:{2*k}] | Always zero, including cards with placed hints. | Revealed hint color set, multi-hot; all zero before reveal. |
| [{2*k}:{2*k+2}] | Current (x,y) position. | Current (x,y) position: before placement (h,-1), afterward the target card's position. |
| [{2*k+2}] | 1 if locked by a hint, else 0. | 1 if placed, else 0. |
| [{2*k+3}] | Public indicator that the other seat observed this card, with memory timing described below. | Always zero. |
| [{2*k+4}] | 1 if this observation's seat is active, else 0; repeated on every row. | Same active-seat flag. |
| [{2*k+5}] | Current substep 0..3; repeated on every row. | Same substep. |
| [{2*k+6}] | Stable card ID 0..{n-1}. | Negative stable ID: hint h has value -(h+1). |

An all-zero card-color vector indicates an unknown or masked color, not an additional color. Hint color sets are
found on hint nodes; they are not copied into the card row's hint-color columns.
An unplaced hint's off-board coordinate is not a legal card destination.
The separate positions array has shape [{n},2] and contains only card positions;
it duplicates the card-node coordinate columns, without hint rows.

Adjacency is a symmetric binary [{n+h},{n+h}] matrix. All nodes have self-loops.
Two different card nodes are connected exactly when their positions are one
orthogonal grid step apart. A placed hint node is connected to its target
card node in both directions. An unplaced hint has no non-self edges. There
are no other hint-card edges and no edges between distinct hint nodes.
Colors do not affect adjacency, and a hint edge does not connect two cards.

## Visibility and memory: {c['memory_mode']}

{memory}

The first observation action reveals its card in the returned substep-1
observation. After the second observation, both are visible at substep 2 and
remain so through substep 3, even if one is moved. At turn end the current-turn
reveal flags are cleared. Movement does not reveal an unobserved card's color.
All positions, locks, revealed hint sets, placements, substep and active-seat
flags remain visible to both seats, including the inactive player.

With perfect memory, the partner-seen column records whether the other player
has ever observed each card in this episode. Without perfect memory, it reports
observations added during the current turn, and is cleared at every full-turn
boundary. This indicator says which IDs were observed, never what colors the
other player saw. Hint colors become public on reveal and remain public after
placement. A known locked card's previously observed color still follows the
configured memory rule; locking does not communicate that color to a partner.

There is no explicit last-action field, episode counter, hidden-color belief
vector or simulator state in the observation. Policy memory and PRNG are
private to each seat. The observation contains no recommendation of which
card to observe, where to move it, which hint to use, or when to end.

## Reset distribution and hint identities

A reset shuffles the fixed multiset of card colors over stable IDs. Card i
starts at (floor(i/{side})+{origin}, (i mod {side})+{origin}), producing a centered
{side}×{side} block. All cards are unlocked and unobserved; hint tiles are unrevealed
and unplaced at (h,-1). The starting player is player 0 and the substep is 0.
The initial adjacency follows these positions. Initially no true card colors
or hint color sets are exposed to either seat.

{hints}
Hint IDs retain their sampled sets for the whole episode. The size category
of each hint ID is fixed by this reset rule; its sampled color set is hidden
until reveal. Sampling does not depend on whether a hint matches a card later
chosen for placement. A board may already have all colors connected at reset,
but that fact is not revealed by a separate observation flag.

## Terminal scoring and episode boundaries

Configured scoring is {c['reward_mode']}. All nonterminal rewards are zero.
At termination, if every color is connected, the shared reward is

5*U + 2*R + C - I,

where U is the number of unrevealed hints, R revealed but unplaced hints,
C correctly placed hints and I incorrectly placed hints. These categories
partition the {h} hints. A placed hint is correct exactly when the underlying
card's color belongs to its revealed set. A successful game can therefore
have a negative score if enough placed hints are incorrect. {failures}

The configured return bounds are {c['score_min']}..{c['score_max']}. For example,
ending immediately on a board where each color forms a connected group gives {5*h}; with all hints
placed on such a board the score is C-I. If each color forms a connected group and all {h}
hints are wrong, the score is {-h}, explaining the negative lower bound even with sparse
rewards. If any color's cards do not form a connected group, the configured failure reward applies. There is no
separate bonus for forming a cluster and no intermediate shaping reward.

Each hint can be revealed once and placed once; substep 3 must do one of those
actions. Thus at most {2*h} full turns ({8*h} substeps) occur before every hint
is placed, unless the players explicitly end earlier at substep 0. The
evaluation safety cap is {c['max_steps']} substeps. An illegal active action or
failure to terminate within the cap invalidates evaluation. After termination
the adapter returns a fresh reset observation, and the next policy call's
reset flag resets private memory. Episode metrics and returns are recorded
separately; rewards and hidden state are never policy inputs.
"""
