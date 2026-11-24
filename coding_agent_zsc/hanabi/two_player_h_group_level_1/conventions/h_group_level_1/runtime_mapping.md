# Mapping H-group documentation to this task

This is an explicit adaptation for the repository's standard two-player Hanabi
environment. Game rules and encodings are in the task specification and policy
API. The copied H-group prose uses its original display conventions.

## Hand order and slots

H-group draws on the left: leftmost is newest, rightmost is oldest. Native
arrays have the opposite order: removal shifts later entries toward index zero,
and a draw appends at the end. For this bundle, interpret a native hand at reset
and thereafter as the sequence of **occupied native slots in descending index
order** when reading H-group examples or positional rules.

| H-group description | Native interpretation |
| --- | --- |
| Leftmost eligible card | Highest occupied native index satisfying the condition |
| Rightmost eligible card | Lowest occupied native index satisfying the condition |
| Chop | Lowest occupied native index among unclued cards |
| Finesse position | Highest occupied native index among unclued cards |
| Leftmost newly touched / re-touched card | Highest occupied native index in the corresponding set |
| Leftmost compatible prompted card | Highest occupied native index among compatible clued candidates |

For a full hand, H-group slots `1,2,3,4,5` map to native indices `4,3,2,1,0`.
For occupied native indices `0,1,2`, H-group's left-to-right hand is `2,1,0`;
do not use the fixed full-hand formula after the hand shrinks.

Example: native indices `1` and `3` are clued and all five slots are occupied.
Chop is native `0`; finesse position is native `4`. If a clue newly touches
native `2` and `4` and misses chop, focus is native `4`. If it also touches chop,
focus is native `0`. Determine these sets before applying the clue's new tags.

On removing native slot `1`, the old slots `2,3,4` move to `1,2,3`. A draw is a
fresh card at `4`; without a draw the appended entry is empty. Apply the same
remapping to literal constraints, inferred identities, saved status, and pending
dependencies. Never move an old card's note onto a new draw. An all-clued hand
has no ordinary chop; an empty slot is never chop, focus, or finesse position.
The Level 1 treatment of already finessed cards still applies when determining
which cards are conventionally new.

## Ranks, suits, players, and actions

- Human rank `r` in `1..5` is native rank index `r - 1` in `0..4`.
- Native suits are anonymous IDs. For translating these examples, use the
  fixed label legend `r=0, y=1, g=2, b=3, p=4` (red, yellow, green, blue, purple).
  This is a documentation label assignment, not an additional game variant.
- Alice/Bob/Cathy/etc. denote positions in an example's turn order, not special
  policy roles. Actual game player 0 moves first. Public last-action player
  fields are observer-relative as documented in the policy API; do not equate
  their indices with the example names or absolute seats.
- Encode plays/discards with native indices, not H-group slot numbers. The legal
  mask determines which actual actions may be returned. No runtime arrays or
  action IDs are reversed by enabling this context.

## Which special moves apply to two players?

The guide teaches several ideas using three or more players. Preserve the
example's distinct clue giver, intermediate reactor, and clue recipient when
reasoning about its premises. Ordinary examples of a finesse use those three
distinct players; this task has no third player. Do not collapse them into a
Level 2 self-finesse, reverse finesse, or any invented two-player signal.

The [prompt chapter](docs/beginner/prompt.md) explicitly includes a **self-prompt**:
the recipient already has a clued connector and a new play clue establishes the
connection. That is part of this Level 1 source and can apply to two players.
For instance, its rank-4 example connects through an already-clued rank 3 before
the rank 4 plays. Follow the documented evidence and retained ambiguity; this is
not permission to blind-play an unclued connector.

Delayed play clues, saves, clue focus, the principles, and early-game reasoning
remain relevant when their premises hold. Only observations and public history
may establish those premises; the example author's knowledge is not an extra
runtime observation.

## Reading the original YAML diagrams

The local Markdown expands each diagram at its original location. The linked
`.yml` file is an unmodified copy of the upstream diagram data. These diagrams
are pedagogical states, not executable policy examples or complete game seeds.

| Field or notation | Meaning |
| --- | --- |
| `stacks` | Played stack heights, keyed by suit abbreviation |
| `players` / `cards` | Example participants and cards, in H-group order |
| `clueGiver: true` | Participant giving the illustrated clue |
| `type: x` | Unclued card with identity unspecified in this depiction |
| `type: r`, `type: 3`, `type: r3` | Displayed color, rank, or identity information |
| `type: b234`, `middleNote: gb4` | Possibility shorthand, e.g. blue 2/3/4 or green/blue 4 |
| `border: false` | Displayed identity need not have been clued; suppress the usual clued border |
| `clue` / `retouched: true` | Current clue's touch / an already-clued card being re-touched |
| `middleNote` | The player's inferred card note, which may exceed literal information |
| `above` / `below` | Explanatory identity or action labels; not extra information visible to every player |
| `trash`, `crossedOut` | Useless-card annotation or excluded possibilities |
| `text`, `space`, `offset`, `name` | Narrative/layout or repeated-player labels, not game moves |

Read a diagram together with its surrounding explanation and question/solution
heading. Some deliberately show invalid clues or misconceptions. Omitted stack
or card details are unspecified, not necessarily empty or known.

The notation explanation is adapted from the upstream
[example-image format documentation](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/example-images.mdx).
No higher-level examples from that technical page are included here.
