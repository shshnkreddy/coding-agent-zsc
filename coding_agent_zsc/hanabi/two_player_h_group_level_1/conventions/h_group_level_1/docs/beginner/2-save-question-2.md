> Copied from [H-group: The 2 Save (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/2-save-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The 2 Save (Question 2)

## Question

- Previously in the game, Bob was given a _Play Clue_ on a 1. He had planned on playing it on his next turn.
- Now, Alice clues number 2 to Bob, touching his slot 5.
- From Bob's perspective, is this a _Play Clue_ or a _Save Clue_?
- What _card note_ does Bob write? (Be specific.)

[Example diagram: TwoSaveQuestion2](../../examples/2-save-question-2/2-save-question-2.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 2
  - b: 0
  - p: 0
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: 1
      - type: x
      - type: x
      - type: x
      - type: 2
        clue: 2
  - cards:
      - type: g5
        border: false
      - type: b3
        border: false
      - type: g4
        border: false
      - type: p3
        border: false
      - type: p3
        border: false
```

## Solution

- Bob already has a _card note_ of "r1, b1, p1" on his 1 (because these are the 1's left to play on the stacks).
- If this is a _Play Clue_, then the identity of his 2 would be:
  - a yellow 2 (directly playable)
  - or a _Delayed Play Clue_ on the 2 that matches his unknown 1 (e.g. red 2, blue 2, or purple 2)
- However, since Alice's clue touched Bob's chop, it could also be a _Save Clue_. If it is a _Save Clue_, then it could be a _2 Save_ on an unplayable 2.
- Bob does not know whether or not this is a _Play Clue_ or a _Save Clue_, but he has to treat it as a _Save Clue_ for the time being until he gets more information.
- Thus, Bob writes a _card note_ of "r2, y2, b2, p2".
  - (It cannot be a green 2 because that card is already played and is useless to the team; Alice would not waste a clue on a useless card like that.)

[Example diagram: TwoSaveAnswer2](../../examples/2-save-question-2/2-save-answer-2.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 2
  - b: 0
  - p: 0
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: 1
      - type: x
      - type: x
      - type: x
      - type: 2
        clue: 2
        below: sp
  - cards:
      - type: g5
        border: false
      - type: b3
        border: false
      - type: g4
        border: false
      - type: p3
        border: false
      - type: p3
        border: false
```
