> Copied from [H-group: The Critical Save (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/critical-save-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Critical Save (Question 1)

## Question

- Alice clues number 3 to Bob, touching a card on slot 3.
- Is this a _Play Clue_ or a _Save Clue_?
- What _card note_ does Bob write?

[Example diagram: CriticalSaveQuestion1](../../examples/critical-save-question-1/critical-save-question-1.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 0
  - b: 0
  - p: 3
discarded:
  - g3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: 3
        clue: 3
      - type: 4
      - type: 4
  - cards:
      - type: p4
        border: false
      - type: r2
      - type: y1
        border: false
      - type: y2
        border: false
      - type: y4
        border: false
```

## Solution

- Bob knows that this could be a _Play Clue_ on:
  - a yellow 3 (directly playable)
  - a red 3 (as a _Delayed Play Clue_ through Cathy's red 2)
- However, since it touched his chop, it could also be a _Save Clue_. Since green 3 is in the trash, a number 3 clue matches that, so it could be the matching green 3.
- Bob does not know whether or not this is a _Play Clue_ or a _Save Clue_, but he has to treat it as a _Save Clue_ for the time being until he gets more information.
- Bob writes a _card note_ of "r3, y3, g3".

[Example diagram: CriticalSaveAnswer1](../../examples/critical-save-question-1/critical-save-answer-1.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 0
  - b: 0
  - p: 3
discarded:
  - g3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: 3
        clue: 3
        below: sp
      - type: 4
      - type: 4
  - cards:
      - type: p4
        border: false
      - type: r2
      - type: y1
        border: false
      - type: y2
        border: false
      - type: y4
        border: false
```
