> Copied from [H-group: The 2 Save (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/2-save-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The 2 Save (Question 1)

## Question

- Alice clues number 2 to Bob, touching his slot 4.
- From Bob's perspective, is this a _Play Clue_ or a _Save Clue_?
- What _card note_ does Bob write?

[Example diagram: TwoSaveQuestion1](../../examples/2-save-question-1/2-save-question-1.yml)

```yaml
stacks:
  - r: 3
  - y: 0
  - g: 0
  - b: 1
  - p: 2
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
      - type: x
      - type: 2
        clue: 2
      - type: x
  - cards:
      - type: y5
        border: false
      - type: y1
      - type: r3
        border: false
      - type: r2
        border: false
      - type: b2
        border: false
```

## Solution

- This cannot be a _Save Clue_ because the clue did not touch the card on Bob's chop. Thus, it must be a _Play Clue_.
- Bob writes a _card note_ that includes the following identities:
  - yellow 2 (as a _Delayed Play Clue_ through Cathy's yellow 1)
  - blue 2 (directly playable)

[Example diagram: TwoSaveAnswer1](../../examples/2-save-question-1/2-save-answer-1.yml)

```yaml
stacks:
  - r: 3
  - y: 0
  - g: 0
  - b: 1
  - p: 2
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
      - type: x
      - type: 2
        clue: 2
        middleNote: y2, b2
        below: Play
      - type: x
        below: Chop
  - cards:
      - type: y5
        border: false
      - type: y1
      - type: r3
        border: false
      - type: r2
        border: false
      - type: b2
        border: false
```
