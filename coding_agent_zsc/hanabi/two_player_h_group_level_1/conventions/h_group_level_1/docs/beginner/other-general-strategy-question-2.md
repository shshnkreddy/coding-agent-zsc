> Copied from [H-group: Other General Strategy (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/other-general-strategy-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Other General Strategy (Question 2)

## Question

- It is the _Early Game_.
- Enumerate all of the legal clues that Alice can give.
- Of these possibilities, what clue should Alice give?

[Example diagram: OtherGeneralStrategyQuestion2](../../examples/other-general-strategy-question-2/other-general-strategy-question-2.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 2
  - b: 0
  - p: 3
bigText:
  text: Early Game
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: 2
  - cards:
      - type: b4
        border: false
      - type: p3
        border: false
      - type: r1
        border: false
      - type: g2
        border: false
      - type: b4
        border: false
  - cards:
      - type: r3
        border: false
      - type: g3
        border: false
      - type: y2
        border: false
      - type: p1
        border: false
      - type: r5
        border: false
```

## Solution

- Alice can clue green to Cathy as a _Play Clue_ on the green 3.
- Alice can clue number 5 to Cathy as a _5 Save_ on the red 5.
- It makes more sense for Alice to give the _Play Clue_ instead of the _Save Clue_, since Cathy won't be discarding anything on her turn if she is preoccupied with playing a card.

[Example diagram: OtherGeneralStrategyAnswer2](../../examples/other-general-strategy-question-2/other-general-strategy-answer-2.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 2
  - b: 0
  - p: 3
bigText:
  text: Early Game
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: 2
  - cards:
      - type: b4
        border: false
      - type: p3
        border: false
      - type: r1
        border: false
      - type: g2
        border: false
      - type: b4
        border: false
  - cards:
      - type: r3
        border: false
      - type: g3
        clue: g
        middleNote: g3
        below: Play
      - type: y2
        border: false
      - type: p1
        border: false
      - type: r5
        border: false
```
