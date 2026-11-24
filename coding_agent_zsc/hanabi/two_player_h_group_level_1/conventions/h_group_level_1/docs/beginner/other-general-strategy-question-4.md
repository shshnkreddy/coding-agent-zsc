> Copied from [H-group: Other General Strategy (Question 4)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/other-general-strategy-question-4.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Other General Strategy (Question 4)

## Question

- Enumerate all of the legal clues that Alice can give.
- Of these possibilities, what clue should Alice give?

[Example diagram: OtherGeneralStrategyQuestion4](../../examples/other-general-strategy-question-4/other-general-strategy-question-4.yml)

```yaml
stacks:
  - r: 4
  - y: 1
  - g: 4
  - b: 1
  - p: 1
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r3
        border: false
      - type: b2
        border: false
      - type: y1
        border: false
      - type: p1
        border: false
      - type: p1
        border: false
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

- Alice can clue blue to Bob as a _Play Clue_ on the blue 2.
  - If she did that, Bob would write a _card note_ of "b2".
- Alice can clue number 2 to Bob as a _Play Clue_ on the blue 2.
  - If she did that, Bob would write a _card note_ of "y2, b2, p2".
- By default, Alice knows that color clues should be preferred over number clues. Specifically, in this situation, Alice knows that a blue clue will communicate the exact identity of the card, while a number clue would communicate three different possibilities.
- Thus, Alice should give a blue clue to Bob.

[Example diagram: OtherGeneralStrategyAnswer4](../../examples/other-general-strategy-question-4/other-general-strategy-answer-4.yml)

```yaml
stacks:
  - r: 4
  - y: 1
  - g: 4
  - b: 1
  - p: 1
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r3
        border: false
      - type: b2
        clue: b
        middleNote: b2
        below: Play
      - type: y1
        border: false
      - type: p1
        border: false
      - type: p1
        border: false
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
