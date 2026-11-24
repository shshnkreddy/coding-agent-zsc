> Copied from [H-group: Other General Strategy (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/other-general-strategy-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Other General Strategy (Question 3)

## Question

- Enumerate all of the legal clues that Alice can give.
- Of these possibilities, what clue should Alice give?

[Example diagram: OtherGeneralStrategyQuestion3](../../examples/other-general-strategy-question-3/other-general-strategy-question-3.yml)

```yaml
stacks:
  - r: 3
  - y: 1
  - g: 3
  - b: 0
  - p: 3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: y3
        border: false
      - type: r2
        border: false
      - type: g2
        border: false
      - type: r1
        border: false
      - type: b5
        border: false
  - cards:
      - type: b4
        border: false
      - type: y2
      - type: p4
        border: false
      - type: r3
        border: false
      - type: p1
        border: false
```

## Solution

- Alice can clue yellow to Bob as a _Play Clue_ on the yellow 3 (through Cathy's yellow 2).
- Alice can clue number 3 to Bob as a _Play Clue_ on the yellow 3 (through Cathy's yellow 2).
- Alice can clue number 5 to Bob as a _5 Save_ on the blue 5.
- Normally, Alice would prefer to give a _Play Clue_ over a _Save Clue_. However, in this case, her _Play Clue_ would be a _Delayed Play Clue_, because Bob would not be able to play the card yet. (Bob would have to wait for Cathy to play the yellow 2.)
- So, if Alice gave a _Play Clue_ to the yellow 3, then Bob might have nothing to do and might discard the critical blue 5. This means that Alice must perform some other action.
- Thus, Alice should clue number 5 to Bob as a _5 Save_.

[Example diagram: OtherGeneralStrategyAnswer3](../../examples/other-general-strategy-question-3/other-general-strategy-answer-3.yml)

```yaml
stacks:
  - r: 3
  - y: 1
  - g: 3
  - b: 0
  - p: 3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: y3
        border: false
      - type: r2
        border: false
      - type: g2
        border: false
      - type: r1
        border: false
      - type: b5
        clue: 5
        below: Save
  - cards:
      - type: b4
        border: false
      - type: y2
      - type: p4
        border: false
      - type: r3
        border: false
      - type: p1
        border: false
```
