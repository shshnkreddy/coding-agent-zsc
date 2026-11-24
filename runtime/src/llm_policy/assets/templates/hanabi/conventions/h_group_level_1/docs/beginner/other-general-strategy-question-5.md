> Copied from [H-group: Other General Strategy (Question 5)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/other-general-strategy-question-5.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Other General Strategy (Question 5)

## Question

- Enumerate all of the legal clues that Alice can give.
- Of these possibilities, what clue should Alice give?

[Example diagram: OtherGeneralStrategyQuestion5](../../examples/other-general-strategy-question-5/other-general-strategy-question-5.yml)

```yaml
stacks:
  - r: 5
  - y: 0
  - g: 2
  - b: 4
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
      - type: y5
        border: false
      - type: b4
        border: false
      - type: g3
        border: false
      - type: y3
        border: false
      - type: r1
        border: false
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

- Alice can clue green to Bob as a _Play Clue_ on the green 3.
  - This would "get" one card.
- Alice can clue number 3 to Bob as a _Play Clue_ on the green 3.
  - This would "get" two cards. (Bob would play the green 3 and hold on to the yellow 3 for later.)
- Alice knows that normally, color clues are preferred to number clues. But that's only when the color clue and the number clue are more or less "the same". Here, it is much better to get two cards than to get one card, so Alice should clue number 3.

[Example diagram: OtherGeneralStrategyAnswer5](../../examples/other-general-strategy-question-5/other-general-strategy-answer-5.yml)

```yaml
stacks:
  - r: 5
  - y: 0
  - g: 2
  - b: 4
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
      - type: y5
        border: false
      - type: b4
        border: false
      - type: g3
        clue: 3
        middleNote: g3
        below: Play
      - type: y3
        clue: 3
        middleNote: y3, p3
      - type: r1
        border: false
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
