> Copied from [H-group: Other General Strategy (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/other-general-strategy-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Other General Strategy (Question 1)

## Question

- It is Alice's turn and she has a yellow 2 to play.
- Cathy has a critical blue 5 on chop.
- Should Alice play the yellow 2 or save Cathy's blue 5?

[Example diagram: OtherGeneralStrategyQuestion1](../../examples/other-general-strategy-question-1/other-general-strategy-question-1.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 0
  - b: 1
  - p: 3
discarded:
  - g3
  - g4
  - p4
players:
  - clueGiver: true
    cards:
      - type: y2
        middleNote: y2
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p2
        border: false
      - type: b1
        border: false
      - type: r4
        border: false
      - type: y1
        border: false
      - type: p3
        border: false
  - cards:
      - type: r3
        border: false
      - type: r3
        border: false
      - type: y1
        border: false
      - type: y5
        border: false
      - type: b5
        border: false
```

## Solution

- Alice has a playable card and Bob doesn't have anything to do.
- Thus, it would be bad teamwork for Alice to save Cathy's blue 5.
- Alice should play the yellow 2 and let Bob handle saving Cathy's blue 5.

[Example diagram: OtherGeneralStrategyAnswer1](../../examples/other-general-strategy-question-1/other-general-strategy-answer-1.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 0
  - b: 1
  - p: 3
discarded:
  - g3
  - g4
  - p4
players:
  - clueGiver: true
    cards:
      - type: y2
        middleNote: y2
        below:
          text: Play
          color: cyan
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p2
        border: false
      - type: b1
        border: false
      - type: r4
        border: false
      - type: y1
        border: false
      - type: p3
        border: false
  - cards:
      - type: r3
        border: false
      - type: r3
        border: false
      - type: y1
        border: false
      - type: y5
        border: false
      - type: b5
        border: false
```
