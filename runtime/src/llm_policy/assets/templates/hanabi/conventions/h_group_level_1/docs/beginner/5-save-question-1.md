> Copied from [H-group: The 5 Save (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/5-save-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The 5 Save (Question 1)

## Question

- What action should Alice perform?

[Example diagram: FiveSaveQuestion1](../../examples/5-save-question-1/5-save-question-1.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 0
  - b: 1
  - p: 1
discarded:
  - g3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p4
        border: false
      - type: g4
        border: false
      - type: y5
        border: false
      - type: y1
        border: false
  - cards:
      - type: y4
        border: false
      - type: g4
        border: false
      - type: g3
      - type: g2
  - cards:
      - type: p5
        border: false
      - type: b5
        border: false
      - type: p3
        border: false
      - type: p3
        border: false
```

## Solution

- Bob has a one 5 in his hand. However, it is not on chop.
- Donald has two 5's in his hand. However, neither of them are on chop.
- Alice knows that you are **only allowed** to give _Save Clues_ to cards that are on chop.
- Thus, Alice has nothing to do and should discard.

[Example diagram: FiveSaveAnswer1](../../examples/5-save-question-1/5-save-answer-1.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 0
  - b: 1
  - p: 1
discarded:
  - g3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
        below: Discard
  - cards:
      - type: p4
        border: false
      - type: g4
        border: false
      - type: y5
        border: false
      - type: y1
        border: false
        below: Chop
  - cards:
      - type: y4
        border: false
      - type: g4
        border: false
        below: Chop
      - type: g3
      - type: g2
  - cards:
      - type: p5
        border: false
      - type: b5
        border: false
      - type: p3
        border: false
      - type: p3
        border: false
        below: Chop
```
