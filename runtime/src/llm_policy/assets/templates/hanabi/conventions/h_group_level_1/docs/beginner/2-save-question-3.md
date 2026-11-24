> Copied from [H-group: The 2 Save (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/2-save-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The 2 Save (Question 3)

## Question

- What action should Alice perform?

[Example diagram: TwoSaveQuestion3](../../examples/2-save-question-3/2-save-question-3.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 0
  - b: 2
  - p: 1
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: 5
      - type: 5
  - cards:
      - type: r4
        border: false
      - type: p3
        border: false
      - type: y2
        border: false
      - type: r2
        border: false
      - type: g2
        border: false
  - cards:
      - type: y5
        border: false
      - type: g2
        border: false
      - type: g1
        border: false
      - type: g1
        border: false
      - type: y2
        border: false
```

## Solution

- Alice is not allowed to perform a _2 Save_ on Bob's green 2 because the other copy of the green 2 is present in Cathy's hand and is not on chop.
- Alice is not allowed to perform a _2 Save_ on Cathy's yellow 2 because the other copy of the yellow 2 is present in Bob's hand and is not on chop.
- Thus, Alice should discard.

[Example diagram: TwoSaveAnswer3](../../examples/2-save-question-3/2-save-answer-3.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 0
  - b: 2
  - p: 1
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
        below: Discard
      - type: 5
      - type: 5
  - cards:
      - type: r4
        border: false
      - type: p3
        border: false
      - type: y2
        border: false
        below: Dupe
      - type: r2
        border: false
      - type: g2
        border: false
        below: Dupe
  - cards:
      - type: y5
        border: false
      - type: g2
        border: false
        below: Dupe
      - type: g1
        border: false
      - type: g1
        border: false
      - type: y2
        border: false
        below: Dupe
```
