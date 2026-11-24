> Copied from [H-group: The Chop (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/chop-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Chop (Question 1)

## Question

- What slot is Alice's chop?

[Example diagram: ChopQuestion1](../../examples/chop-question-1/chop-question-1.yml)

```yaml
players:
  - cards:
      - type: b5
        border: false
      - type: y1
        border: false
      - type: y2
        border: false
      - type: y3
        border: false
      - type: g1
        border: false
```

## Solution

- Slot 5, the green 1 (because no cards have clues on them).

[Example diagram: ChopAnswer1](../../examples/chop-question-1/chop-answer-1.yml)

```yaml
players:
  - cards:
      - type: b5
        border: false
      - type: y1
        border: false
      - type: y2
        border: false
      - type: y3
        border: false
      - type: g1
        border: false
        below: Chop
```
