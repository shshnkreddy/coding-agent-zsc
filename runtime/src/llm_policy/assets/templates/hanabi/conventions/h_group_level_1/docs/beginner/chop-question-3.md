> Copied from [H-group: The Chop (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/chop-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Chop (Question 3)

## Question

- What slot is Alice's chop?

[Example diagram: ChopQuestion3](../../examples/chop-question-3/chop-question-3.yml)

```yaml
players:
  - cards:
      - type: y1
        border: false
      - type: r3
      - type: b2
        border: false
      - type: r4
      - type: r5
```

## Solution

- Slot 3, the blue 2 (because slots 4 and 5 have clues on them).

[Example diagram: ChopAnswer3](../../examples/chop-question-3/chop-answer-3.yml)

```yaml
players:
  - cards:
      - type: y1
        border: false
      - type: r3
      - type: b2
        border: false
        below: Chop
      - type: r4
      - type: r5
```
