> Copied from [H-group: The Chop (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/chop-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Chop (Question 2)

## Question

- What slot is Alice's chop?

[Example diagram: ChopQuestion2](../../examples/chop-question-2/chop-question-2.yml)

```yaml
players:
  - cards:
      - type: y2
        border: false
      - type: b3
        border: false
      - type: g2
      - type: p4
      - type: b1
```

## Solution

- Slot 2, the blue 3 (because slot 3, slot 4, and slot 5 have clues on them).

[Example diagram: ChopAnswer2](../../examples/chop-question-2/chop-answer-2.yml)

```yaml
players:
  - cards:
      - type: y2
        border: false
      - type: b3
        border: false
        below: Chop
      - type: g2
      - type: p4
      - type: b1
```
