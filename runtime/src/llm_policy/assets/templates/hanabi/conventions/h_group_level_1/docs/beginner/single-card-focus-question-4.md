> Copied from [H-group: Single Card Focus (Question 4)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/single-card-focus-question-4.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Single Card Focus (Question 4)

## Question

- Bob has a 5 and a 2 clued in his hand.
- Alice clues red to Bob, touching three cards.
- Which slot is focused?

[Example diagram: SingleCardFocusQuestion4](../../examples/single-card-focus-question-4/single-card-focus-question-4.yml)

```yaml
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: r
        clue: r
      - type: r
        clue: r
      - type: 5
      - type: r2
        clue: r
        retouched: true
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

- Slot 3, because there are two new cards and slot 3 is the chop.
  - (Slot 5 was previously touched, so the red clue "fills it in".)

[Example diagram: SingleCardFocusAnswer4](../../examples/single-card-focus-question-4/single-card-focus-answer-4.yml)

```yaml
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: r
        clue: r
      - type: r
        clue: r
        below: Focus
      - type: 5
      - type: r2
        clue: r
        retouched: true
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
