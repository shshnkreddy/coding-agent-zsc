> Copied from [H-group: Single Card Focus (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/single-card-focus-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Single Card Focus (Question 3)

## Question

- Bob has two 2's clued in his hand.
- Alice clues blue to Bob, touching two cards.
- Which slot is focused?

[Example diagram: SingleCardFocusQuestion3](../../examples/single-card-focus-question-3/single-card-focus-question-3.yml)

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
      - type: b2
        clue: b
        retouched: true
      - type: b
        clue: b
      - type: x
      - type: 2
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

- Slot 3, because it is the only new card.
  - (Slot 2 was previously touched, so the blue clue "fills it in".)

[Example diagram: SingleCardFocusAnswer3](../../examples/single-card-focus-question-3/single-card-focus-answer-3.yml)

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
      - type: b2
        clue: b
        retouched: true
      - type: b
        clue: b
        below: Focus
      - type: x
      - type: 2
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
