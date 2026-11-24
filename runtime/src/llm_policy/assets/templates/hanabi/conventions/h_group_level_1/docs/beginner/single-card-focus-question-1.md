> Copied from [H-group: Single Card Focus (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/single-card-focus-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Single Card Focus (Question 1)

## Question

- Bob has a completely unclued hand.
- Alice clues purple to Bob, touching four cards.
- Which slot is focused?

[Example diagram: SingleCardFocusQuestion1](../../examples/single-card-focus-question-1/single-card-focus-question-1.yml)

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
      - type: p
        clue: p
      - type: p
        clue: p
      - type: p
        clue: p
      - type: p
        clue: p
      - type: x
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

- Slot 1, because it is the leftmost card.
  - (All of the cards are "new" and none of them are on chop.)

[Example diagram: SingleCardFocusAnswer1](../../examples/single-card-focus-question-1/single-card-focus-answer-1.yml)

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
      - type: p
        clue: p
        below: Focus
      - type: p
        clue: p
      - type: p
        clue: p
      - type: p
        clue: p
      - type: x
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
