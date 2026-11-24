> Copied from [H-group: Single Card Focus (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/single-card-focus-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Single Card Focus (Question 2)

## Question

- Bob has a completely unclued hand.
- Alice clues green to Bob, touching five cards.
- Which slot is focused?

[Example diagram: SingleCardFocusQuestion2](../../examples/single-card-focus-question-2/single-card-focus-question-2.yml)

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
      - type: g
        clue: g
      - type: g
        clue: g
      - type: g
        clue: g
      - type: g
        clue: g
      - type: g
        clue: g
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

- Slot 5, because it is the chop card.
  - (All of the cards are "new".)

[Example diagram: SingleCardFocusAnswer2](../../examples/single-card-focus-question-2/single-card-focus-answer-2.yml)

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
      - type: g
        clue: g
      - type: g
        clue: g
      - type: g
        clue: g
      - type: g
        clue: g
      - type: g
        clue: g
        below: Focus
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
