> Copied from [H-group: Delayed Play Clues (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/delayed-play-clues-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Delayed Play Clues (Question 1)

## Question

- Alice clues red to Donald, which touches one card on slot 2. (This is a _Play Clue_.)
- Bob clues red to Cathy, which touches one card on slot 3. (This is a _Play Clue_.)
- Cathy writes a _card note_ with the identity of the card.

1. What _card note_ does Cathy write?
1. Is this a _Delayed Play Clue_?
1. What action should Cathy perform?

[Example diagram: DelayedPlayCluesQuestion1](../../examples/delayed-play-clues-question-1/delayed-play-clues-question-1.yml)

```yaml
stacks:
  - r: 1
  - y: 4
  - g: 4
  - b: 2
  - p: 3
players:
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: r
        clue: r
        below: Play
      - type: x
  - cards:
      - type: x
      - type: r2
      - type: x
      - type: x
```

## Solution

1. Cathy writes a _card note_ of "r3".
1. Yes, this is a _Delayed Play Clue_ through the red 2.
1. Since Cathy cannot play the red 3 yet, she should give an unrelated clue or discard.

[Example diagram: DelayedPlayCluesAnswer1](../../examples/delayed-play-clues-question-1/delayed-play-clues-answer-1.yml)

```yaml
stacks:
  - r: 1
  - y: 4
  - g: 4
  - b: 2
  - p: 3
players:
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: r
        clue: r
        middleNote: r3
        below: Play
      - type: x
  - cards:
      - type: x
      - type: r2
      - type: x
      - type: x
```
