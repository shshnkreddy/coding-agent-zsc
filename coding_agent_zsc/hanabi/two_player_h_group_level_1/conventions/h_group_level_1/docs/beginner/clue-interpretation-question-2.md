> Copied from [H-group: Clue Interpretation (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/clue-interpretation-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Clue Interpretation (Question 2)

## Question

- Bob has a completely unclued hand.
- Alice clues number 2 to Bob, touching slot 5.

1. What slot is focused?
1. Is this a _Play Clue_ or a _Save Clue_?
1. What _card note_ does Bob write on his slot 5 card?

[Example diagram: ClueInterpretationQuestion2](../../examples/clue-interpretation-question-2/clue-interpretation-question-2.yml)

```yaml
stacks:
  - r: 2
  - y: 1
  - g: 1
  - b: 2
  - p: 3
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
      - type: x
      - type: x
      - type: x
      - type: 2
        clue: 2
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

1. Slot 5 is focused.
1. This is a _Play Clue_.
   - (Normally, this would look like a _2 Save_ clue, but all of the 2's are currently playable. So it can't be a _2 Save_ and is just a simple _Play Clue_ instead.)
1. Bob writes a _card note_ of "y2, g2".

[Example diagram: ClueInterpretationAnswer2](../../examples/clue-interpretation-question-2/clue-interpretation-answer-2.yml)

```yaml
stacks:
  - r: 2
  - y: 1
  - g: 1
  - b: 2
  - p: 3
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
      - type: x
      - type: x
      - type: x
      - type: 2
        clue: 2
        middleNote: y2, g2
        below: Play
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
