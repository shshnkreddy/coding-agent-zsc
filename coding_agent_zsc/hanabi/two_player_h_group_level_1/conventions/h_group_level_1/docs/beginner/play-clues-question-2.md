> Copied from [H-group: Play Clues (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/play-clues-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Play Clues (Question 2)

## Question

- Alice clues number 4 to Bob, which touches one card on slot 4.
- From Bob's perspective, this must be a _Play Clue_. Bob marks down the identity of the card.
- What _card note_ does Bob write? (Be specific.)

[Example diagram: PlayCluesQuestion2](../../examples/play-clues-question-2/play-clues-question-2.yml)

```yaml
stacks:
  - r: 3
  - y: 4
  - g: 3
  - b: 3
  - p: 2
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
      - type: 4
        clue: 4
        below: Play
      - type: x
```

## Solution

- Bob writes "r4, g4, b4".
  - (These are the suits that have playable 4's, so the card can be any of these possibilities.)

[Example diagram: PlayCluesAnswer2](../../examples/play-clues-question-2/play-clues-answer-2.yml)

```yaml
stacks:
  - r: 3
  - y: 4
  - g: 3
  - b: 3
  - p: 2
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
      - type: 4
        clue: 4
        middleNote: rgb4
        below: Play
      - type: x
```
