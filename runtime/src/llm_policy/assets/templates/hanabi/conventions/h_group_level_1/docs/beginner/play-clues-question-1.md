> Copied from [H-group: Play Clues (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/play-clues-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Play Clues (Question 1)

## Question

- Alice clues blue to Bob, which touches one card on slot 3.
- From Bob's perspective, this must be a _Play Clue_. Bob marks down the identity of the card.
- What _card note_ (i.e. card identity) does Bob write?

[Example diagram: PlayCluesQuestion1](../../examples/play-clues-question-1/play-clues-question-1.yml)

```yaml
stacks:
  - r: 5
  - y: 3
  - g: 2
  - b: 1
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
      - type: b
        clue: b
        below: Play
      - type: x
      - type: x
```

## Solution

- Bob writes "b2".
  - (Since the blue 1 is already played, the next blue card to play is the blue 2.)

[Example diagram: PlayCluesAnswer1](../../examples/play-clues-question-1/play-clues-answer-1.yml)

```yaml
stacks:
  - r: 5
  - y: 3
  - g: 2
  - b: 1
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
      - type: b
        clue: b
        middleNote: b2
        below: Play
      - type: x
      - type: x
```
