> Copied from [H-group: Delayed Play Clues (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/delayed-play-clues-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Delayed Play Clues (Question 2)

## Question

- Previously, Cathy was clued about some 1's, and will go on to play both of them.
- Alice clues number 3 to Bob, which touches one card on slot 1. (This is a _Play Clue_.)
- Bob writes a _card note_ with the identity of the card.

1. What _card note_ does Bob write? (Be specific.)
1. Is this a _Delayed Play Clue_?
1. What action should Bob perform?

[Example diagram: DelayedPlayCluesQuestion2](../../examples/delayed-play-clues-question-2/delayed-play-clues-question-2.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 0
  - b: 2
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
      - type: 3
        clue: 3
        below: Play
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: r1
      - type: g1
      - type: x
```

## Solution

1. Bob writes a _card note_ of "b3, p3".
1. No, this is not a _Delayed Play Clue_. Bob's 3 cannot connect to Cathy's red 1 or green 1 because no-one else has any connecting 2's.
1. Bob should probably play the 3 immediately (unless he has a good clue to give).

[Example diagram: DelayedPlayCluesAnswer2](../../examples/delayed-play-clues-question-2/delayed-play-clues-answer-2.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 0
  - b: 2
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
      - type: 3
        clue: 3
        middleNote: bp3
        below: Play
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: r1
      - type: g1
      - type: x
```
