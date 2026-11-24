> Copied from [H-group: Delayed Play Clues (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/delayed-play-clues-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Delayed Play Clues (Question 3)

## Question

- Alice clues number 4 to Bob, which touches one card on slot 1. (This is a _Play Clue_.)
- Bob writes a _card note_ with the identity of the card.

1. What _card note_ does Bob write? (Be specific.)
1. Is this a _Delayed Play Clue_?
1. What action should Bob perform?

[Example diagram: DelayedPlayCluesQuestion3](../../examples/delayed-play-clues-question-3/delayed-play-clues-question-3.yml)

```yaml
stacks:
  - r: 3
  - y: 1
  - g: 0
  - b: 2
  - p: 3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: 4
        clue: 4
        below: Play
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r1
        border: false
      - type: y2
      - type: y3
      - type: g5
  - cards:
      - type: p5
        border: false
      - type: g2
        border: false
      - type: p1
        border: false
      - type: b3
```

## Solution

1. Bob writes a _card note_ of "r4, y4, b4, p4".
   - First, it could be a direct _Play Clue_ on a red 4 or a purple 4.
   - Second, it could be a _Delayed Play Clue_ on a yellow 4 through Cathy's yellow 2 and yellow 3.
   - Third, it could be a _Delayed Play Clue_ on a blue 4 through Donald's blue 3.
1. Bob does not know whether this is a _Play Clue_ or a _Delayed Play Clue_. However, since Bob has to treat all _Play Clues_ as potential _Delayed Play Clues_, Bob cannot play the 4 on this turn.
1. Since Bob cannot play the 4 yet, he should give an unrelated clue or discard.

[Example diagram: DelayedPlayCluesAnswer3](../../examples/delayed-play-clues-question-3/delayed-play-clues-answer-3.yml)

```yaml
stacks:
  - r: 3
  - y: 1
  - g: 0
  - b: 2
  - p: 3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: 4
        clue: 4
        below: Play
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r1
        border: false
      - type: y2
      - type: y3
      - type: g5
  - cards:
      - type: p5
        border: false
      - type: g2
        border: false
      - type: p1
        border: false
      - type: b3
```
