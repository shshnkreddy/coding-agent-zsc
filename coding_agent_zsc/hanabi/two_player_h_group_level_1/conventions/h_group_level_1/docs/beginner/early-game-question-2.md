> Copied from [H-group: The Early Game (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/early-game-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Early Game (Question 2)

## Question

- It is currently the _Early Game_. There is 1 clue token left.
- Previously, Bob was clued about some 1's. He will go on to play the purple 1 and the green 1 in the future.
- What action should Alice perform?

[Example diagram: EarlyGameQuestion2](../../examples/early-game-question-2/early-game-question-2.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 0
  - b: 1
  - p: 0
bigText:
  text: Early Game
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: 5
  - cards:
      - type: r1
        border: false
      - type: g2
        border: false
      - type: p1
      - type: g1
      - type: y2
        border: false
  - cards:
      - type: b3
        border: false
      - type: p4
        border: false
      - type: p4
        border: false
      - type: p5
      - type: g5
```

## Solution

- The **only** legal action that Alice can perform is to clue green to Bob as a _Delayed Play Clue_ on the green 2.
  - Other than this green clue, there are no other possible legal clues to give.
  - Alice cannot clue number 2 to Bob, since that would touch the yellow 2 and would violate _Good Touch Principle_.
  - Alice is not allowed to discard because it is the _Early Game_ and not all of the _Play Clues_ have been extinguished yet.

[Example diagram: EarlyGameAnswer2](../../examples/early-game-question-2/early-game-answer-2.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 0
  - b: 1
  - p: 0
bigText:
  text: Early Game
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: 5
  - cards:
      - type: r1
        border: false
      - type: g2
        clue: g
        middleNote: g2
        below: Play
      - type: p1
      - type: g1
        clue: g
        retouched: true
        middleNote: g1
      - type: y2
        border: false
  - cards:
      - type: b3
        border: false
      - type: p4
        border: false
      - type: p4
        border: false
      - type: p5
      - type: g5
```
