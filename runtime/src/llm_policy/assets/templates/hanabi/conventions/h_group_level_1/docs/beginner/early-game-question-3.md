> Copied from [H-group: The Early Game (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/early-game-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Early Game (Question 3)

## Question

- It is currently the _Early Game_. There is 1 clue token left.
- What action should Alice perform?

[Example diagram: EarlyGameQuestion3](../../examples/early-game-question-3/early-game-question-3.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 1
  - b: 0
  - p: 1
bigText:
  text: Early Game
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p4
        border: false
      - type: r2
      - type: r3
      - type: p1
        border: false
  - cards:
      - type: g1
        border: false
      - type: p2
        border: false
      - type: r2
        border: false
      - type: y4
        border: false
  - cards:
      - type: g4
        border: false
      - type: r1
        border: false
      - type: g5
        border: false
      - type: b4
        border: false
```

## Solution

- The **only** legal action that Alice can perform is to clue purple to Cathy as a _Play Clue_ on the purple 2.
  - Alice cannot clue number 2 to Cathy, since that would touch the red 2 and would violate _Good Touch Principle_.

[Example diagram: EarlyGameAnswer3](../../examples/early-game-question-3/early-game-answer-3.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 1
  - b: 0
  - p: 1
bigText:
  text: Early Game
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p4
        border: false
      - type: r2
      - type: r3
      - type: p1
        border: false
  - cards:
      - type: g1
        border: false
      - type: p2
        clue: p
        middleNote: p2
        below: Play
      - type: r2
        border: false
      - type: y4
        border: false
  - cards:
      - type: g4
        border: false
      - type: r1
        border: false
      - type: g5
        border: false
      - type: b4
        border: false
```
