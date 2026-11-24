> Copied from [H-group: The Early Game (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/early-game-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Early Game (Question 1)

## Question

- On the first turn of the game, Alice clues number 1 to Cathy, touching three cards.
- Bob discards a red 1.
- Is it now the _Early Game_ or the _Mid-Game_?

[Example diagram: EarlyGameQuestion1](../../examples/early-game-question-1/early-game-question-1.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 0
  - b: 0
  - p: 0
discarded:
  - r1
players:
  - cards:
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
      - type: x
  - clueGiver: true
    cards:
      - type: b1
      - type: g1
      - type: r1
      - type: x
      - type: x
```

## Solution

- It is the _Mid-Game_ because Bob discarded his chop card.

[Example diagram: EarlyGameAnswer1](../../examples/early-game-question-1/early-game-answer-1.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 0
  - b: 0
  - p: 0
bigText:
  text: Mid-Game
discarded:
  - r1
players:
  - cards:
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
      - type: x
  - clueGiver: true
    cards:
      - type: b1
      - type: g1
      - type: r1
      - type: x
      - type: x
```
