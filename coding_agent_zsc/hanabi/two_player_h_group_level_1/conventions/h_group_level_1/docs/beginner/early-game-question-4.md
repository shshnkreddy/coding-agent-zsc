> Copied from [H-group: The Early Game (Question 4)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/early-game-question-4.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Early Game (Question 4)

## Question

- It is currently the _Early Game_. There is 1 clue token left.
- What action should Alice perform?

[Example diagram: EarlyGameQuestion4](../../examples/early-game-question-4/early-game-question-4.yml)

```yaml
stacks:
  - r: 2
  - y: 1
  - g: 3
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
  - cards:
      - type: r4
        border: false
      - type: r1
        border: false
      - type: p3
        border: false
      - type: g1
        border: false
  - cards:
      - type: p2
        border: false
      - type: b4
        border: false
      - type: b4
        border: false
      - type: r2
        border: false
  - cards:
      - type: p5
        border: false
      - type: g3
        border: false
      - type: b1
        border: false
      - type: r1
        border: false
```

## Solution

- With beginner conventions, there are no legal actions that can be performed. Thus, Alice must discard.

[Example diagram: EarlyGameAnswer4](../../examples/early-game-question-4/early-game-answer-4.yml)

```yaml
stacks:
  - r: 2
  - y: 1
  - g: 3
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
        below: Discard
  - cards:
      - type: r4
        border: false
      - type: r1
        border: false
      - type: p3
        border: false
      - type: g1
        border: false
  - cards:
      - type: p2
        border: false
      - type: b4
        border: false
      - type: b4
        border: false
      - type: r2
        border: false
  - cards:
      - type: p5
        border: false
      - type: g3
        border: false
      - type: b1
        border: false
      - type: r1
        border: false
```
