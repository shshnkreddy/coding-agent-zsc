> Copied from [H-group: Good Touch Principle (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/good-touch-principle-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Good Touch Principle (Question 1)

## Question

- What action should Alice perform?

[Example diagram: GoodTouchPrincipleQuestion1](../../examples/good-touch-principle-question-1/good-touch-principle-question-1.yml)

```yaml
stacks:
  - r: 1
  - y: 4
  - g: 5
  - b: 2
  - p: 1
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p1
        border: false
      - type: r3
        border: false
      - type: g1
        border: false
      - type: b2
        border: false
      - type: b3
        border: false
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

- Alice needs to give a _Play Clue_ to Bob's blue 3 before it is discarded.
- If Alice clues blue, then it would focus the blue 3, but the blue 2 would be duplicated (because it is already played on the stacks).
- If Alice clues number 3, then it would focus the blue 3, but it would also touch the red 3. The red 3 is useful to the team in the future, so this is a good thing.
- Thus, Alice should clue number 3 to Bob.

[Example diagram: GoodTouchPrincipleAnswer1](../../examples/good-touch-principle-question-1/good-touch-principle-answer-1.yml)

```yaml
stacks:
  - r: 1
  - y: 4
  - g: 5
  - b: 2
  - p: 1
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p1
        border: false
      - type: r3
        clue: 3
      - type: g1
        border: false
      - type: b2
        border: false
      - type: b3
        clue: 3
        middleNote: b3
        below: Play
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
