> Copied from [H-group: Good Touch Principle (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/good-touch-principle-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Good Touch Principle (Question 3)

## Question

- Alice has a clued 2 in her hand of an unknown color. (It was _2 Saved_ earlier on in the game.)
- What action should Alice perform?
- If Alice should give a clue, what card note will be written on the focused card?

[Example diagram: GoodTouchPrincipleQuestion3](../../examples/good-touch-principle-question-3/good-touch-principle-question-3.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 1
  - b: 0
  - p: 0
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: 2
  - cards:
      - type: y2
        border: false
      - type: p3
        border: false
      - type: b3
        border: false
      - type: p5
        border: false
      - type: p3
        border: false
  - cards:
      - type: b4
        border: false
      - type: b5
        border: false
      - type: b4
        border: false
      - type: g1
        border: false
      - type: g3
        border: false
```

## Solution

- Alice could clue yellow to Bob or clue number 2 to Bob. Either clue would be a _Play Clue_ on the yellow 2.
- However, it is possible that Alice's clued 2 is a yellow 2. If it is, then Alice would be violating _Good Touch Principle_ and would be wasting a clue.
- Thus, it is not Alice's responsibility to clue the yellow 2. Someone else on the team will take care of that. Alice should discard, since there are no other clues to give.

[Example diagram: GoodTouchPrincipleAnswer3](../../examples/good-touch-principle-question-3/good-touch-principle-answer-3.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 1
  - b: 0
  - p: 0
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
        below: Discard
      - type: 2
        middleNote: y2?
  - cards:
      - type: y2
        border: false
      - type: p3
        border: false
      - type: b3
        border: false
      - type: p5
        border: false
      - type: p3
        border: false
  - cards:
      - type: b4
        border: false
      - type: b5
        border: false
      - type: b4
        border: false
      - type: g1
        border: false
      - type: g3
        border: false
```
