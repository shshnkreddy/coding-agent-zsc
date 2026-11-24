> Copied from [H-group: Save Principle (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/save-principle-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Save Principle (Question 2)

## Question

- What action should Alice perform?
- If Alice should give a clue, what _card note_ will be written on the focused card? (Be specific.)

[Example diagram: SavePrincipleQuestion2](../../examples/save-principle-question-2/save-principle-question-2.yml)

```yaml
stacks:
  - r: 2
  - y: 0
  - g: 1
  - b: 1
  - p: 0
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: g4
        border: false
      - type: b5
        border: false
      - type: g1
        border: false
      - type: r2
        border: false
      - type: y2
        border: false
  - cards:
      - type: r5
        border: false
      - type: p3
        border: false
      - type: g3
        border: false
      - type: p5
      - type: p4
```

## Solution

- Alice needs to give a _Save Clue_ to Bob's yellow 2 before it is discarded (because everyone agrees that all unique 2's **must** be saved).
- Alice is not allowed to clue yellow, because 2's must be saved with a _2 Save_.
  - If Alice did clue yellow, then it would look like a _Play Clue_ on a yellow card and Bob would try to play it as the yellow 1.
- If Alice clues number 2, then it would be a _2 Save_ on the yellow 2, but it would also duplicate the red 2 (which is already played on the stacks).
- Thus, Alice should clue number 2 to Bob as a _2 Save_ (because it is more important to save cards than it is to satisfy _Good Touch Principle_).
- That clue would be focused on slot 5. Bob will write a _card note_ with the following identities:
  - green 2, blue 2 (as a _Play Clue_)
  - yellow 2, purple 2 (as a _2 Save_)

[Example diagram: SavePrincipleAnswer2](../../examples/save-principle-question-2/save-principle-answer-2.yml)

```yaml
stacks:
  - r: 2
  - y: 0
  - g: 1
  - b: 1
  - p: 0
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: g4
        border: false
      - type: b5
        border: false
      - type: g1
        border: false
      - type: r2
        clue: 2
        middleNote: 2, !r
      - type: y2
        clue: 2
        middleNote: 2, !r
        below: sp
  - cards:
      - type: r5
        border: false
      - type: p3
        border: false
      - type: g3
        border: false
      - type: p5
      - type: p4
```
