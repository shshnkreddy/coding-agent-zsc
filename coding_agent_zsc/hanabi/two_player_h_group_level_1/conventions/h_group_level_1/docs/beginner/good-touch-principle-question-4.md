> Copied from [H-group: Good Touch Principle (Question 4)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/good-touch-principle-question-4.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Good Touch Principle (Question 4)

## Question

- Green 3 needs be played on the stacks.
- What action should Alice perform?
- If Alice should give a clue, what card note will be written on the focused card?

[Example diagram: GoodTouchPrincipleQuestion4](../../examples/good-touch-principle-question-4/good-touch-principle-question-4.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 2
  - b: 0
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
      - type: p3
        border: false
      - type: r2
        border: false
      - type: g3
        border: false
      - type: p4
        border: false
      - type: g3
        border: false
  - cards:
      - type: r5
        border: false
      - type: b3
        border: false
      - type: y4
        border: false
      - type: y2
        border: false
      - type: g1
        border: false
```

## Solution

- Both a color and number clue would duplicate the green 3 and violate _Good Touch Principle_.
- Thus, Alice should wait patiently for Bob to discard the green 3. Once the first copy of the green 3 is discarded, the team can clue the remaining green 3.
- Since there are no legal clues to give, Alice must discard.

[Example diagram: GoodTouchPrincipleAnswer4](../../examples/good-touch-principle-question-4/good-touch-principle-answer-4.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 2
  - b: 0
  - p: 0
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
        below: Discard
  - cards:
      - type: p3
        border: false
      - type: r2
        border: false
      - type: g3
        border: false
        below: Dupe
      - type: p4
        border: false
      - type: g3
        border: false
        below: Dupe
  - cards:
      - type: r5
        border: false
      - type: b3
        border: false
      - type: y4
        border: false
      - type: y2
        border: false
      - type: g1
        border: false
```
