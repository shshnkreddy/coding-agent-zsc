> Copied from [H-group: Save Principle (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/save-principle-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Save Principle (Question 1)

## Question

- Alice clues blue to Bob, touching a card on slot 1. This must be a _Play Clue_ on the blue 1.
- Should Bob play the blue 1?

[Example diagram: SavePrincipleQuestion1](../../examples/save-principle-question-1/save-principle-question-1.yml)

```yaml
stacks:
  - r: 1
  - y: 1
  - g: 0
  - b: 0
  - p: 3
discarded:
  - g4
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: b
        clue: b
        middleNote: b1
        below: Play
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r1
        border: false
      - type: p4
        border: false
      - type: p3
        border: false
      - type: r1
        border: false
      - type: g4
        border: false
```

## Solution

- No. If Bob plays the blue 1 and Cathy discards the green 4, then the maximum score would be decreased.
- As long as there is a chance that Cathy will discard, Bob **must** save the green 4 right now before playing any playable cards.

[Example diagram: SavePrincipleAnswer1](../../examples/save-principle-question-1/save-principle-answer-1.yml)

```yaml
stacks:
  - r: 1
  - y: 1
  - g: 0
  - b: 0
  - p: 3
discarded:
  - g4
players:
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - clueGiver: true
    cards:
      - type: b
        middleNote: b1
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r1
        border: false
      - type: p4
        clue: 4
      - type: p3
        border: false
      - type: r1
        border: false
      - type: g4
        clue: 4
        middleNote: g4, p4
        below: sp
```
