> Copied from [H-group: The Critical Save (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/critical-save-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Critical Save (Question 2)

## Question

- The team has 2 clue tokens left.
- Cathy has two critical cards in a row.
- What action should Alice perform?

[Example diagram: CriticalSaveQuestion2](../../examples/critical-save-question-2/critical-save-question-2.yml)

```yaml
stacks:
  - r: 4
  - y: 0
  - g: 1
  - b: 2
  - p: 0
discarded:
  - p3
  - p4
players:
  - clueGiver: true
    cards:
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
  - cards:
      - type: r1
        border: false
      - type: r1
        border: false
      - type: g4
        border: false
      - type: p3
        border: false
      - type: p5
        border: false
```

## Solution

- Alice might be tempted to clue purple to Cathy, which would "get" both purple cards in one fell swoop.
- However, it would be illegal for Alice to do that because 5's must be saved with a _5 Save_.
- Thus, Alice should clue number 5 to Cathy as a _5 Save_.
- Next, Bob would clue purple to Cathy as a _Critical Save_ on the purple 3.
- Cathy would write a _card note_ on her slot 4 card that contains the following identities:
  - purple 1 (as a _Play Clue_)
  - purple 3 (as a _Critical Save_)
  - purple 4 (as a _Critical Save_)

[Example diagram: CriticalSaveAnswer2](../../examples/critical-save-question-2/critical-save-answer-2.yml)

```yaml
stacks:
  - r: 4
  - y: 0
  - g: 1
  - b: 2
  - p: 0
discarded:
  - p3
  - p4
players:
  - clueGiver: true
    cards:
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
  - cards:
      - type: r1
        border: false
      - type: r1
        border: false
      - type: g4
        border: false
      - type: p3
        border: false
      - type: p5
        clue: 5
        middleNote: 5
        below: Save
  - text: Then Bob clues
  - clueGiver: true
    offset: -2
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r1
        border: false
      - type: r1
        border: false
      - type: g4
        border: false
      - type: p3
        clue: p
        middleNote: p134
        below: sp
      - type: p5
        clue: p
        retouched: true
        middleNote: p5
```
