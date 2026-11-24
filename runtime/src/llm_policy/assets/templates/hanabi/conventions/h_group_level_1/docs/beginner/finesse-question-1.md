> Copied from [H-group: The Finesse (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/finesse-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Finesse (Question 1)

## Question

- Enumerate the 5 legal clues that Alice can give right now, along with their clue names.
- What is the best clue?

[Example diagram: FinesseQuestion1](../../examples/finesse-question-1/finesse-question-1.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 2
  - b: 0
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
      - type: b1
        border: false
      - type: y5
      - type: r4
        border: false
      - type: g5
      - type: r3
        border: false
  - cards:
      - type: p4
        border: false
      - type: y2
        border: false
      - type: b2
        border: false
      - type: g4
      - type: p3
```

## Solution

1. Alice can clue blue to Bob, touching the blue 1 as a _Play Clue_.
1. Alice can clue number 1 to Bob, touching the blue 1 as a _Play Clue_.
1. Alice can clue yellow to Cathy, touching the yellow 2 as a _Play Clue_.
1. Alice can clue blue to Cathy, touching the blue 2 as a _Finesse_ on the blue 1.
1. Alice can clue number 2 to Cathy, touching the yellow 2 and the blue 2 as a _2 Save_.

The best clue is the _Finesse_ because it gets two cards played right now. (The team can give a _Play Clue_ to the yellow 2 later on.)

[Example diagram: FinesseAnswer1](../../examples/finesse-question-1/finesse-answer-1.yml)

```yaml
stacks:
  - r: 0
  - y: 1
  - g: 2
  - b: 0
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
      - type: b1
        border: false
        below: Finesse
      - type: y5
      - type: r4
        border: false
      - type: g5
      - type: r3
        border: false
  - cards:
      - type: p4
        border: false
      - type: y2
        border: false
      - type: b2
        clue: b
        middleNote: b2
        below: Play
      - type: g4
      - type: p3
```
