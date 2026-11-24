> Copied from [H-group: The Critical Save (Question 4)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/critical-save-question-4.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Critical Save (Question 4)

## Question

- Alice clues red to Bob, touching a card on slot 5.
- Is this a _Play Clue_ or a _Save Clue_?
- What _card note_ does Bob write? (Be specific.)

[Example diagram: CriticalSaveQuestion4](../../examples/critical-save-question-4/critical-save-question-4.yml)

```yaml
stacks:
  - r: 0
  - y: 2
  - g: 2
  - b: 2
  - p: 2
discarded:
  - r2
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
      - type: r
        clue: r
  - cards:
      - type: p1
        border: false
      - type: g1
        border: false
      - type: b2
        border: false
      - type: b1
        border: false
      - type: g2
        border: false
```

## Solution

- Bob knows that this could be a _Play Clue_ on the red 1.
- However, since it touched his chop, it could also be a _Save Clue_. Since red 2 is in the trash and a red clue matches that card, it could be red 2.
- Bob does not know whether or not this is a _Play Clue_ or a _Save Clue_, but he has to treat it as a _Save Clue_ for the time being until he gets more information.
- Bob writes a _card note_ that includes the following identities:
  - red 1 (as a _Play Clue_)
  - red 2 (as a _Critical Save_)
- Note that most of the time, players save 2's with a _2 Save_. However, if the first copy of a 2 gets discarded, the second copy of that 2 can be _Critical Saved_ in the exact same way that a 3 or a 4 would be.

[Example diagram: CriticalSaveAnswer4](../../examples/critical-save-question-4/critical-save-answer-4.yml)

```yaml
stacks:
  - r: 0
  - y: 2
  - g: 2
  - b: 2
  - p: 2
discarded:
  - r2
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
      - type: r
        clue: r
        middleNote: r1, r2
        below: sp
  - cards:
      - type: p1
        border: false
      - type: g1
        border: false
      - type: b2
        border: false
      - type: b1
        border: false
      - type: g2
        border: false
```
