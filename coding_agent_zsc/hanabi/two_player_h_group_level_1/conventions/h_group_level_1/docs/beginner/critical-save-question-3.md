> Copied from [H-group: The Critical Save (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/critical-save-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Critical Save (Question 3)

## Question

- Alice clues yellow to Bob, touching a card on slot 4.
- Is this a _Play Clue_ or a _Save Clue_?
- What _card note_ does Bob write? (Be specific.)

[Example diagram: CriticalSaveQuestion3](../../examples/critical-save-question-3/critical-save-question-3.yml)

```yaml
stacks:
  - r: 2
  - y: 0
  - g: 1
  - b: 4
  - p: 3
discarded:
  - y3
  - y4
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
      - type: y
        clue: y
      - type: 3
  - cards:
      - type: g3
        border: false
      - type: g4
        border: false
      - type: y3
        border: false
      - type: g1
        border: false
      - type: p1
        border: false
```

## Solution

- Bob knows that this could be a _Play Clue_ on the yellow 1.
- However, since it touched his chop, it could also be a _Save Clue_:
  - It cannot be a yellow 2, because a yellow 2 would have to be saved with a _2 Save_.
  - It cannot be a yellow 3, because there are only two copies of yellow 3 in the deck, and Bob sees that one copy is in the trash and one copy is in Cathy's hand.
  - It could be a yellow 4, because one copy of yellow 4 is in the trash.
  - It cannot be a yellow 5, because a yellow 5 would have to be saved with a _5 Save_.
- Bob does not know whether or not this is a _Play Clue_ or a _Save Clue_, but he has to treat it as a _Save Clue_ for the time being until he gets more information.
- Bob writes a _card note_ that includes the following identities:
  - yellow 1 (as a _Play Clue_)
  - yellow 4 (as a _Critical Save_)

[Example diagram: CriticalSaveAnswer3](../../examples/critical-save-question-3/critical-save-answer-3.yml)

```yaml
stacks:
  - r: 2
  - y: 0
  - g: 1
  - b: 4
  - p: 3
discarded:
  - y3
  - y4
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
      - type: y12345
        crossedOut: 3
        clue: y
        middleNote: y1, y4
        below: sp
      - type: 3
  - cards:
      - type: g3
        border: false
      - type: g4
        border: false
      - type: y3
        border: false
      - type: g1
        border: false
      - type: p1
        border: false
```
