> Copied from [H-group: Clue Interpretation (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/clue-interpretation-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Clue Interpretation (Question 1)

## Question

- Cathy has a completely unclued hand.
- Alice clues green to Cathy, touching slot 3 and slot 4.

1. What slot is focused?
1. Is this a _Play Clue_ or a _Save Clue_?
1. What _card note_ does Cathy write on the focused card?

[Example diagram: ClueInterpretationQuestion1](../../examples/clue-interpretation-question-1/clue-interpretation-question-1.yml)

```yaml
stacks:
  - r: 1
  - y: 3
  - g: 2
  - b: 3
  - p: 1
discarded:
  - g1
  - b2
  - g3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: g
        clue: g
      - type: g
        clue: g
  - cards:
      - type: b3
        border: false
      - type: g2
        border: false
      - type: p3
      - type: r3
```

## Solution

1. Slot 4 is focused (i.e. the chop card).
1. Since green 2 is played on the stacks, the card on slot 4 must be either a green 3, a green 4, or a green 5.
   - If it is a green 3, then this would be a _Chop-Focus Play Clue_. That is legal.
   - If it is a green 4, then it wouldn't make any sense, because green 4 is neither playable nor critical. (All clues must either be _Play Clues_ or _Save Clues_.)
   - If it is a green 5, then it wouldn't make any sense, because green 5 is not playable. Green 5 is critical, but it must be saved with a number 5 clue as a _5 Save_.
   - Thus, it must be a _Chop-Focus Play Clue_ on the green 3.
1. Cathy writes a _card note_ of "g3" on her slot 4 card.

[Example diagram: ClueInterpretationAnswer1](../../examples/clue-interpretation-question-1/clue-interpretation-answer-1.yml)

```yaml
stacks:
  - r: 1
  - y: 3
  - g: 2
  - b: 3
  - p: 1
discarded:
  - g1
  - b2
  - g3
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: x
      - type: x
      - type: g
        clue: g
      - type: g
        clue: g
        middleNote: g3
        below: Play
  - cards:
      - type: b3
        border: false
      - type: g2
        border: false
      - type: p3
      - type: r3
```
