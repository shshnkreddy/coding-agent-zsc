> Copied from [H-group: Clue Interpretation (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/clue-interpretation-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Clue Interpretation (Question 3)

## Question

- Bob has a clued 2 and a clued 5 in his hand.
- Alice clues red to Bob, touching slot 2, slot 3, slot 4, and slot 5.

1. What slot is focused?
1. Is this a _Play Clue_ or a _Save Clue_?
1. What _card note_ does Bob write on the focused card?

[Example diagram: ClueInterpretationQuestion3](../../examples/clue-interpretation-question-3/clue-interpretation-question-3.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 3
  - b: 5
  - p: 4
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
      - type: r
        clue: r
      - type: r
        clue: r
      - type: r2
        clue: r
        retouched: true
      - type: r5
        clue: r
        retouched: true
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

1. Slot 3 is focused (i.e. the "new" card that was on chop).
1. Red 1 is played on the stacks and Bob already has a red 2 and a red 5. Thus, the focused red card can be either a red 3 or a red 4.
   - If it is a red 3, then this would be a _Chop-Focus Delayed Play Clue_ through his own red 2. That is legal.
   - If it is a red 4, then it wouldn't make any sense, because red 4 is neither playable nor critical. (All clues must either be _Play Clues_ or _Save Clues_.)
   - Thus, it must be a _Chop-Focus Delayed Play Clue_ on the red 3.
1. Bob writes a _card note_ of "r3" on his slot 3 card.

[Example diagram: ClueInterpretationAnswer3](../../examples/clue-interpretation-question-3/clue-interpretation-answer-3.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 3
  - b: 5
  - p: 4
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
      - type: r
        clue: r
      - type: r
        clue: r
        middleNote: r3
        below: Play
      - type: r2
        clue: r
        retouched: true
      - type: r5
        clue: r
        retouched: true
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```
