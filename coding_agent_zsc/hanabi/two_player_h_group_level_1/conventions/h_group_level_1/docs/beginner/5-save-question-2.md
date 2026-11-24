> Copied from [H-group: The 5 Save (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/5-save-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The 5 Save (Question 2)

## Question

- List all of the legal actions that Alice can perform.
- What action should Alice perform?

[Example diagram: FiveSaveQuestion2](../../examples/5-save-question-2/5-save-question-2.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 0
  - b: 3
  - p: 2
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r1
        border: false
      - type: p5
        border: false
      - type: g5
        border: false
      - type: y2
        border: false
  - cards:
      - type: r4
        border: false
      - type: y5
        border: false
      - type: b5
      - type: y2
```

## Solution

- Bob has a yellow 2 on chop, which would normally need to be saved. However, Cathy already has the other copy clued in her hand, so Bob's yellow 2 is trash and should be ignored.
- Bob also has two 5's, but neither of them are on chop, so they cannot be saved yet.
- Cathy has two 5's. One of them is already saved (i.e. has a clue on it), but one of them is not already saved and is on chop.
  - (Cathy's chop is slot 2 because the cards on slot 3 and slot 4 have a clue on them.)
- Thus, the actions that Alice can perform are:
  - to save Cathy's 5
  - to discard
- But which is better? In this situation, either Alice or Bob can save the 5:
  - If Alice saves Cathy's 5, then Bob would discard a yellow 2 (which is trash).
  - If Alice discards, then she would discard an unknown card, and Bob would save Cathy's 5.
- Since discarding trash is better than discarding an unknown card, Alice should give the clue to let Bob discard.

[Example diagram: FiveSaveAnswer2](../../examples/5-save-question-2/5-save-answer-2.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 0
  - b: 3
  - p: 2
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r1
        border: false
      - type: p5
        border: false
      - type: g5
        border: false
      - type: y2
        border: false
  - cards:
      - type: r4
        border: false
      - type: y5
        clue: 5
        below: Save
      - type: b5
        clue: 5
        retouched: true
      - type: y2
```
