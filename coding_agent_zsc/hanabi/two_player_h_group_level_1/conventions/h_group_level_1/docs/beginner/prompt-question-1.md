> Copied from [H-group: The Prompt (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/prompt-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Prompt (Question 1)

## Question

- Bob has two unknown 2's previously clued in his hand.
- Cathy has an unknown 5 and an unknown 4 previously clued in her hand.
- Alice clues red to Cathy.
- What is Bob's interpretation of Alice's clue?

[Example diagram: PromptQuestion1](../../examples/prompt-question-1/prompt-question-1.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 3
  - b: 0
  - p: 5
discarded:
  - g1
  - g4
  - r3
  - b3
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
      - type: 2
      - type: 2
      - type: x
  - cards:
      - type: b4
        border: false
      - type: y3
        border: false
      - type: r3
        clue: r
      - type: g4
      - type: r5
        clue: r
        retouched: true
```

## Solution

- Since the red 3 is in this trash, this is a _Critical Save_ on the red 3 (i.e. a _Save Clue_).
- Bob should therefore not react in any way.
- If the red 3 were not in the trash, then it would be a _Prompt_ on Bob's leftmost 2.

[Example diagram: PromptAnswer1](../../examples/prompt-question-1/prompt-answer-1.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 3
  - b: 0
  - p: 5
discarded:
  - g1
  - g4
  - r3
  - b3
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
      - type: 2
      - type: 2
      - type: x
  - cards:
      - type: b4
        border: false
      - type: y3
        border: false
      - type: r3
        clue: r
        below: Save
      - type: g4
      - type: r5
        clue: r
        retouched: true
```
