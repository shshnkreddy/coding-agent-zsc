> Copied from [H-group: The Prompt (Question 3)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/prompt-question-3.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Prompt (Question 3)

## Question

- Cathy has two unknown purple cards in her hand.
- Alice wants to give a _Play Clue_ to the purple 4.
- What clue should Alice give?

[Example diagram: PromptQuestion3](../../examples/prompt-question-3/prompt-question-3.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 1
  - b: 2
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
      - type: p4
        border: false
      - type: b2
        border: false
      - type: g1
        border: false
      - type: r5
  - cards:
      - type: g4
        border: false
      - type: b1
        border: false
      - type: g4
        border: false
      - type: p3
        middleNote: p
      - type: p2
        middleNote: p
```

## Solution

- It is possible for Alice to give a _Play Clue_ to the purple 4 either by cluing purple or by cluing number 4. Both clues would focus the purple 4.
- However, this would signal a _Prompt_ to Cathy, and she would play her purple cards from left to right. That means that the purple 3 would misplay from slot 4.
- Thus, Alice is not allowed to give a _Play Clue_ to the purple 4.
- Instead, Alice can clue number 3 to Cathy. This would focus the purple 3 as a _Self-Prompt_.
- After that, Alice can give a _Play Clue_ to the purple 4 without any problems.
- Note that normally, players are not supposed to give _Tempo Clues_, because they do not meet _Minimum Clue Value Principle_. But here, Alice has a very good reason to give a _Tempo Clue_ - Cathy's cards are "out of order". So, it is okay to break the rule.

[Example diagram: PromptAnswer3](../../examples/prompt-question-3/prompt-answer-3.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 1
  - b: 2
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
      - type: p4
        border: false
      - type: b2
        border: false
      - type: g1
        border: false
      - type: r5
  - cards:
      - type: g4
        border: false
      - type: b1
        border: false
      - type: g4
        border: false
      - type: p3
        clue: 3
        middleNote: p3
        below: Play
      - type: p2
        middleNote: p2
        below: Prompt
```
