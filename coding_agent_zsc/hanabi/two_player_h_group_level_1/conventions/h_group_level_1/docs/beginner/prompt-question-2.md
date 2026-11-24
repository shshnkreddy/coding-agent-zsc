> Copied from [H-group: The Prompt (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/prompt-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Prompt (Question 2)

## Question

- Cathy has two unknown 3's in her hand.
- What are the two possible clues that Alice can give?
- Which one is better and why?

[Example diagram: PromptQuestion2](../../examples/prompt-question-2/prompt-question-2.yml)

```yaml
stacks:
  - r: 1
  - y: 2
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
      - type: y4
        border: false
      - type: b4
        border: false
      - type: p1
        border: false
      - type: g1
        border: false
      - type: p4
        border: false
  - cards:
      - type: y2
        border: false
      - type: r1
        border: false
      - type: y3
      - type: b3
      - type: p5
```

## Solution

- First, Alice can clue yellow to Bob:
  - This would be a _Play Clue_ on the yellow 4, which would _Prompt_ Cathy.
  - Cathy would play her leftmost 3 and then stop.
- Second, Alice can also clue blue to Bob:
  - This would be a _Play Clue_ on the blue 4, which would _Prompt_ Cathy.
  - Cathy would try to play blue 3 from slot 3. However, it would be yellow 3, which would successfully play.
  - Cathy would know that she is promised to have the blue 3, so she would go on to play her other 3 as the blue 3.
- The second clue is better than the first clue because it gets Cathy to play two cards instead of one.

[Example diagram: PromptAnswer2](../../examples/prompt-question-2/prompt-answer-2.yml)

```yaml
stacks:
  - r: 1
  - y: 2
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
      - type: y4
        border: false
      - type: b4
        clue: b
        middleNote: b4
        below: Play
      - type: p1
        border: false
      - type: g1
        border: false
      - type: p4
        border: false
  - cards:
      - type: y2
        border: false
      - type: r1
        border: false
      - type: y3
        middleNote: b3?
        below: Prompt
      - type: b3
        middleNote: b3
        below: Prompt
      - type: p5
```
