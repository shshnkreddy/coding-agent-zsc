> Copied from [H-group: The Finesse (Question 2)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/finesse-question-2.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Finesse (Question 2)

## Question

- Alice clues yellow to Cathy, touching a yellow 4 as a _Play Clue_.
- From Bob's perspective, is this a _Finesse_? Why or why not?

[Example diagram: FinesseQuestion2](../../examples/finesse-question-2/finesse-question-2.yml)

```yaml
stacks:
  - r: 0
  - y: 2
  - g: 2
  - b: 4
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
      - type: x
      - type: x
      - type: 3
      - type: 5
  - cards:
      - type: y4
        clue: y
        below: Play
      - type: x
      - type: x
      - type: x
      - type: x
```

## Solution

- No, it is not a _Finesse_.
- Bob has a clued 3 in his hand, which could be the yellow 3. Therefore, this could be a _Prompt_ on Bob.
- Bob knows that _Prompts_ take precedence over _Finesses_. Since a _Prompt_ is possible, it cannot be a _Finesse_.

[Example diagram: FinesseAnswer2](../../examples/finesse-question-2/finesse-answer-2.yml)

```yaml
stacks:
  - r: 0
  - y: 2
  - g: 2
  - b: 4
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
      - type: x
      - type: x
      - type: 3
        middleNote: y3
        below: Prompt
      - type: 5
  - cards:
      - type: y4
        clue: y
        middleNote: y4
        below: Play
      - type: x
      - type: x
      - type: x
      - type: x
```
