> Copied from [H-group: Minimum Clue Value Principle (Question 1)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/minimum-clue-value-principle-question-1.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Minimum Clue Value Principle (Question 1)

## Question

- Alice is in a special situation where she cannot discard.
- Bob's blue 4 has been previously touched with a blue clue.
- Cathy's red 2 and yellow 2 have previously been touched with a number 2 clue.
- Enumerate all of the legal _Tempo Clues_ that Alice can give.

[Example diagram: MinimumClueValuePrincipleQuestion1](../../examples/minimum-clue-value-principle-question-1/minimum-clue-value-principle-question-1.yml)

```yaml
stacks:
  - r: 1
  - y: 1
  - g: 0
  - b: 3
  - p: 0
bigText:
  text: Must clue
  color: cyan
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p5
        border: false
      - type: g5
        border: false
      - type: g3
        border: false
      - type: b4
        middleNote: b
      - type: r1
        border: false
  - cards:
      - type: b2
        border: false
      - type: g4
        border: false
      - type: y1
        border: false
      - type: r2
        middleNote: 2
      - type: y2
        middleNote: 2
```

## Solution

- Alice could give a _Tempo Clue_ to the blue 4 in Bob's hand:
  - A number 4 clue would be legal, since it would only touch slot 4. This would "fill-in" the 4 to be a blue 4.
  - A blue clue would be legal, since it would only touch slot 4. This would not "fill-in" the card, but the clue would still mean to "play the card", so it would still work properly.
- Alice could give a _Tempo Clue_ to the red 2 in Cathy's hand:
  - A number 2 clue would not be legal, since it would focus slot 1 instead of slot 4.
  - A red clue would be legal, since it would only touch slot 4.
- Alice cannot give a _Tempo Clue_ to the yellow 2 in Cathy's hand:
  - A number 2 clue would not be legal, since it would focus slot 1 instead of slot 5.
  - A yellow clue would not be legal, since it would focus slot 3 instead of slot 5.

[Example diagram: MinimumClueValuePrincipleAnswer1](../../examples/minimum-clue-value-principle-question-1/minimum-clue-value-principle-answer-1.yml)

```yaml
stacks:
  - r: 1
  - y: 1
  - g: 0
  - b: 3
  - p: 0
bigText:
  text: Must clue
  color: cyan
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: p5
        border: false
      - type: g5
        border: false
      - type: g3
        border: false
      - type: b4
        clue: 4
        retouched: true
        middleNote: b4
        below: Tempo
      - type: r1
        border: false
  - text: OR
  - offset: -1
    cards:
      - type: p5
        border: false
      - type: g5
        border: false
      - type: g3
        border: false
      - type: b4
        clue: b
        retouched: true
        middleNote: b4
        below: Tempo
      - type: r1
        border: false
  - text: OR
  - cards:
      - type: b2
        border: false
      - type: g4
        border: false
      - type: y1
        border: false
      - type: r2
        clue: r
        retouched: true
        middleNote: r2
        below: Tempo
      - type: y2
        middleNote: 2
```
