> Copied from [H-group: Card Layout](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/card-layout.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Card Layout

In this guide, there will be some images to show off the concepts introduced.

## Play Stacks & Player Hands

- The five **play stacks** are on the left.
- The **player hands** are on the right.
- Alice is always player 1. (She will always go first.)
- Bob is always player 2. (He will always go second.)
- Cards are drawn and added to the left side to each hand.
  - Thus, Alice's newest card is the blue 1 and Bob's newest card is the red 2.
- Conversely, the oldest card is the rightmost card.
  - Thus, Alice's oldest card is the blue 5 and Bob's oldest card is the red 1.
- A card's _slot_ refers to its position in the hand, from left to right.
  - Thus, the blue 1 is on slot 1, the blue 2 is on slot 2, etc.

[Example diagram: CardLayout1](../../examples/card-layout/card-layout-1.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 0
  - b: 0
  - p: 0
players:
  - cards:
      - type: b1
        border: false
        below:
          text:
            - Slot 1
            - Newest
      - type: b2
        border: false
        below: Slot 2
      - type: b3
        border: false
        below: Slot 3
      - type: b4
        border: false
        below: Slot 4
      - type: b5
        border: false
        below:
          text:
            - Slot 5
            - Oldest
  - cards:
      - type: r2
        border: false
        below:
          text:
            - Slot 1
            - Newest
      - type: r3
        border: false
        below: Slot 2
      - type: r4
        border: false
        below: Slot 3
      - type: r5
        border: false
        below: Slot 4
      - type: r1
        border: false
        below:
          text:
            - Slot 5
            - Oldest
```

## Clues

- Arrows indicate that a card was "touched" by the last clue that was given.
- Clued cards will have permanent yellow borders around them. Unclued cards will not have these yellow borders.

[Example diagram: CardLayout2](../../examples/card-layout/card-layout-2.yml)

```yaml
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
      - type: x
  - cards:
      - type: x
      - type: x
      - type: 1
        clue: 1
      - type: 1
        clue: 1
      - type: x
```

- When a card is "touched" by a clue for the first time, a white arrow will point to the card and a yellow border will be applied.
- When an already-clued card is "retouched" by subsequent clues, a gray arrow will point to the card.

[Example diagram: CardLayout3](../../examples/card-layout/card-layout-3.yml)

```yaml
players:
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: b
        clue: b
      - type: x
      - type: 1
      - type: b1
        clue: b
        retouched: true
      - type: x
```
