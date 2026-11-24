> Copied from [H-group: The Chop](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/chop.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Chop

- In Hanabi, players discard cards from their hand to generate clues for the team. But which card should they discard?
- The rightmost **unclued card** in a player's hand is called their "chop" card.
  - This is a metaphor of a card being "on the chopping block".
- When a player needs to discard, they should discard their chop card. This way, **everyone discards in a predictable way**.

## Without Clues

- In the example below:
  - Alice's chop card is the red 5 (on slot 5).
  - Bob's chop card is the red 3 (on slot 5).

[Example diagram: Chop1](../../examples/chop/chop-1.yml)

```yaml
players:
  - cards:
      - type: y3
        border: false
      - type: b1
        border: false
      - type: y2
        border: false
      - type: g1
        border: false
      - type: r5
        border: false
        below: Chop
  - cards:
      - type: g5
        border: false
      - type: r2
        border: false
      - type: b4
        border: false
      - type: b3
        border: false
      - type: r3
        border: false
        below: Chop
```

## With Clues

- In the example below:
  - Alice's chop card is the blue 4 (on slot 4), because slot 5 is already clued.
  - Bob's chop card is the blue 3 (on slot 2), because slot 3, slot 4, and slot 5 are already clued.

[Example diagram: Chop2](../../examples/chop/chop-2.yml)

```yaml
players:
  - cards:
      - type: r2
        border: false
      - type: r3
        border: false
      - type: r3
        border: false
      - type: b4
        border: false
        below: Chop
      - type: r5
  - cards:
      - type: b4
        border: false
      - type: b3
        border: false
        below: Chop
      - type: g3
      - type: g4
      - type: g5
```
