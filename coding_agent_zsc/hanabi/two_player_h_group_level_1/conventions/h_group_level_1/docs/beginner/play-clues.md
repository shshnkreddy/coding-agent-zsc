> Copied from [H-group: Play Clues](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/play-clues.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Play Clues

- The first type of clue is called a _Play Clue_.
- The goal of Hanabi is to get all of the cards played on the stacks. Thus, we give _Play Clues_ to each other to communicate that specific cards are playable.
- Here, Alice clues red to Bob, touching a card on slot 2.
- This is a message to "play this card". Thus, it must be the red 1 (because no red cards are played on the stacks yet).
- After the clue, all players make a note on the card that it is a red 1.

[Example diagram: PlayClue1](../../examples/play-clues/play-clue-1.yml)

```yaml
stacks:
  - r: 0
  - y: 0
  - g: 0
  - b: 0
  - p: 0
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
        middleNote: r1
        below: Play
      - type: x
      - type: x
      - type: x
```

- _Play Clues_ can be given with either a color clue or a number clue.
- Here, Alice clues number 1 to Bob, touching a card on slot 3.
- Similar to before, this is a message to "play this card". Thus, it must be a yellow 1 or a blue 1 (since those are the only two 1's that are not played yet).
- After the clue, all players make a note on the card that it is yellow 1 or a blue 1.

[Example diagram: PlayClue2](../../examples/play-clues/play-clue-2.yml)

```yaml
stacks:
  - r: 2
  - y: 0
  - g: 1
  - b: 0
  - p: 3
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
      - type: 1
        clue: 1
        middleNote: y1, b1
        below: Play
      - type: x
      - type: x
```
