> Copied from [H-group: Delayed Play Clues](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/delayed-play-clues.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Delayed Play Clues

- It is permitted to give a _Play Clue_ to a card that can only play on the stacks _after_ other people have played cards of the same suit. This is called a _Delayed Play Clue_.
- We agree to treat **all _Play Clues_** as **potential _Delayed Play Clues_**.
- For example, in a 3-player game:
  - Alice clues Bob number 2, which is a _Play Clue_ on a 2.
  - Now, it is Bob's turn. If his 2 is the red 2, then he can play it right now.
  - However, Bob also knows that this could be a _Delayed Play Clue_ through the yellow 1, the green 1, or the blue 1. So, Bob must wait for Cathy to play all of her 1's first before he can try to play the 2.
  - Bob records this information on the card as a note, and then discards.

[Example diagram: DelayedPlayClues](../../examples/delayed-play-clues/delayed-play-clues.yml)

```yaml
stacks:
  - r: 1
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
      - type: x
      - type: 2
        clue: 2
        below: Play
      - type: x
      - type: x
  - cards:
      - type: x
      - type: y1
      - type: g1
      - type: b1
      - type: x
```
