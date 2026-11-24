> Copied from [H-group: Minimum Clue Value Principle (MCVP)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/minimum-clue-value-principle.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Minimum Clue Value Principle (MCVP)

### No Nonsense Clues

- Usually, players only discard when they have nothing else to do. But don't take this the wrong way.
- Players are **not allowed** to just clue any card:
  - Players are **only allowed** to give _Play Clues_ to actual playable cards.
  - Players are **only allowed** to give _Save Clues_ to the specific cards outlined in the _Save Principle_ section.
- If a player has no cards to play and cannot give a legal _Play Clue_ or _Save Clue_, then they **must discard**.

### Tempo Clues

- Furthermore, players might be tempted to **clue a card again that already has a clue on it** in order to make it play right now. We call this kind of clue a _Tempo Clue_.
- A _Tempo Clue_ is defined as a _Play Clue_ that does not "get" any brand new cards.
- _Tempo Clues_ are quite inefficient - the team is using two clues to get one card. Since _Tempo Clues_ are so bad, we say that players are **only allowed to perform _Tempo Clues_ in special situations**, like when a player is forced to give a clue.
- For example, in a 3-player game:
  - Alice is in a special situation where she cannot discard.
  - Bob has a 2 in his hand that was saved with a _2 Save_ clue earlier on in the game. He does not know what color it is but assumes that it is either a red 2 (which is playable) or a blue 2 (which is not playable).
  - Alice clues red to Bob, "filling in" the red 2. This is a _Tempo Clue_ because there are no new cards introduced. (Since this is so inefficient, Alice would never do this unless she were in a special stalling situation.)
  - Bob can now play the red 2.

[Example diagram: TempoClue](../../examples/minimum-clue-value-principle/tempo-clue.yml)

```yaml
stacks:
  - r: 1
  - y: 3
  - g: 3
  - b: 0
  - p: 3
players:
  - clueGiver: true
    cards:
      - type: 5
      - type: 5
      - type: 5
      - type: 5
      - type: 5
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: r2
        clue: r
        retouched: true
        below: Tempo
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
```

### Minimum Clue Value Principle (MCVP)

- So, if a player only has the option to give a _Tempo Clue_ or to discard, then they **must discard**.
- We formally express this concept as _Minimum Clue Value Principle_: **all clues must get at least one previously unclued card played or saved**.
- In other words, as a group, we hold ourselves to a high standard, and we don't give inefficient clues.
