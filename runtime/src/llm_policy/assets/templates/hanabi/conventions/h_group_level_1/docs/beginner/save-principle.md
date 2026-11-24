> Copied from [H-group: Save Principle](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/save-principle.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Save Principle

- Earlier, we explained that the team must save certain types of cards with _Save Clues_.
- But above and beyond this, we also agree that we must "get" all of the playable cards. In other words, **we don't let each other discard playable cards!**
- So, the **full list** of what we must **not discard** is the following:
  - All 5's
  - Unique 2's (i.e. 2's that are the only copy currently visible)
  - Critical cards (i.e. cards that have all other copies discarded)
  - Unique playable cards (i.e. playable cards that are the only copy currently visible)
- Not letting each other discard important cards is called _Save Principle_ and **it is the most important principle of them all**. You are allowed to **break any other principle or rule** in order to prevent a card in this list from being discarded.

### Violating Good Touch Principle

- Sometimes, you will have to violate _Good Touch Principle_ in order to save a specific card, and that is okay.
- When this happens, the clue receiver will write wrong notes on some of their cards.
- In the example below:
  - Alice needs to save the final blue 3 on Bob's chop.
  - Alice knows that she is allowed to perform _Critical Saves_ with either a color clue or a number clue.
  - If Alice clues blue, it will violate _Good Touch Principle_ because it will also touch both copies of blue 4.
  - If Alice clues number 3, it will violate _Good Touch Principle_ because it will touch the red 3 (and the red 3 is already played on the stacks).
  - Here, Alice knows that it is better to violate _Good Touch Principle_ than to lose the chance at a perfect score. Thus, Alice must choose between the lesser of two evils and decide between cluing blue and cluing number 3.
  - Alice decides to clue blue to Bob instead of number 3 because at least one of the blue 4's will be useful later on.
  - Bob will write "b2, b3" on his saved card on slot 5. (From Bob's perspective, the card could be either a _Play Clue_ on blue 2, or a _Save Clue_ on blue 3.)
  - From _Good Touch Principle_, Bob will write "b2, b3, b4, b5" on his two other blue cards.

[Example diagram: SavePrinciple](../../examples/save-principle/save-principle.yml)

```yaml
stacks:
  - r: 3
  - y: 0
  - g: 2
  - b: 1
  - p: 0
discarded:
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
      - type: b4
        border: false
      - type: r3
        border: false
      - type: g1
        border: false
      - type: b4
        border: false
      - type: b3
        border: false
  - text: "Save Principle overrides Good Touch Principle:"
  - space
  - offset: -1
    cards:
      - type: b4
        clue: b
        middleNote: b2345
      - type: r3
        border: false
      - type: g1
        border: false
      - type: b4
        clue: b
        middleNote: b2345
      - type: b3
        clue: b
        middleNote: b2, b3
        below: Save
```
