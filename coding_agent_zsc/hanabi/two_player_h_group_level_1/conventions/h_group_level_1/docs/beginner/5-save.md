> Copied from [H-group: The 5 Save](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/5-save.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The 5 Save

- In Hanabi, there is only one copy of each 5. So, we need to stop players from discarding them.
- We agree that 5's must be saved with **a number 5 clue**. This is called a _5 Save_.
- In the example below:
  - Alice clues Bob number 5, which touches a single 5 on slot 5.
  - Before this clue, Bob did not have any clued cards in his hand.
  - Since the clue was focused on his chop, Bob knows that this must be a _5 Save_ on the 5. It can be any 5 in the game.

[Example diagram: FiveSave](../../examples/5-save/5-save.yml)

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
      - type: 5
        clue: 5
        below: Save
```
