> Copied from [H-group: The 2 Save](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/2-save.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The 2 Save

- Even though there are two copies of each 2, we agree that 2's are so important that we need to save them in a manner similar to 5's.
- We agree that 2's should be saved with **a number 2 clue**. This is called a _2 Save_.
- In the example below:
  - Alice clues Bob number 2, which touches a 2 on slot 5.
  - Before this clue, Bob did not have any clued cards in his hand.
  - Since the clue was focused on his chop, Bob knows that this must be a _2 Save_ on the 2. It can be any 2 in the game.

[Example diagram: TwoSave](../../examples/2-save/2-save.yml)

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
      - type: 2
        clue: 2
        below: Save
```

### When Multiple Copies are Visible

- You can think of the _2 Save_ as a last resort to save a 2. Thus, players are not allowed to perform a _2 Save_ on a 2 if that same 2 is visible in someone else's hand.
- This is because the team does not actually need to save that 2 quite yet - they can save the other copy later on.
- For example, in a 3-player game:
  - Bob has a red 2 on slot 5 (his chop).
  - Cathy has a red 2 on slot 3.
  - Thus, it would be illegal for Alice to perform a _2 Save_ on Bob's red 2.

[Example diagram: TwoSaveIllegal](../../examples/2-save/2-save-illegal.yml)

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
      - type: g5
        border: false
      - type: b3
        border: false
      - type: b3
        border: false
      - type: y3
        border: false
      - type: r2
        border: false
        below: Dupe
  - cards:
      - type: b5
        border: false
      - type: y2
        border: false
      - type: r2
        border: false
        below: Dupe
      - type: y3
        border: false
      - type: y4
        border: false
```
