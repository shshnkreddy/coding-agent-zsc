> Copied from [H-group: Single Card Focus](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/single-card-focus.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Single Card Focus

- So far, we have learned about the two types of clues: _Play Clues_ and _Save Clues_.
- But all of the _Play Clues_ and _Save Clues_ that we have seen so far have only "touched" a single card.
- Obviously, it is also possible to give a clue that touches **two or more** cards.
- We agree that if a clue touches two or more cards, the clue only delivers a message about **one specific card**. This is referred to as the _focus_ of the clue.
- With every clue, only **one card is focused**. Nothing specific is implied about non-focused cards (except that they will be useful at some point later on).
- For example, if Alice gives a _Play Clue_ to Bob that touches two cards, then Bob can **only play the focused card**. He has to hold on to the non-focused card for later.

### New Cards

- When determining the focus of a clue, you need to look for "new" cards.
- New cards are specifically defined as **cards that are touched by a clue that did not have any clues "on" them already**.

### Determining the Focus

So, when two or more cards are touched by a clue, which card is focused?

1. If no cards are new, then **the focus is on the leftmost re-touched card**.
1. If the chop card was touched, then **the focus is on the chop**.
   (This is still true even if other cards were touched in addition to
   the chop!)
1. Otherwise, **the focus is on the leftmost new card**.

### Example 1: Leftmost Focus

- In the example below, Alice clues red to Bob. This must be a _Play Clue_.
- Before this clue, Bob's hand was completely unclued. So, with this clue, **two new cards were introduced** - the red card on slot 2 and the red card on slot 4.
- Bob knows that in this situation, the focus of the clue is the leftmost of the two new cards. This must be a _Play Clue_ on the next red card.

[Example diagram: LeftMostFocus](../../examples/single-card-focus/leftmost-focus.yml)

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
      - type: r
        clue: r
        below: Focus
      - type: x
      - type: r
        clue: r
      - type: x
```

### Example 2: Chop-Focus

- In the example below, no cards have been discarded and Alice clues red to Bob. This must be a _Play Clue_.
- Before this clue, Bob's hand was completely unclued. So, with this clue, **three new cards were introduced** - slot 1, slot 2, and slot 5.
- Before this clue, Bob's chop was slot 5. So, since the clue touched his chop and _Chop-Focus_ has precedence, this must be a _Chop-Focus Play Clue_ on the next red card.

[Example diagram: ChopFocus](../../examples/single-card-focus/chop-focus.yml)

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
      - type: r
        clue: r
      - type: r
        clue: r
      - type: x
      - type: x
      - type: r
        clue: r
        below: Focus
```

- Note that even though it touches Bob's chop card, Bob knows that this cannot be a _Save Clue_. Specifically:
  - It can't be a _5 Save_ on a red 5, because Alice would have clued number 5 for that.
  - It can't be a _2 Save_ on a red 2, because Alice would have clued number 2 for that.
  - It can't be a _Critical Save_, because there are no red cards in the trash yet.
