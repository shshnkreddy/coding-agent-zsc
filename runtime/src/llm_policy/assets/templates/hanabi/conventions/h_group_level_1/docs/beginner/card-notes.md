> Copied from [H-group: Card Notes](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/card-notes.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Card Notes

When you play Hanabi, you will need to write _card notes_. This page explains why.

## Explicit and Implicit Information

- In Hanabi, players will know _explicit_ information and _implicit_ information about the cards in their hand:
  - _Explicit_ information is from the clues "on" the card.
  - _Implicit_ information is from the context of the game _when_ the card was clued.
- For example, in a 3-player game:
  - It is the first turn of the game.
  - Alice clues Bob red, touching a card on his slot 1. Alice intends for this to be a _Play Clue_, signaling to Bob that he has a red 1 and that he should play it on the stacks right now.
  - From Bob's perspective, he knows that _explicitly_, his red card can be red 1, red 2, red 3, red 4, or red 5.
  - But Bob also knows that _implicitly_, Alice intends for him to play this red card, so it must be exactly the red 1.
  - Colors in card notes are conventionally shortened to the first letter.

[Example diagram: CardNotes1](../../examples/card-notes/card-notes-1.yml)

```yaml
players:
  - cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r
        clue: r
        middleNote: r1
      - type: x
      - type: x
      - type: x
      - type: x
```

## Players Need to Track Implicit Information

- On [Hanab Live](https://hanab.live/), the interface automatically will keep track of clues that touch cards. And the interface will automatically narrow down the identities of the cards as they "get" more clues.
- However, this automated tracking of _explicit_ information is not sufficient - there is a lot more _implicit_ information than _explicit_ information.
- In Hanabi games where players only give basic _Play Clues_ to each other (like in the previous example), remembering all of the _implicit_ information about the cards is easy. But as games get more and more complicated, remembering _implicit_ information about every card in the game is impossible.
- Thus, in order to play Hanabi properly, players **must** go above and beyond the automated _explicit_ tracking and record all of the implicit information themselves.

## Avoiding Mistakes

- Since the red clue above was so simple, you may think that writing a note of "r1" is silly and a waste of time. However, realize that Bob **might not play the red 1 for many, many turns**, at which point the game state may have dramatically changed.
- The most common beginner mistake that we see is failing to write a note on a card, and then forgetting that piece of information later on in the game.
- Do not forget to do this! Write _card notes_ after **every single clue**, no matter how simple or complicated it is!
