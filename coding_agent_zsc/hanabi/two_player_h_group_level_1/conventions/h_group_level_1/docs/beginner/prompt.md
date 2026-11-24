> Copied from [H-group: The Prompt](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/prompt.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# The Prompt

- In the example below:
  - Bob has a clued red card in his hand, but the full identity of the card is still unknown to him. In this situation, he knows he is supposed to just keep the card in his hand and continue to discard unclued cards.
  - Alice clues red to Cathy, which touches a red 4. This must be a _Play Clue_ on the red 4.
  - Bob sees that red 4 is currently unplayable, since the red 3 is not played yet. Thus, this clue implies that someone has the red 3.
  - Bob does not see a red 3 clued in anyone else's hand. So, Alice must be promising him that his unknown red card is red 3 and that he can safely play it.

[Example diagram: Prompt1](../../examples/prompt/prompt-1.yml)

```yaml
stacks:
  - r: 2
  - y: 2
  - g: 2
  - b: 2
  - p: 2
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
      - type: r
        above: Red 3
        below: Prompt
      - type: x
      - type: x
  - cards:
      - type: x
      - type: r
        clue: r
        middleNote: r4
        below: Play
      - type: x
      - type: x
      - type: x
```

- We call this move a _Prompt_ because it is "prompting" a player to play a card that they would have otherwise held on to for a while longer.
- In this way, clues can deliver messages to a different player than the person who actually received the clue.
- If a player is _Prompted_ and there are multiple cards in the hand that the _Prompt_ could apply to, they should play the leftmost.
- In the example below:
  - Bob is _Prompted_ for a red 3 in the exact same way as the previous example.
  - Bob has two different cards clued as red in slot 4 and slot 5.
  - Since both cards are unknown to Bob, Bob should choose to play the slot 4 card since it is the leftmost.

[Example diagram: Prompt2](../../examples/prompt/prompt-2.yml)

```yaml
stacks:
  - r: 2
  - y: 2
  - g: 2
  - b: 2
  - p: 2
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
      - type: r
        above: Red 3
        below: Prompt
      - type: r
  - cards:
      - type: x
      - type: r
        clue: r
        middleNote: r4
        below: Play
      - type: x
      - type: x
      - type: x
```

- In the example below:
  - Bob has an unknown 3 in his hand.
  - Alice clues number 4 to Bob, which touches a 4. It must be a _Play Clue_ on the 4.
  - There is no 4 that could possibly be playable right now. Thus, this must be a _Self-Prompt_, meaning that Bob's 3 must be either green 3 or blue 3.
  - Bob plays his 3, and it is blue 3. Thus, he knows that the 4 must connect and be a blue 4.

[Example diagram: Prompt3](../../examples/prompt/prompt-3.yml)

```yaml
stacks:
  - r: 1
  - y: 1
  - g: 2
  - b: 2
  - p: 1
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
      - type: 4
        clue: 4
        middleNote: gb4
        below: Play
      - type: x
      - type: 3
        above: Blue 3
        middleNote: gb3
        below: Prompt
      - type: x
  - text: "After blue 3 plays:"
  - offset: -1
    cards:
      - type: x
      - type: x
      - type: 4
        middleNote: b4
      - type: x
      - type: x
```

- The _Prompt_ is a **promise** that the player has the connecting card. Therefore, if the leftmost card is not the correct one, the player should continue to play his clued cards (left to right) until the expected card is found. (It is the responsibility of the player giving the clue to ensure that all those previous cards will play successfully.)
- In the example below:
  - Bob has three separate unknown 3's in his hand.
  - Alice clues red to Cathy, which touches a red 4. This is a _Play Clue_ on the red 4.
  - Bob knows that he must have the red 3 and this is a _Prompt_. Bob plays his leftmost 3 from slot 2.
  - However, after playing it, Bob sees that it is not the red 3 - it was the blue 3 (which played successfully). Bob now knows that the next-leftmost 3 must be the red 3 (on slot 3).
  - Cathy can see the clued red 3 in Bob's hand. Thus, she knows from the _Good Touch Principle_ that her red card must be the red 4.
  - Cathy does something unrelated.
  - On his next turn, Bob tries to play his slot 3 card as the red 3. However, after playing it, Bob sees that it was not the red 3 - it was the green 3 (which played successfully). Bob now knows that his final 3 must be the red 3 (on slot 4).
  - On his next turn, Bob plays his slot 4 card, and it is red 3.
  - Cathy can now finally play her red 4.

[Example diagram: Prompt4](../../examples/prompt/prompt-4.yml)

```yaml
stacks:
  - r: 2
  - y: 1
  - g: 2
  - b: 2
  - p: 1
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
      - type: 3
        above: Blue 3
        middleNote: r3?
        below: Prompt
      - type: 3
        above: Green 3
        middleNote: r3?
        below: Prompt
      - type: 3
        above: Red 3
        middleNote: r3!
        below: Prompt
      - type: x
  - space
  - cards:
      - type: x
      - type: x
      - type: r
        clue: r
        middleNote: r4
        below: Play
      - type: x
      - type: x
```
