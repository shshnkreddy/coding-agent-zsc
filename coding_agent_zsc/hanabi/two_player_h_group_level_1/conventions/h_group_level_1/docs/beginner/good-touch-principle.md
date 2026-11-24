> Copied from [H-group: Good Touch Principle (GTP)](https://github.com/hanabi/hanabi.github.io/blob/a1b0e25ec3536b61dc48efe09468d534b5806962/docs/beginner/good-touch-principle.mdx). CC BY-SA 4.0; offline formatting and scoped omissions are recorded in [the source manifest](../../source_manifest.json).

# Good Touch Principle (GTP)

- When cluing cards, we follow _Good Touch Principle_ - **we only touch cards that will eventually be played**.
  - It would be a _Bad Touch_ if someone touched a card that was already played on the stacks or was already clued in someone else's hand.
- _Good Touch Principle_ is a powerful building block for logic and reasoning. If everyone only touches good cards, then players can greatly narrow down the possibilities for the clued cards in their hand!
- In the example below:
  - Alice wants to give a _Play Clue_ to the red 2.
  - Alice has a choice between cluing red and cluing number 2.
  - If she clues number 2, it would uniquely touch the red 2 on slot 1.
  - If she clues red, then it would touch slot 1, slot 2, and slot 3. This clue would violate _Good Touch Principle_, since the red 4's would be duplicated in Bob's hand.
  - Thus, the correct clue for Alice to give is number 2.

[Example diagram: GoodTouchPrinciple1Correct](../../examples/good-touch-principle/good-touch-principle-1-correct.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 0
  - b: 0
  - p: 0
bigText:
  text: Correct
  color: Green
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r2
        clue: 2
        below: Play
      - type: r4
        border: false
      - type: r4
        border: false
      - type: x
      - type: x
```

[Example diagram: GoodTouchPrinciple1Illegal](../../examples/good-touch-principle/good-touch-principle-1-illegal.yml)

```yaml
stacks:
  - r: 1
  - y: 0
  - g: 0
  - b: 0
  - p: 0
bigText:
  text: Illegal!
players:
  - clueGiver: true
    cards:
      - type: x
      - type: x
      - type: x
      - type: x
      - type: x
  - cards:
      - type: r2
        clue: r
        below: Play
      - type: r4
        clue: r
        below: Dupe
      - type: r4
        clue: r
        below: Dupe
      - type: x
      - type: x
```

- In the example below:
  - Alice has an unknown purple card in her hand.
  - Alice can see that the purple stack is played up to purple 4. The only other useful purple card left is the purple 5.
  - Alice also knows that according to _Good Touch Principle_, every clued card will eventually be played.
  - Thus, by process of elimination, she can reason that the unknown purple card in her hand must be purple 5, and she can immediately play it.

[Example diagram: GoodTouchPrinciple2](../../examples/good-touch-principle/good-touch-principle-2.yml)

```yaml
stacks:
  - r: 3
  - y: 2
  - g: 2
  - b: 2
  - p: 4
players:
  - cards:
      - type: x
      - type: x
      - type: x
      - type: p
      - type: x
  - text: "From Good Touch Principle:"
  - offset: -1
    cards:
      - type: x
      - type: x
      - type: x
      - type: p
        middleNote: p5
        below:
          text: Play
          color: cyan
      - type: x
```

- In the example below:
  - In the middle of a game, Alice clues yellow to Bob, touching slot 2 and slot 3.
  - The focus of the clue is on slot 2. This must be a _Play Clue_, so Bob marks his slot 2 card as a yellow 3.
  - Slot 3 was also touched by the clue, but it was not the focus. From _Good Touch Principle_, Bob knows that it must be playable in the future.
  - Therefore, Bob marks his slot 3 card as either yellow 4, or yellow 5.

[Example diagram: GoodTouchPrinciple3](../../examples/good-touch-principle/good-touch-principle-3.yml)

```yaml
stacks:
  - r: 1
  - y: 2
  - g: 0
  - b: 1
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
      - type: y
        clue: y
        middleNote: y3
        below: Play
      - type: y
        clue: y
        middleNote: y4, y5
      - type: x
      - type: x
```
