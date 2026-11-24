# Selected layout: block_grid_2p

Source: repository custom layout

```text
WWWWWWWWWRWWWWWWWWW
W                 W
W W0W W1W W1W W0W W
W WWW WWW WWW WWW W
W WPW W2W W2W WPW W
W                 W
W WBW WBW WBW WBW W
W WWW WWW WWW WWW W
W WWW WLW WLW WWW W
W    A       A    W
W WWW WLW WLW WWW W
W WWW WWW WWW WWW W
W WBW WBW WBW WBW W
W                 W
W WPW W2W W2W WPW W
W WWW WWW WWW WWW W
W W0W W1W W1W W0W W
W                 W
WWWWWWWWWXWWWWWWWWW
```

Legend: `W` wall/counter, `A` agent spawn, `X` delivery goal, `P` pot,
`B` plate pile, `R` recipe indicator, `L` button recipe indicator, and each
digit is the pile for that ingredient index. Spaces are traversable floor.

Configured recipes:

```json
[
  [
    0,
    0,
    1
  ],
  [
    1,
    1,
    2
  ],
  [
    0,
    2,
    2
  ],
  [
    0,
    1,
    2
  ]
]
```
