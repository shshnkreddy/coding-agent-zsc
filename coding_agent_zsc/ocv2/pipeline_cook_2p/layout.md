# Selected layout: pipeline_cook_2p

Source: repository custom layout

```text
WW0W1W2WWWWWWPWWWWWWWWWWWWW
W       WW       WW       W
R       WW       WW       W
0  0R1  WL  PBP  WL  BWB  W
1 AWWW      WWW      WWWA X
2  2WW  WW  PBP  WW  BWB  W
R       WW       WW       W
W       WW       WW       W
WW0W1W2WWWWWWPWWWWWWWWWWWWW
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
