# Cat & Dog

Alice goes first knowing the pet. Bob sees the public consequences of her decision. A cat and a dog are equally likely. Both seats receive the same team reward.

## Local observation

`CatDogObservation` contains `features` (float32, shape `[10]`) and
`legal_actions` (boolean, shape `[4]`). Every feature is zero or one.

| Index | Meaning |
| --- | --- |
| 0, 1 | Observer is Alice / Bob |
| 2, 3 | Current turn is Alice's / Bob's |
| 4, 5 | Private pet is cat / dog; visible only to Alice, always zero for Bob |
| 6, 7 | Alice chose signal off / on; visible to both only on Bob's turn |
| 8, 9 | Revealed pet is cat / dog; visible to both only on Bob's turn after reveal |

All signal and revealed-pet features are zero at reset. Reveal produces no
cheap-talk signal. Signals carry no built-in pet meaning. Bob does not receive
Alice's private features. Policies receive only their own observation, reset,
private memory and PRNG; they do not receive hidden state or rewards as inputs.

## Actions and legality

Each policy outputs a local integer action index. The following mappings run
from local index to canonical role-specific action index:

- Alice: `[0, 1, 2, 3]`
- Bob: `[0, 1, 2, 3]`

| Canonical index | Alice | Bob |
| --- | --- | --- |
| 0 | Signal off | Bail |
| 1 | Signal on | Guess cat |
| 2 | Reveal the pet | Guess dog |
| 3 | Bail | Extra bail (masked out) |

On Alice's turn all four of her actions are legal. On Bob's turn only his
canonical actions 0, 1 and 2 are legal. The native game defines Bob's extra
bail as another bail, but the evaluator rejects it because it is masked out.
Masks are expressed in local indices after applying the configured permutation.
The inactive role's mask allows only its canonical action 0 as a placeholder.
Both policy functions process each transition; inactive-role outputs do not
affect the game. Illegal active-role actions invalidate the evaluation.

## Rewards and termination

- Alice bail: shared reward `1.0`; terminate immediately.
- Alice signal off/on: reward `0`; proceed to Bob.
- Alice reveal: reward `-5.0`; disclose the pet and proceed to Bob.
- Bob bail: reward `0.5`; terminate.
- Bob correct guess: reward `10.0`; terminate.
- Bob wrong guess: reward `-10.0`; terminate.

The reveal cost is charged on Alice's turn regardless of Bob's later action.
Episodes last at most two transitions. The runtime automatically resets after
termination and signals the episode boundary to both policies. Policy memory
is private to each role and reset at episode boundaries.
