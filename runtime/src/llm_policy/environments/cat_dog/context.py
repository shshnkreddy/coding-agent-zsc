"""Factual, non-executable Cat & Dog documentation for coding agents."""


def specification(runtime):
    native = runtime.native_env
    return {
        "roles": {"agent_0": "Alice", "agent_1": "Bob"},
        "feature_indices": {
            "is_alice": 0, "is_bob": 1, "alice_turn": 2, "bob_turn": 3,
            "alice_private_cat": 4, "alice_private_dog": 5,
            "signal_off": 6, "signal_on": 7, "revealed_cat": 8, "revealed_dog": 9,
        },
        "canonical_actions": {
            "alice": ["signal_off", "signal_on", "reveal", "bail"],
            "bob": ["bail", "guess_cat", "guess_dog", "extra_bail"],
        },
        "local_to_canonical_actions": {
            "alice": native.alice_action_permutation.tolist(),
            "bob": native.bob_action_permutation.tolist(),
        },
        "rewards": {name: getattr(native, name) for name in (
            "alice_bail_reward", "bob_bail_reward", "reveal_cost", "correct_reward", "wrong_reward")},
        "pet_probabilities": {"cat": 0.5, "dog": 0.5},
        "episode_length_max": 2,
    }


def render(record):
    rewards = record["rewards"]
    permutations = record["local_to_canonical_actions"]
    return f"""# Cat & Dog

This is a two-player cooperative communication game. At each reset a cat or
dog is sampled uniformly. Agent 0 is Alice; agent 1 is Bob. Alice acts first
and privately observes the pet. Unless she bails, Bob acts next and the episode
ends. Both players receive the same reward. Episode return is the sum of the
turn rewards, counted once rather than summed over players.

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

- Alice: `{permutations['alice']}`
- Bob: `{permutations['bob']}`

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

- Alice bail: shared reward `{rewards['alice_bail_reward']}`; terminate immediately.
- Alice signal off/on: reward `0`; proceed to Bob.
- Alice reveal: reward `{-rewards['reveal_cost']}`; disclose the pet and proceed to Bob.
- Bob bail: reward `{rewards['bob_bail_reward']}`; terminate.
- Bob correct guess: reward `{rewards['correct_reward']}`; terminate.
- Bob wrong guess: reward `{rewards['wrong_reward']}`; terminate.

The reveal cost is charged on Alice's turn regardless of Bob's later action.
Episodes last at most two transitions. The runtime automatically resets after
termination and signals the episode boundary to both policies. Policy memory
is private to each role and reset at episode boundaries.
"""
