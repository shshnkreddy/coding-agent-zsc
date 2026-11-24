from functools import partial
from typing import Dict, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


CAT = 0
DOG = 1

ALICE_SIGNAL_OFF = 0
ALICE_SIGNAL_ON = 1
ALICE_REVEAL = 2
ALICE_BAIL = 3

BOB_BAIL = 0
BOB_GUESS_CAT = 1
BOB_GUESS_DOG = 2
BOB_EXTRA_BAIL = 3


def _validate_action_permutation(name: str, permutation: Sequence[int]) -> Tuple[int, ...]:
    permutation = tuple(int(action) for action in permutation)
    expected = tuple(range(4))
    if len(permutation) != len(expected) or tuple(sorted(permutation)) != expected:
        raise ValueError(
            f"{name} must map the four local action indices to a permutation "
            f"of {expected}, got {permutation}."
        )
    return permutation


@chex.dataclass(frozen=True)
class CatDogState:
    step_count: chex.Array
    pet: chex.Array
    alice_action: chex.Array
    done: chex.Array


class CatDog(MultiAgentEnv):
    """Cat & Dog toy cooperative communication game from Off-Belief Learning.

    Alice privately observes a random pet. She can use a binary cheap-talk
    signal, pay to reveal the pet, or bail. Bob then observes Alice's outcome
    and either bails or guesses the pet.
    """

    def __init__(
        self,
        alice_bail_reward: float = 1.0,
        bob_bail_reward: float = 0.5,
        reveal_cost: float = 5.0,
        correct_reward: float = 10.0,
        wrong_reward: float = -10.0,
        agent_ids: Tuple[str, str] = ("0", "1"),
        alice_action_permutation: Sequence[int] = (0, 1, 2, 3),
        bob_action_permutation: Sequence[int] = (0, 1, 2, 3),
    ):
        assert len(agent_ids) == 2
        super().__init__(num_agents=2)
        self.agents = list(agent_ids)
        self.max_steps = 2
        self.episode_length = 2
        self.alice_bail_reward = float(alice_bail_reward)
        self.bob_bail_reward = float(bob_bail_reward)
        self.reveal_cost = float(reveal_cost)
        self.correct_reward = float(correct_reward)
        self.wrong_reward = float(wrong_reward)
        self.alice_action_permutation = jnp.asarray(
            _validate_action_permutation(
                "alice_action_permutation",
                alice_action_permutation,
            ),
            dtype=jnp.int32,
        )
        self.bob_action_permutation = jnp.asarray(
            _validate_action_permutation(
                "bob_action_permutation",
                bob_action_permutation,
            ),
            dtype=jnp.int32,
        )

        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(10,),
                dtype=jnp.float32,
            )
            for agent in self.agents
        }
        # Use a shared action dimension for compatibility with existing agents.
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}

    @property
    def num_actions(self) -> int:
        return 4

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], CatDogState]:
        pet = jax.random.randint(key, shape=(), minval=0, maxval=2, dtype=jnp.int32)
        state = CatDogState(
            step_count=jnp.array(0, dtype=jnp.int32),
            pet=pet,
            alice_action=jnp.array(-1, dtype=jnp.int32),
            done=jnp.array(False),
        )
        return self.get_obs(state), state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: CatDogState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, chex.Array],
        CatDogState,
        Dict[str, chex.Array],
        Dict[str, chex.Array],
        Dict,
    ]:
        del key
        alice_local_action = jnp.asarray(
            actions[self.agents[0]],
            dtype=jnp.int32,
        )
        bob_local_action = jnp.asarray(
            actions[self.agents[1]],
            dtype=jnp.int32,
        )
        # Each permutation maps the policy's local output index to the
        # role-specific canonical action consumed by the game dynamics.
        alice_action = self.alice_action_permutation[alice_local_action]
        bob_action = self.bob_action_permutation[bob_local_action]
        active = ~state.done
        is_alice_turn = active & (state.step_count == 0)
        is_bob_turn = active & (state.step_count == 1)

        next_alice_action = jnp.where(is_alice_turn, alice_action, state.alice_action)
        alice_bailed = is_alice_turn & (alice_action == ALICE_BAIL)
        reveal = next_alice_action == ALICE_REVEAL

        bob_effective_action = jnp.where(
            bob_action == BOB_EXTRA_BAIL,
            jnp.asarray(BOB_BAIL, dtype=jnp.int32),
            bob_action,
        )
        bob_bailed = is_bob_turn & (bob_effective_action == BOB_BAIL)
        guessed_cat = bob_effective_action == BOB_GUESS_CAT
        guessed_dog = bob_effective_action == BOB_GUESS_DOG
        correct_guess = (
            (state.pet == CAT) & guessed_cat
        ) | ((state.pet == DOG) & guessed_dog)
        guess_reward = jnp.where(correct_guess, self.correct_reward, self.wrong_reward)

        reward = jnp.where(
            is_alice_turn,
            jnp.where(
                alice_bailed,
                self.alice_bail_reward,
                jnp.where(reveal, -self.reveal_cost, 0.0),
            ),
            jnp.where(
                is_bob_turn,
                jnp.where(bob_bailed, self.bob_bail_reward, guess_reward),
                0.0,
            ),
        ).astype(jnp.float32)

        next_step = jnp.where(
            active,
            state.step_count + jnp.array(1, dtype=jnp.int32),
            state.step_count,
        )
        done = state.done | alice_bailed | (active & (next_step >= self.episode_length))
        next_state = CatDogState(
            step_count=next_step,
            pet=state.pet,
            alice_action=next_alice_action,
            done=done,
        )

        obs = self.get_obs(next_state)
        rewards = {agent: reward for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        info = {
            "pet": state.pet,
            "alice_local_action": alice_local_action,
            "alice_action": next_alice_action,
            "bob_local_action": bob_local_action,
            "bob_action": bob_effective_action,
            "alice_bailed": alice_bailed,
            "bob_bailed": bob_bailed,
            "revealed": reveal,
            "correct_guess": is_bob_turn & (~bob_bailed) & correct_guess,
            "step_count": next_step,
            "shaped_reward": {
                agent: jnp.array(0.0, dtype=jnp.float32) for agent in self.agents
            },
        }
        return obs, next_state, rewards, dones, info

    def _obs_for_agent(self, state: CatDogState, agent_idx: int) -> chex.Array:
        is_alice = agent_idx == 0
        is_bob = agent_idx == 1
        alice_turn = state.step_count == 0
        bob_turn = state.step_count == 1
        pet_cat = state.pet == CAT
        pet_dog = state.pet == DOG
        signal_off = state.alice_action == ALICE_SIGNAL_OFF
        signal_on = state.alice_action == ALICE_SIGNAL_ON
        revealed = state.alice_action == ALICE_REVEAL

        alice_pet = jnp.asarray([pet_cat & is_alice, pet_dog & is_alice], dtype=jnp.float32)
        signal = jnp.asarray(
            [signal_off & bob_turn, signal_on & bob_turn],
            dtype=jnp.float32,
        )
        revealed_pet = jnp.asarray(
            [revealed & bob_turn & pet_cat, revealed & bob_turn & pet_dog],
            dtype=jnp.float32,
        )
        return jnp.concatenate(
            [
                jnp.asarray([is_alice, is_bob, alice_turn, bob_turn], dtype=jnp.float32),
                alice_pet,
                signal,
                revealed_pet,
            ]
        )

    def get_obs(self, state: CatDogState) -> Dict[str, chex.Array]:
        return {
            self.agents[0]: self._obs_for_agent(state, 0),
            self.agents[1]: self._obs_for_agent(state, 1),
        }

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: CatDogState) -> Dict[str, chex.Array]:
        alice_turn = state.step_count == 0
        alice_actions = jnp.where(
            alice_turn,
            jnp.ones((4,), dtype=jnp.float32),
            jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        )
        bob_actions = jnp.where(
            alice_turn,
            jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.asarray([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32),
        )
        # Availability is defined in canonical action space above. Reindex it
        # into each role's local policy-output space.
        return {
            self.agents[0]: alice_actions[self.alice_action_permutation],
            self.agents[1]: bob_actions[self.bob_action_permutation],
        }

    @property
    def agent_classes(self) -> dict:
        return {agent: agent for agent in self.agents}

    @property
    def name(self) -> str:
        return "CatDog"

