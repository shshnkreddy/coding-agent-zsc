"""Deterministic Hanabi context assets and opt-in convention documentation."""

from dataclasses import replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from llm_policy.resources import asset_path
import re

from llm_policy.context.guidance import validate_guidance_mode

CONVENTION_ROOT = asset_path("templates/hanabi/conventions")
CONVENTION_ROBUSTNESS = (
    "Seek competent play and robust coordination with other convention-following agents. "
    "Assume shared convention meanings, while allowing differences in aspects such as "
    "implementation, tactical choices, risk thresholds, and tie-breaking wherever the "
    "conventions leave choices open."
)
CONVENTION_XP_INSTRUCTION = (
    "Maximize expected cooperative episode return when your policy is paired with "
    "independently generated policies that follow H-group Level 1 conventions, evaluated "
    "in both player positions. All partners receive the same pinned convention context. "
    + CONVENTION_ROBUSTNESS + " "
    "Self-play return is a useful competence diagnostic, but the optimization target is "
    "cross-play performance with independently generated convention-following partners."
)
CONVENTION_READING = (
    "Read context/conventions/h_group_level_1/README.md, instructions.md, and "
    "runtime_mapping.md in that directory, then the locally copied documentation. "
    "You are paired with independently generated policies that follow H-group "
    "Level 1 conventions. Use the documented meanings as the shared coordination "
    "protocol. All partners receive this bundle. " + CONVENTION_ROBUSTNESS + " "
    "Do not invent additional shared signals or assume that self-play "
    "success establishes cross-play performance with independent implementations."
)


def score_completion_nudge(contract):
    """Render the same performance reference for every context profile."""
    return f"""## Score-completion sanity check

Aim for an average score close to {contract["score_max"]} within the available
budget. Don't stop at a working policy if clear, avoidable losses remain.
Use {contract["score_max"]} as an aspirational reference, recognizing that some
deck configurations make it unattainable, and prioritize reliable coordination
over small score gains.
"""


def resolve_convention_profile(config):
    profile = config.get("llm_policy", {}).get("convention_profile", "none")
    if profile not in ("none", "h_group_level_1"):
        raise ValueError("convention_profile must be none or h_group_level_1")
    if profile != "none" and config.get("env_name", "").lower() != "hanabi":
        raise ValueError("h_group_level_1 requires the Hanabi environment")
    return profile


def convention_partner_wording(text):
    """Qualify the partner population without duplicating qualifiers on re-render."""
    def qualify(match):
        if re.match(r"\s+that\s+follows?\s+H-group\s+Level\s+1\s+conventions\b",
                    text[match.end():]):
            return match[0]
        if match[1] in ("policies", "partners"):
            return "independently generated policies that follow H-group Level 1 conventions"
        return "independently generated partner policy that follows H-group Level 1 conventions"
    return re.sub(
        r"\bindependently generated\s+(partner policy|policies|policy|partners|partner)\b",
        qualify, text)


def convention_objective(objective, profile, num_agents=2):
    if profile == "none":
        return objective
    if objective.evaluator == "deferred_population_cross_play":
        return replace(objective, instruction=(CONVENTION_XP_INSTRUCTION if num_agents == 2 else
                           "Maximize expected cooperative episode return when paired with "
                           "independently generated policies that follow "
                           "H-group Level 1 conventions. All receive the same pinned context. "
                           + CONVENTION_ROBUSTNESS),
                       selection_rule=convention_partner_wording(objective.selection_rule))
    premise = (
        ("Both players execute your policy. Follow the supplied H-group Level 1 " if num_agents == 2 else
         f"All {num_agents} players execute your policy. Follow the supplied H-group Level 1 ") +
        "conventions as the shared coordination protocol."
    )
    instruction = objective.instruction
    if premise not in instruction:
        instruction += " " + premise
    return replace(objective, instruction=instruction)


def convention_metadata(profile):
    if profile == "none":
        return {"profile": "none"}
    path = CONVENTION_ROOT / profile / "source_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"profile": profile, "source_revision": manifest["source_revision"],
            "source_repository": manifest["source_repository"],
            "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "context_path": f"context/conventions/{profile}"}


def capabilities(num_agents=2):
    return {"api_version": 1, "adapter": "hanabi",
            "information_boundary": {
                "available": ["own native observation and legal-action mask", "private history/memory", "reset flag", "private PRNG"],
                "unavailable": ["unrevealed own cards", "deck order", "partner private observation/memory", "runtime environment state", "runtime host I/O"]},
            "facts": ["all visible teammate hands", "public board/counts", "public last action and hint matches"],
            "belief": "exact literal own-hand marginals and joint samples; conditional partner own-hand beliefs given hypothetical own hands; no convention/intent model",
            "actions": ["play", "discard", "hint_color", "hint_rank", "noop (inactive only)"],
            "memory": f"private literal constraints tracking all {num_agents} players' publicly hinted cards"}


def policy_interface(contract, convention_profile="none"):
    n = contract["num_agents"]
    if n > 2:
        functions = ("init_agent() and agent_act(obs, reset, memory, rng)" if contract["policy_structure"] == "shared_local" else
                     ", ".join(f"init_agent_{i}() and agent_{i}_act(obs, reset, memory, rng)" for i in range(n)))
        return (f"Keep policy_structure='{contract['policy_structure']}' and implement {functions}. "
                f"All {n} seats have private observations, memory and PRNG keys and process every transition. "
                "Self-play uses this module in every seat. Directed XP cell [A,B] uses A in seat 0 "
                f"and private instances of B in seats 1..{n-1}; there is no seat rotation. "
                + ("All partners follow the supplied H-group Level 1 conventions." if convention_profile != "none"
                   else "Partners need not use your code or conventions."))
    partner = (
        "Partners are independently generated policies that follow H-group Level 1 "
        "conventions and receive the same pinned context. Differences may include aspects "
        "such as implementation, tactical choices, risk thresholds, and tie-breaking "
        "wherever the conventions leave choices open."
        if convention_profile != "none" else "Partners need not use your code or conventions."
    )
    if contract["policy_structure"] == "shared_local":
        cross_play = (
            "independent generation runs in both seat assignments; partners need "
            "not use your code or conventions." if convention_profile == "none" else
            "independent generation runs in both seat assignments. " + partner
        )
        return ("Keep the starter's metadata, including policy_structure='shared_local'. "
                "Implement only init_agent() and agent_act(obs, reset, memory, rng); "
                "do not define role-specific entry points. Both players run this same local "
                "code with separate observations, memory and PRNG keys. Self-play pairs "
                "two private instances of this policy. Cross-play pairs policies from " + cross_play)
    return ("Keep the starter's metadata, including policy_structure='separate_roles'. "
            "Implement init_agent_0(), init_agent_1(), agent_0_act(obs, reset, memory, rng) "
            "and agent_1_act(obs, reset, memory, rng). Each role has private memory and "
            "PRNG keys. Self-play pairs this module's two roles; cross-play pairs roles "
            "from independent generation runs in both seat assignments."
            + (" " + partner if convention_profile != "none" else ""))


def write_sources(environment, environment_dir):
    directory = Path(environment_dir)
    source_dir = directory / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for module_name in ("jaxmarl.environments.hanabi.hanabi", "jaxmarl.environments.hanabi.hanabi_game",
                        "jaxmarl.environments.multi_agent_env", "llm_policy.environments.hanabi.adapter",
                        "llm_policy.environments.hanabi.layout"):
        source = inspect.getsource(importlib.import_module(module_name))
        name = module_name.replace(".", "_") + ".py"
        (source_dir / name).write_text(source, encoding="utf-8")
        sources[name] = hashlib.sha256(source.encode()).hexdigest()
    (directory / "README.md").write_text(
        "# Hanabi environment reference\n\nInstalled JaxMARL source is authoritative for native encodings and dynamics. "
        "The repository adapter supplies private observations, evaluation logging, illegal-action rejection, "
        "and the configured final-life termination correction. No pretrained policy is included. "
        "See ../task_spec.md for rules and scoring, config.yaml for configuration, "
        "and ../policy_api.md for the policy interface.\n", encoding="utf-8")
    (directory / "source_manifest.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    return sources


def briefing(environment, runtime_contract=None):
    if runtime_contract is None:
        from llm_policy.environments.hanabi.adapter import HanabiAdapter
        runtime_contract = HanabiAdapter(environment).runtime_contract()
    if runtime_contract["num_agents"] > 2:
        from llm_policy.context.hanabi_description import rules
        return rules(runtime_contract)
    colors = runtime_contract["num_colors"]
    ranks = runtime_contract["num_ranks"]
    maximum = runtime_contract["score_max"]
    minimum = runtime_contract["score_min"]
    fixed = environment.config.get("llm_policy", {}).get("fix_final_life_termination", True)
    termination = ("The episode ends immediately when the last life is lost."
                   if fixed else "Legacy mode retains JaxMARL 0.1.0's one-turn-late final-life termination.")
    return f"""# Two-player Hanabi rules

Five colors, five ranks, five cards per player; per-color rank multiplicities
are 3,2,2,2,1. There are initially eight information tokens and three lives.
Player 0 takes the first turn. Each player sees the partner's cards, not their
own identities or deck order. Card positions are observable; identities become
public when played or discarded. Play/discard removes a slot, shifts later
cards left, and appends a fresh card if the deck was nonempty.

A legal hint costs one information token and identifies ALL cards of the named
color/rank in the partner's hand, including negative information for unmatched
slots. A discard restores a token and is illegal when tokens are already full.
Playing the next rank advances that color's firework by one; a failed play
discards the card and loses a life. Completing a color can restore a token.
These are game rules, not recommendations for choosing an action or interpreting
the partner's intent.

## Score range: {minimum}–{maximum}

Each successful play adds one point. Completing all {ranks} ranks in all
{colors} colors gives {colors} × {ranks} = {maximum} points, the theoretical ceiling
under the configured standard scoring rules. Losing all lives gives a final
return of zero. This describes the scoring rules, not a requirement to reach
the ceiling before finishing policy search.

## Episode termination

{termination}
Completing all fireworks ends the game. Drawing the last card starts the
native final round; there are two further player turns. No artificial game
horizon is added. Evaluation.max_steps is a safety cap, and incomplete episodes
are rejected, not assigned a normal score.

Native 658-feature observations have blocks: visible hands/missing flags [0:127],
board [127:203], discards [203:253], last action [253:308], and V0/hint features
[308:658]. There are 21 action IDs: discard 0..4, play 5..9, color hints 10..14,
rank hints 15..19, inactive no-op 20. Primitive color/rank/slot indices are 0..4;
human-readable ranks are 1..5. Public last-action player indices are relative
to the observer. The adapter adds only that player's native legal-action mask.

The native V0 features are approximate and are retained unchanged for inspection.
They omit visible-hand subtraction and joint card-count constraints. Positive
hint flags alone omit negative information. Use the documented literal knowledge
and inference APIs when those semantics are needed; no helper infers a convention.
"""


def task_dossier(environment, objective, runtime_env, policy_guidance, guidance_mode, meta_update_id):
    contract = runtime_env.runtime_contract()
    profile = resolve_convention_profile(environment.config)
    objective = convention_objective(objective.for_policy_structure(contract["policy_structure"]), profile, contract["num_agents"]).for_runtime_contract(contract)
    guided = validate_guidance_mode(guidance_mode, policy_guidance, meta_update_id)
    record = {"schema_version": 1, "objective": objective.to_record(),
              "environment": {"name": environment.config.get("name", "hanabi_2p"), "env_name": "hanabi",
                              "env_kwargs": environment.env_kwargs, "config_sha256": hashlib.sha256(environment.config_path.read_bytes()).hexdigest()},
              "runtime_contract": contract, "capabilities": capabilities(contract["num_agents"]),
              "meta_guidance": {"mode": guidance_mode, "meta_update_id": meta_update_id,
                                "present": guided, "sha256": hashlib.sha256(policy_guidance.encode()).hexdigest()}}
    guidance = ""
    if guided:
        guidance = "\n## Policy guidance\n\nRead context/policy_guidance.md and follow it subject to the objective, rules, and information boundary.\n"
    elif guidance_mode == "meta_baseline":
        guidance = "\n## Meta-learning baseline\n\nGeneration 0; no policy guidance is supplied.\n"
    if profile != "none":
        record["conventions"] = convention_metadata(profile)
        guidance += "\n## Shared convention context\n\n" + CONVENTION_READING + "\n"
    text = f"""# Hanabi policy-search task

Design a decentralized policy module using the configured interface below. During development you may
inspect the supplied rules and environment source and evaluate candidate code.
During execution each role sees only its own observation, legal mask, reset,
private memory and PRNG. Never expose hidden cards, partner memory or deck state.

## Objective: {objective.name}

{objective.instruction}

Selection rule: {objective.selection_rule}

{objective.final_selection_instruction}
{guidance}
## Exact runtime contract

```json
{json.dumps(contract, indent=2, sort_keys=True)}
```

{briefing(environment, contract)}

{score_completion_nudge(contract)}

## Interface and primitives

Keep POLICY_API_VERSION=1, adapter='hanabi', adapter_version=1 and the starter's
dimension metadata. {policy_interface(contract, profile)} Return a scalar integer action,
same-structure fixed-shape memory and PolicyTrace. All {contract["num_agents"]} player instances run at every
global transition; only the active player acts, but all can update memory.

context/policy_api.md documents the supported Hanabi facts, legal actions,
literal knowledge, exact own-hand beliefs and joint samples, conditional partner
beliefs given hypothetical own hands, memory and trace helpers. Their implementation
is context/hanabi_primitives.py. context/rollout_api.py is the evaluator facade.
context/environment/config.yaml and context/environment/source/ provide the exact
configuration and implementation; the rules and scoring are described above.
No helper prescribes a signaling convention, risk threshold or tactical policy.
"""
    if profile != "none":
        text = convention_partner_wording(text)
    return record, text


def provided_starter(contract):
    """Render the optional primitive-backed starter for the exact task."""
    n, h = contract["num_agents"], contract["hand_size"]
    metadata = {"name": "hanabi_candidate", "adapter": "hanabi", "adapter_version": 1,
                **{k: contract[k] for k in ("policy_structure", "num_agents", "num_colors", "num_ranks", "hand_size", "observation_size", "num_actions")}}
    text = ('"""Neutral legal-random Hanabi starter; every seat has private memory."""\n'
            'from llm_policy import hanabi_primitives as p\n\n'
            'POLICY_API_VERSION = 1\nPOLICY_METADATA = ' + repr(metadata) + '\n')
    for suffix in (("",) if contract["policy_structure"] == "shared_local" else tuple(f"_{i}" for i in range(n))):
        text += f"""

def init_agent{suffix}():
    return p.init_knowledge(num_agents={n}, hand_size={h})


def agent{suffix}_act(obs, reset, memory, rng):
    facts, memory = p.observe(obs, reset, memory)
    action = p.sample_legal(facts.legal_actions, rng)
    return action, memory, p.make_trace(action, num_agents={n}, hand_size={h})
"""
    return text


def provided_api(contract, profile="none"):
    n, h = contract["num_agents"], contract["hand_size"]
    return f"""# Hanabi policy API

Import `from llm_policy import hanabi_primitives as p`. All helpers are pure JAX.
No runtime I/O, subprocesses, simulator imports, callbacks or mutable globals.

{policy_interface(contract, profile)}

Input: HanabiObservation(features[{contract['observation_size']}], legal_actions[{contract['num_actions']}]),
reset flag, private memory and PRNG. Return scalar integer action, same-shape/dtype
memory and PolicyTrace. Illegal active actions invalidate evaluation.

## Observations and private literal memory

`p.init_knowledge(num_agents={n}, hand_size={h})` initializes literal masks
possible[{n},{h},5,5], previous_deck_remaining and episode_step.
`facts, knowledge = p.observe(obs, reset, knowledge)` decodes the observation and
updates public-event constraints once per transition, including inactive turns.
`p.decode_observation(obs)` and `p.update_knowledge(facts, reset, knowledge)` may
also be used separately. Reset removes old evidence; do not process events twice.

Facts expose other_hands[{n-1},{h},5,5], partner_hand[{h},5,5] (relative seat 1),
occupied[{n},{h}], fireworks[5], discards[5,5], deck_remaining, info_tokens,
life_tokens, is_my_turn and legal_actions[{contract['num_actions']}]. All player
axes are relative to self=0. Other hands follow relative seats 1..{n-1}.
Public events expose has_last_action, last_actor, last_kind, last_target,
last_color, last_rank, last_matches[{h}], last_slot, last_card[5,5],
last_play_scored and last_play_added_token. Read only applicable event fields.
upstream_v0[{n},{h},5,5], positive_colors[{n},{h},5] and positive_ranks[{n},{h},5]
are unchanged native features, not exact posteriors or intent inference.

## Actions

Pass static `num_agents={n}, hand_size={h}` to `hint_color(color, target=t, ...)`,
`hint_rank(rank, target=t, ...)`, `noop(...)`, `decode_action(action, ...)`,
`action_target(action, ...)` and `make_trace(action, ..., num_agents=..., hand_size=...)`.
Targets are relative 1..{n-1}; colors/ranks are 0..4. Invalid hint targets or values
return -1. `play(slot, hand_size={h})` and `discard(slot)` use native slots.
Dimension arguments default to the legacy two-player game; always pass this task's
values. `decode_action` returns (kind, detail); `action_target` returns the hint
recipient or -1 for non-hints. Kinds: PLAY, DISCARD, HINT_COLOR, HINT_RANK, NOOP.
`legal_actions(facts)`, `is_legal(mask, action)`, `masked_argmax(values, mask)` and
`sample_legal(mask, rng)` operate on the configured mask. The latter two return
-1 if no action is legal. `make_trace` also accepts phase_id and decision_id.

## Literal constraints and exact beliefs (optional)

`apply_hint_constraints(possible[H,5,5], matches[H], kind, value)` applies both
positive and negative evidence. `hint_matches(visible_hand, kind, value)` and
`literal_hint_effect(possible, visible_hand, kind, value)` describe literal clues.
`remap_after_card_removal(per_slot, slot, replacement)` shifts arbitrary per-card
memory and appends replacement. No removed card's evidence transfers to a draw.

`public_remaining_counts(facts)` excludes plays and discards. `unseen_counts(facts)`
also subtracts ALL visible hands, leaving own hidden hand plus undealt deck.
`literal_hand_marginals(possible[H,5,5], counts[5,5], occupied[H])` returns exact
without-replacement probabilities and feasible masks [H,5,5], valid and normalizer.
`sample_literal_hand(possible, counts, occupied, rng)` draws a joint one-hot hand
[H,5,5] with valid and log_probability. Invalid evidence yields zero arrays and
valid=False (sample log_probability=-inf), never a V0 or uniform fallback.
`own_hand_belief(facts, knowledge)` and `sample_own_hand(facts, knowledge, rng)`
use own evidence and all visible hands.

`partner_hand_belief(facts, knowledge, hypothetical_own_hand, target=t)` computes
the selected teammate's conditional literal belief: subtract the hypothetical
own hand and every visible hand EXCEPT target t from public remaining counts.
The own-hand hypothesis must respect own literal evidence and available counts.
Target defaults to relative seat 1. Never supply hidden runtime truth.
`build_belief_context(facts, knowledge, rng, num_samples=K, target=t)` bundles own
belief, K joint samples and K conditional target beliefs, with valid[K]. K is a
positive static integer; samples are fresh hypotheses, not persistent particles.

`card_properties(facts)` returns identity masks for playable, already_played,
unreachable and critical cards. `card_property_probabilities(belief, facts)`
returns per-slot probabilities and validity. These helpers make no tactical choice.
`select_pytree(condition, yes, no)` selects matching fixed-shape PyTrees.
The optional `init_partner_belief_model_template()` and
`update_partner_belief_model_template(facts, reset, belief_context, model_state)`
are literal-only scaffolds, not learned or convention-conditioned partner models.

No exact inference runs unless the policy calls it. Keep literal constraints
separate from convention inferences; use only public history and private memory.
"""


def adapt_convention_bundle(directory, contract):
    """Adapt only repository-authored guides; copied upstream files stay pinned."""
    directory = Path(directory)
    n, h = contract["num_agents"], contract["hand_size"]
    path = directory / "README.md"
    text = path.read_text().replace("which examples apply to two\nplayers", f"which examples apply to {n}\nplayers")
    path.write_text(text)
    path = directory / "instructions.md"
    text = path.read_text().replace("in both seats", "with independently generated partners")
    text = text.replace("The runtime guide explains why\n  ordinary three-player finesse examples do not become two-player self-finesses.",
                        "Ordinary Level 1 finesses can apply when the distinct giver, reactor and recipient\n  and required turn order exist. Do not import higher-level conventions.")
    path.write_text(text)
    path = directory / "runtime_mapping.md"
    text = path.read_text()
    suffix = text[text.index("## Reading the original YAML diagrams"):]
    path.write_text(f"""# Mapping H-group documentation to this task

This adapts the pinned Level 1 documentation to standard {n}-player Hanabi,
with {h} cards per full hand. The task's game rules and legal mask are authoritative.

## Hand order and slots

H-group displays newest cards on the left, oldest on the right. Native arrays
shift later cards left on removal and append draws at index {h-1}. Read the
**occupied native indices in descending order** as the H-group display order.
For a full hand, H-group slot s maps to native index {h}-s. After hands shrink,
use occupied indices rather than the full-hand formula.

| H-group description | Native interpretation |
|---|---|
| Leftmost eligible | Highest occupied eligible index |
| Rightmost eligible | Lowest occupied eligible index |
| Chop | Lowest occupied unclued index |
| Finesse position | Highest occupied unclued index |
| Leftmost newly touched / re-touched | Highest index in the relevant occupied set |
| Leftmost compatible prompted | Highest occupied compatible clued index |

Determine focus and chop from the pre-clue state. Follow Level 1's treatment
of already finessed cards. Empty slots are never focus, chop, or finesse positions;
an all-clued hand has no ordinary chop. Removal shifts all later notes left;
append fresh notes for a draw and empty notes when the deck is empty. Never
transfer notes from a removed card to a new draw.

## Ranks, suits, players, and actions

Human rank r in 1..5 means native r-1. Diagram suits map r=0, y=1, g=2, b=3,
p=4. These labels introduce no variant. Alice/Bob/Cathy/etc. are example turn
positions, not special roles. Absolute seat 0 moves first; turns advance cyclically.
From each observer, self=0 and subsequent seats are 1..{n-1}. Last actor and hint
recipient fields are observer-relative. A hint targets one relative seat 1..{n-1};
use the task's action encoding. Enabling conventions does not reverse runtime arrays.

## Applicable Level 1 special moves

The distinct clue giver, intermediate reactor and clue recipient in ordinary
finesse and prompt examples can exist here. Apply those examples only when their
public evidence, distinct participants and actual cyclic turn order hold. Track
connections across intervening turns and the relevant players' card movements.
Self-prompts explicitly described by Level 1 remain available under their stated
premises. Do not replace absent evidence with a higher-level self-finesse, reverse
finesse, or invented signal. More players do not authorize higher-level conventions.

The source diagrams can use a different player count or hand size. Preserve their
logical premises; do not invent a missing card slot or participant. Delayed plays,
saves, focus and early-game rules apply when their documented premises hold.
Only observations and public history establish runtime evidence; diagram-author
knowledge does not become an extra observation.

""" + suffix)
