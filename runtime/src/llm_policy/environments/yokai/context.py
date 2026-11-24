"""Deterministic Yokai rules, interfaces and source provenance for policy search."""

import hashlib
import importlib
import inspect
import json
import shutil
from pathlib import Path

from llm_policy.context.guidance import validate_guidance_mode


def capabilities():
    return dict(api_version=1, adapter="yokai", information_boundary=dict(
        available=["own native graph observation", "legal-action mask", "private memory", "reset", "private PRNG"],
        unavailable=["unseen card colors", "unrevealed hint colors", "partner private knowledge/memory", "runtime environment state", "runtime host I/O"]),
        facts=["stable card IDs and positions", "own revealed colors", "public hint colors and placements", "locks", "partner observation indicators", "active player and substep"],
        actions=["observe", "move", "reveal hint", "place hint", "end game", "inactive no-op"],
        memory="Private colors observed by this player, indexed by stable card ID; hints do not create hard color constraints.")


def policy_interface(contract):
    return ("Keep policy_structure='shared_local'. Implement only init_agent() and "
            "agent_act(obs, reset, memory, rng), returning scalar integer action, "
            "same-structure fixed-shape memory, and PolicyTrace. Both seats run this "
            "local code with separate observations, memory and PRNG keys. Self-play "
            "pairs two private instances. Cross-play pairs independently generated "
            "policies in both seat assignments. Partners may use different implementations "
            "and conventions. Both seats process every transition; only the active seat acts.")


def write_sources(environment, environment_dir):
    from llm_policy.environments.yokai.adapter import SOURCE_MODULES
    directory = Path(environment_dir)
    source_dir = directory / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for module_name in SOURCE_MODULES:
        source = inspect.getsource(importlib.import_module(module_name))
        name = module_name.replace(".", "_") + ".py"
        (source_dir / name).write_text(source, encoding="utf-8")
        sources[name] = hashlib.sha256(source.encode()).hexdigest()
    from llm_policy.environments.yokai.native import YokaiGrid9
    upstream_dir = Path(inspect.getfile(YokaiGrid9)).parent
    for name in ("LICENSE", "UPSTREAM.json"):
        shutil.copy2(upstream_dir / name, source_dir / name)
        sources[name] = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
    (directory / "source_manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    (directory / "README.md").write_text(
        "# Yokai environment reference\n\nPinned upstream source supplies native rules and encodings. "
        "UPSTREAM.json records the revision and compatibility edits. The adapter supplies private "
        "observations, terminal accounting, illegal-action rejection and the configured scoring mode. "
        "Read ../task_spec.md, ../policy_api.md and config.yaml for the exact task.\n")
    return sources


def briefing(environment, runtime_contract=None):
    if runtime_contract is None:
        from llm_policy.environments.yokai.adapter import YokaiAdapter
        runtime_contract = YokaiAdapter(environment).runtime_contract()
    c = runtime_contract
    memory = ("Perfect memory: on your own turns the observation exposes all colors you personally "
              "observed earlier, including cards that moved. It never exposes a partner's private "
              "colors. Colors are masked while you are inactive; your own private memory can persist."
              if c["past_colors_revealed"] else
              "Standard memory: the observation exposes only cards currently revealed during your "
              "own turn. Retaining prior observations requires your own private policy memory.")
    failure = ("Failed games return zero." if c["reward_mode"] == "upstream_sparse" else
               f"Failed games return -early_end - ({c['num_colors']} - correctly_clustered_colors) - incorrect_hints, "
               "where early_end is 1 for the explicit end-game action and otherwise 0.")
    n, colors, grid, hints = (c[k] for k in ("deck_size", "num_colors", "grid_size", "num_hints"))
    side = 3 if n == 9 else 4
    move_start = 1 + n
    reveal_start = move_start + n * grid * grid
    place_start = reveal_start + hints
    end_action = c["num_actions"] - 1
    hint_description = ("one single-color hint and three two-color hints" if n == 9 else
                        "two single-color hints, three two-color hints and two three-color hints")
    return f"""# Two-player {n}-card Yokai

{n} face-down cards have {colors} colors, {side} copies each. They start in a {side}x{side}
block in a {grid}x{grid} grid. Card IDs 0..{n-1} remain stable when cards move. {hints} hints are
available: {hint_description}. The native reset
determines the starting player; the observation identifies the current player.

Each turn has four substeps: observe a card (or end the game), observe a distinct
second card, move an unlocked card, then reveal an unused hint or place a revealed
hint on an unlocked card. Moves target empty cells and must preserve connectivity
of the entire board through orthogonal adjacency. The native mask is authoritative.
Moving is not restricted to a card just observed. Placing a hint locks that card
against further observation and movement. Hints may be false: a placement is not
proof that the underlying card has one of its listed colors.

## Memory: {c['memory_mode']}

{memory} Public indicators of which cards the partner saw also follow the native
memory flag: full history in perfect-memory mode, recent indicators in standard mode.

## Scoring: {c['reward_mode']}

Rewards are zero until termination. A win requires each color to form a connected
cluster. Winning score = 5 * unrevealed_hints + 2 * revealed_unplaced_hints
+ correctly_placed_hints - incorrectly_placed_hints. {failure}
No intermediate shaping rewards are used. The configured return range is
{c['score_min']} to {c['score_max']}; the maximum requires an immediate successful
ending and is not attainable from every initial board.

The episode ends on an explicit end-game action or when all hints are placed.
The longest legal game has {hints*8} substeps. The evaluator safety cap is {c['max_steps']};
incomplete episodes and illegal active actions invalidate evaluation.

## Observation and action contract

Graph node rows 0..{n-1} are cards and {n}..{n+hints-1} are hints. Features have columns:
0:{colors} own visible colors; {colors}:{2*colors} public hint colors; {2*colors}:{2*colors+2} position; {2*colors+2} locked/placed;
{2*colors+3} partner-seen indicator; {2*colors+4} current-player flag; {2*colors+5} substep; {2*colors+6} stable ID
(cards 0..{n-1}, hints -1..-{hints}). Zero color vectors mean unknown, not an additional color.
Adjacency includes card-neighbor edges, placed-hint edges and self-loops.

Actions: inactive no-op 0; observe card i = 1+i; move card i to (x,y) =
{move_start}+i*{grid*grid}+x*{grid}+y; reveal hint h = {reveal_start}+h; place hint h at (x,y) =
{place_start}+h*{grid*grid}+x*{grid}+y; end game = {end_action}. Only masked-legal actions may be selected.
The inactive seat is forced to no-op; an illegal active action is rejected.
"""


def task_dossier(environment, objective, runtime_env, policy_guidance, guidance_mode, meta_update_id):
    contract = runtime_env.runtime_contract()
    objective = objective.for_policy_structure(contract["policy_structure"])
    guided = validate_guidance_mode(guidance_mode, policy_guidance, meta_update_id)
    record = dict(schema_version=1, objective=objective.to_record(),
        environment=dict(name=environment.config.get("name", "yokai_2p_9c"), env_name=environment.env_name,
            env_kwargs=environment.env_kwargs, config_sha256=hashlib.sha256(environment.config_path.read_bytes()).hexdigest()),
        runtime_contract=contract, capabilities=capabilities(),
        meta_guidance=dict(mode=guidance_mode, meta_update_id=meta_update_id, present=guided,
            sha256=hashlib.sha256(policy_guidance.encode()).hexdigest()))
    guidance = ("Read context/policy_guidance.md subject to the objective, rules and information boundary."
                if guided else "No policy guidance is supplied.")
    primitive_module = "yokai16_primitives" if contract["deck_size"] == 16 else "yokai_primitives"
    text = f"""# Yokai policy-search task

Design a decentralized policy for the exact environment below. During development
you may inspect the supplied rules and source and evaluate candidate code. During
execution each seat receives only its own observation, legal mask, reset, private
memory and PRNG. Never expose unseen colors, partner knowledge or environment state.

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

## Interface and primitives

Keep POLICY_API_VERSION=1, adapter='yokai', adapter_version=1 and the starter's
dimension metadata. {policy_interface(contract)}

context/policy_api.md documents observation decoding, action encoders, legality,
private observed-card memory, graph connectivity, optional joint card beliefs,
hint/lock events, hypothetical moves and traces. The implementation is
context/{primitive_module}.py. context/rollout_api.py supplies evaluation functions.
context/environment/ contains the exact configuration and source provenance.
Helpers do not prescribe a signaling convention, risk threshold or hint interpretation.
"""
    return record, text
