# Coding Agent ZSC

Generate independent policies for two-player zero-shot coordination and measure how well they play together. Each AutoAstra session sees **development self-play**. Cross-play uses the frozen selected policies after training.

## Install on the execution host

Use Linux, Python 3.11+, Docker, and credentials for your chosen [AutoAstra provider](https://github.com/shshnkreddy/autoastra). From this checkout:

```sh
python -m venv .venv
source .venv/bin/activate
pip install .
docker build -t autoastra-zsc:cpu runtime/
```

The package depends on the public AutoAstra 0.1.0 release. The Docker build includes the game implementations and evaluator. To use an existing compatible image, set `AUTOASTRA_ZSC_IMAGE` before launching.

## Run the main experiments

The main suite covers six standard Overcooked V2 layouts, three Yōkai settings, and both two-player Hanabi settings. Each environment gets six independent sessions. Seed 0 receives paraphrase p0, through seed 5 with p5. Generation seed and prompt wording vary together.

```sh
# Inspect the plan without model calls or containers.
coding-zsc --paper-models --jobs 6 --output runs/paper --dry-run

# Run the seven historical model configurations, then score within and between models.
coding-zsc --paper-models --jobs 6 --output runs/paper

# Run the same 11 environments with one available model.
coding-zsc --model gpt-6.1-sol --jobs 6 --output runs/sol

# Or run one environment.
coding-zsc --env grounded_coord_simple --jobs 6 --output runs/grounded
```

The full main XP grid has **462 sessions** (7 models × 11 environments × 6 seeds). It requires access to each named model on its original provider. The registry in [experiment.py](coding_agent_zsc/experiment.py) records the model IDs. The archived grid used high reasoning effort. Set `model_reasoning_effort = "high"` in the dedicated Codex profile’s `config.toml` before a paper run. The bundled image sets `CLAUDE_CODE_EFFORT_LEVEL=high` for Claude Code. [Claude Code model configuration](https://code.claude.com/docs/en/model-config) This is a fresh AutoAstra implementation of the protocol; model availability, sampling, harness versions, and hardware can affect results.

| Setting | Default |
| --- | --- |
| Search budget per seed | 2 hours, unlimited tokens and evaluation count |
| Development evaluation | 1,024 episodes per candidate, 600-second timeout |
| Final selection | Best development fitness for ordinary runs. The paper XP grid uses the agent’s nominated revision. Override with `--selection model` or `--selection best-fitness`. |
| Parallelism | One environment and model at a time. `--jobs` controls concurrent seeds (1–6, default 1). |
| Held-out evaluation | 1,024 episodes per ordered pair, common root seed 2000003 |

Use `--suite novel` for Block Grid and Pipeline Cook, `--suite cat-dog` for Cat and Dog, or `--suite all` for all 14 tasks with one model. Use `autoastra preview coding_agent_zsc.hanabi.two_player_h_group_level_1:p0 --output previews/hanabi` to inspect one task.

## Results

Each environment directory contains `seed-0/` through `seed-5/` with final policies and experiment ledgers. **SP** is the mean of the six diagonal returns of the 6 × 6 cross-play matrix. **IXP** is the mean of the 30 directed off-diagonal returns. The runner saves `reports/<environment>/xp.csv`, a heatmap, and a JSON summary under each model’s output directory.

The full grid also writes `between/<environment>/between.csv`. Rows select the seat-0 model and columns the seat-1 model. Each entry averages 30 pairings with unequal seed IDs, including when the two model names are the same. Re-running with the same output reuses completed populations and cached evaluation cells. An incomplete generation population is reported before scoring; use a fresh output directory to regenerate it.

To score existing AutoAstra directories independently:

```sh
coding-zsc-xp runs/sol/grounded_coord_simple/seed-* --output reports/within
coding-zsc-between --env grounded_coord_simple \
  --population sol=runs/sol/grounded_coord_simple \
  --population claude=runs/claude/grounded_coord_simple --output reports/between
```

Task definitions, paraphrases, and scoring code live in [coding_agent_zsc/](coding_agent_zsc/). The container runtime is adapted from the meta_zsc evaluator and includes its [license](runtime/LICENSE) and [third-party notices](runtime/THIRD_PARTY_NOTICES.md).
