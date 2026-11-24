<h1 align="center">Coding Agent ZSC</h1>

<p align="center"><strong>Can coding agents design policies that coordinate with partners they have never played with?</strong></p>

<p align="center">
  <a href="#set-up-with-an-agent">Agent setup</a> ·
  <a href="#install-on-the-execution-host">Quickstart</a> ·
  <a href="#run-the-main-experiments">Paper experiments</a> ·
  <a href="#results">Results</a> ·
  <a href="coding_agent_zsc/">Tasks</a>
</p>

Each coding agent develops a two-player policy using only **self-play** feedback. After development, we freeze the policies and measure **cross-play** by assigning independently generated policies to opposite seats. The main suite covers six Overcooked V2 layouts, three Yōkai settings, and two Hanabi settings.

## Set up with an agent

Paste this into your coding agent from the repository checkout:

```text
Set up this repository on my Linux Docker host.
```

## Install on the execution host

Use Linux with Docker and Python 3.11+. From this checkout:

```sh
python -m venv .venv
source .venv/bin/activate
pip install .
docker build -t autoastra-zsc:cpu runtime/
```

This installs AutoAstra 0.1.0 and builds the game runtime. To inspect what the agent sees, generate a task preview:

```sh
autoastra preview coding_agent_zsc.hanabi.two_player_h_group_level_1:p0 --output previews/hanabi
```

## Run the main experiments

The paper grid has **462 sessions**: seven models × eleven environments × six seeds. Each seed receives its matching prompt variant (`0/p0` through `5/p5`), so seed and wording vary together. The first command prints the plan. The second launches generation and held-out scoring.

```sh
coding-zsc --paper-models --jobs 6 --output runs/paper --dry-run
coding-zsc --paper-models --jobs 6 --output runs/paper
```

The full grid requires access to the seven historical models. For one environment with your configured provider, run `coding-zsc --env grounded_coord_simple --jobs 6 --output runs/grounded`. [Model registry](coding_agent_zsc/experiment.py) · [AutoAstra provider setup](https://github.com/shshnkreddy/autoastra#run-the-demo)

<details>
<summary><strong>Experiment settings and additional suites</strong></summary>

| Setting | Default |
| --- | --- |
| Development | Two hours per seed, unlimited tokens and evaluations. Each candidate gets 1,024 self-play episodes and a 600-second evaluation timeout. |
| Selection | The paper grid uses the agent’s nominated revision. Ordinary runs use the best development score. Override with `--selection model` or `--selection best-fitness`. |
| Parallelism | One model and environment at a time. `--jobs` controls concurrent seeds (1–6, default 1). |
| Held-out scoring | 1,024 episodes per ordered policy pair, with common root seed `2000003`. |

The archived grid used high reasoning effort. Set `model_reasoning_effort = "high"` in the dedicated Codex profile’s `config.toml`. The bundled image sets `CLAUDE_CODE_EFFORT_LEVEL=high`. Fresh runs can vary with model availability, sampling, harness versions, and hardware.

To run all eleven main environments with one model, use `coding-zsc --model gpt-6.1-sol --jobs 6 --output runs/sol`. Use `--suite novel` for Block Grid and Pipeline Cook, `--suite cat-dog` for Cat and Dog, or `--suite all` for all 14 tasks. Set `AUTOASTRA_ZSC_IMAGE` to use an existing compatible image.

</details>

## Results

Each model’s `reports/<environment>/xp.csv` is a 6 × 6 matrix. **SP** averages the six diagonal scores, where one policy plays both seats. **IXP** averages the 30 off-diagonal scores, where the seats use policies from different seeds. A heatmap and JSON summary sit beside the CSV. The final policies and experiment ledgers live under each environment’s `seed-0/` through `seed-5/` directories.

For the full grid, `between/<environment>/between.csv` compares models. Rows identify the seat-0 model and columns the seat-1 model. Each entry averages 30 pairings with unequal seed IDs.

<details>
<summary><strong>Score existing runs</strong></summary>

```sh
coding-zsc-xp runs/sol/grounded_coord_simple/seed-* --output reports/within
coding-zsc-between --env grounded_coord_simple \
  --population sol=runs/sol/grounded_coord_simple \
  --population claude=runs/claude/grounded_coord_simple --output reports/between
```

Reusing an output directory preserves completed populations and cached evaluation cells. If generation stopped with an incomplete population, use a fresh output directory to regenerate it.

</details>

The Docker runtime adapts the meta_zsc evaluator. [License](runtime/LICENSE) · [Third-party notices](runtime/THIRD_PARTY_NOTICES.md)

## Citation
