<h1 align="center">Coding Agents Are Zero-Shot Coordinators</h1>

<p align="center"><strong>Can coding agents design policies that coordinate with partners they have never played with?</strong></p>

<p align="center">
  <a href="#astra--fable-on-overcooked-v2">Episodes</a> ·
  <a href="#set-up-with-an-agent">Agent setup</a> ·
  <a href="#install-on-the-execution-host">Quickstart</a> ·
  <a href="#run-one-environment">Run one environment</a> ·
  <a href="#results">Results</a> ·
  <a href="coding_agent_zsc/">Tasks</a>
</p>

Each coding agent develops a two-player policy using only **self-play** feedback. We measure **cross-play** by pairing independently developed policies. The main suite covers six Overcooked V2 layouts, three Yōkai settings, and two Hanabi settings.

<a name="astra--fable-on-overcooked-v2"></a>

## Astra 🔴 × Fable 🔵 on Overcooked V2

<table>
  <tr>
    <td align="center" valign="top">
      <strong>Grounded Coordination Simple</strong><br>
      <img src="docs/episodes/grounded_coord_simple.gif" alt="Astra and Fable playing Grounded Coordination Simple" width="190">
    </td>
    <td align="center" valign="top">
      <strong>Grounded Coordination Ring</strong><br>
      <img src="docs/episodes/grounded_coord_ring.gif" alt="Astra and Fable playing Grounded Coordination Ring" width="190">
    </td>
    <td align="center" valign="top">
      <strong>Test-time Simple</strong><br>
      <img src="docs/episodes/test_time_simple.gif" alt="Astra and Fable playing Test-time Simple" width="190">
    </td>
    <td align="center" valign="top">
      <strong>Test-time Wide</strong><br>
      <img src="docs/episodes/test_time_wide.gif" alt="Astra and Fable playing Test-time Wide" width="190">
    </td>
  </tr>
  <tr>
    <td align="center" valign="top">
      <strong>Demo Cook Simple</strong><br>
      <img src="docs/episodes/demo_cook_simple.gif" alt="Astra and Fable playing Demo Cook Simple" width="190">
    </td>
    <td align="center" valign="top">
      <strong>Demo Cook Wide</strong><br>
      <img src="docs/episodes/demo_cook_wide.gif" alt="Astra and Fable playing Demo Cook Wide" width="190">
    </td>
    <td align="center" valign="top">
      <strong>Block Grid</strong><br>
      <img src="docs/episodes/block_grid_2p.gif" alt="Astra and Fable playing Block Grid" width="190">
    </td>
    <td align="center" valign="top">
      <strong>Pipeline Cook</strong><br>
      <img src="docs/episodes/pipeline_cook_2p.gif" alt="Astra and Fable playing Pipeline Cook" width="190">
    </td>
  </tr>
</table>

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

## Set up model access

On the Linux host that will run the experiments, install the agent CLI and configure its credentials. For a single-model run through OpenRouter, export `OPENROUTER_API_KEY` in the launch shell and pass `--provider openrouter --model PROVIDER/MODEL` to `coding-zsc`.

For sign-in instructions and the key each provider uses, expand **Configure model access** in the [AutoAstra quickstart](https://github.com/shshnkreddy/autoastra#run-the-demo).

## Run one environment

With Codex configured, this command runs six independent seeds in parallel, then evaluates the six selected policies in all 36 ordered pairings.

```sh
coding-zsc --model gpt-6.1-sol --env cat_dog --jobs 6 --output runs/cat-dog
```

Open `runs/cat-dog/reports/cat_dog/summary.json` for the self-play (SP) and cross-play (IXP) scores, or `xp.csv` in the same directory for the full matrix.

<details>
<summary><strong>Experiment settings</strong></summary>

| Setting | Default |
| --- | --- |
| Development | Two hours per seed, unlimited tokens and evaluations. Each candidate gets 1,024 self-play episodes and a 600-second evaluation timeout. |
| Selection | Uses the best development score. Pass `--selection model` to use the agent’s nominated revision. |
| Parallelism | One model and environment at a time. `--jobs` controls concurrent seeds (1–6, default 1). |
| Held-out scoring | 1,024 episodes per ordered policy pair, with common root seed `2000003`. |

Seeds 0–5 use prompt variants p0–p5 respectively. Set `AUTOASTRA_ZSC_IMAGE` to use an existing compatible image.

</details>

## Results

Each model’s `reports/<environment>/xp.csv` is a 6 × 6 matrix. **SP** averages the six diagonal scores, where one policy plays both seats. **IXP** averages the 30 off-diagonal scores, where the seats use policies from different seeds. A heatmap and JSON summary sit beside the CSV. The final policies and experiment ledgers live under each environment’s `seed-0/` through `seed-5/` directories.

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

[License](runtime/LICENSE) · [Third-party notices](runtime/THIRD_PARTY_NOTICES.md)

## Citation
