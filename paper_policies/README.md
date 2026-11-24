# Final paper policies

The main cross-play experiments produced 462 selected policies across seven models, eleven environments, and six independent seeds. Each model directory uses the environment IDs accepted by the scoring commands. See the [scoring examples](../README.md#results) to evaluate a population or compare two models.

```text
<model>/<environment>/
├── paper_population.json        # Identifies the six selections and their environment configuration
└── seed-0/ … seed-5/            # One directory per policy
    ├── policy.py                # Defines the policy for both seats
    └── custom_primitives.py     # Present when the selected policy includes a helper
```

The sources come from the paper submission’s `paper_assets/manifest.json` at revision `a2d722dbfd6d66aad341e455b92d6db6470d1370`. Each population manifest records the original archive IDs, context and candidate hashes, seed, prompt variant, and available model metadata. The scorer checks the environment configuration and the complete policy bundle before evaluation. The output records the Docker image ID used for that evaluation.

The paper archive records one Yōkai imperfect-memory episode with no legal action and assigns it the minimum return of −4. A fresh evaluation that encounters an invalid rollout raises an error. The archived policy sources retain their original bytes.
