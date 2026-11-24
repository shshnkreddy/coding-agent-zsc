# Hanabi hybrid trainer

The reference is the archived `hanabi_ce_teachers_fable_astra_20261018` experiment. Each of the six H-Group teachers from each model supervised one student in standard two-player Hanabi. Every student used seed 5. The requested budget of 10 billion environment steps gives 76,290 PPO updates across ten evaluation intervals, or 9,999,482,880 steps per student.

The [paper configuration](paper_config.json) uses a two-layer LSTM with 512 units per layer, a cross-entropy weight of 0.1, and an entropy coefficient of 0.05. Each PPO update collects 128 steps from 1,024 environments. After each evaluation interval, the student plays 1,024 held-out episodes in each seat with the teacher, with both greedy and sampled actions.

The actor-critic and belief modules come from JAX-OBL revision `dc762ccfc59822f19d1889c005b42b73a9b414b4`. The policy update, independent teacher RNG stream, greedy and sampled evaluation in both seats, and checkpoint selection come from the frozen hybrid experiment. Training logs and weights are written locally. The best checkpoint maximizes the mean greedy return with the teacher over both seats. [Upstream implementation](https://github.com/jforkel/JAX-OBL/tree/dc762ccfc59822f19d1889c005b42b73a9b414b4)

For a small CPU run, build `runtime/Dockerfile.hybrid` with `--build-arg BASE_IMAGE=autoastra-zsc:cpu` after building the project's CPU image. Remove `--gpus 1 -e JAX_PLATFORMS=cuda` and add `--updates 2 --num-envs 4 --rollout-steps 8 --hidden-size 16 --eval-episodes 4` to the training command. This exercises training and held-out selection with a smaller network and budget.
