# Optional discrete SAC

`python -m pufferlib.pufferl train ENV --learner sac --env-factory module:factory
--output NEW_DIRECTORY --stop-at 2026-09-08T08:00:00-07:00`

The factory accepts `seed=` and returns a Gymnasium environment with flat float
observations and MultiDiscrete actions. Collection stacks four observations and
runs sixteen environments. Full-match termination belongs to the environment;
time limits bootstrap from the last observation, before resetting history.

The learner uses a factorized categorical actor, twin C51 critics, entropy
regularization, and a next-state latent prediction objective. It is inspired by
[SAC-BBF](https://github.com/lezhang-thu/bigger-better-faster-SAC), not a
reproduction of its Atari recipe. Four policy samples with a leave-one-out
baseline adapt its sampled discrete policy gradient to joint MultiDiscrete
actions. Critics retain the complete action, including cross-factor effects.
There is no image augmentation, multi-step SPR, periodic network reset, or
claim to reproduce published sample-efficiency results. One-step Bellman targets
avoid uncorrected multi-step off-policy targets.

Torch uses HIP through its CUDA interface on ROCm. Replay storage, proportional
priority sampling, critic/actor updates and target interpolation stay on GPU.
CPU action inference batches environments; weights synchronize every eight
updates. Replay holds at most 131072 transitions and applies importance weights
with beta increasing from .4 to 1. The default is one batch256 update per16 new
decisions. Reward is scaled by .1 for learning; logged episode reward is unscaled.
Critic support is [-100,100] scaled units. Progress metrics report critic loss,
actor loss, entropy, temperature and the fraction of target atoms clipped to
support. Sustained clipping warrants revisiting the support before extending a
campaign. Performance is measured through the real environment, not advertised
from kernel-only benchmarks.

Every five minutes, `checkpoint.pt` atomically replaces the resumable learner,
optimizer, target, RNG and bounded replay snapshot. Small `actor-N.pt` snapshots
remain independently evaluable. `--resume checkpoint.pt` restores training and
starts fresh matches on unused seeds; it does not restore simulator mid-match
state. `--steps` is the cumulative decision limit. Only load trusted checkpoints.
At `--stop-at`, the active collection/update finishes and final state is saved.

## Protein tuning

Use `pufferl sweep ENV --learner sac --env-factory module:factory
--score-function module:score --output NEW_DIRECTORY --stop-at ISO_TIMESTAMP`.
The score function returns episode rows containing `combined_return`. Protein
receives held-out combined reward and measured training-plus-evaluation cost.
The default eight trials each collect 200000 decisions and use the same32
calibration seeds. Three Sobol suggestions precede GP-guided suggestions.
Search dimensions are learning rate, batch size, entropy fraction and prediction
loss weight. The controller keeps all trial checkpoints and resumes the highest
calibration-score trial for the remaining time. This selection is not a claim
of held-out superiority: evaluate it on fresh seeds before promotion.
Protein's small GP runs on CPU; SAC retains the GPU. Trials run sequentially.
SIGTERM/SIGINT request a final save at the next training-loop boundary.
