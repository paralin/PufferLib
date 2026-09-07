# Optional PufferLib Q learner

PPO remains the default. The staged fork also supports frozen-proposal Q
learning through the same entry point:

```sh
python -m pufferlib.pufferl train llb-rust --learner q \
  --q-adapter play_llb.qplan.adapter:LLBAdapter \
  --actor /absolute/actor.bin --output /absolute/new-run \
  --episodes 64 --held-out 200 --updates 5000 \
  --replay-capacity 131072 --device cuda --seed 17000000
```

Use the existing Puffer staging/build procedure. Put the staged Puffer package
and the adapter project's `python/` directory on `PYTHONPATH`, and set
`LLB_RUST_SIM_LIBRARY` to the selected Rust library. The optional `qplan` library
is supplied by the LLB adapter project; it provides replay records and categorical
critic primitives. Q mode does not require a compiled Puffer PPO extension.
CUDA/ROCm PyTorch is required only when selecting `--device cuda`.

PufferLib owns bounded retention, prioritized sampling, critic optimization,
checkpoint continuation and comparison. The LLB adapter owns the actor and game
contracts. The actor remains frozen: sixteen legal proposals are ranked by the
critic. This mode does not jointly train the actor or replace native PPO's
update rule. It supports single-device training; Q sweep orchestration is not
yet connected to Protein.

The critic retains the historical bounded terminal HP objective. Matched
sampled-roster CPU7 evaluation reports the shared combined gameplay reward,
with its coefficients and separate evaluation decisions. These are different
quantities. Small integration runs do not establish stronger play.

Replay retains at most `--replay-capacity` transitions with FIFO eviction.
Retained transitions are sampled by proportional TD priority, with importance
weights. Actor context, executed next actions, optimizer/target state, priorities
and sampling RNG retain the existing checkpoint contract.

Continue fitting the same immutable replay in a fresh output directory:

```sh
python -m pufferlib.pufferl train llb-rust --learner q \
  --q-adapter play_llb.qplan.adapter:LLBAdapter \
  --actor /absolute/actor.bin --resume /absolute/previous-run \
  --output /absolute/resumed-run --updates 10000 --device cuda
```

`--updates` is the total completed update count. Resume preserves the actor,
kernel, replay, runtime and held-out seed identities and collects no new training
transitions. Use `--init-from /absolute/previous-run` with disjoint seeds to
collect fresh guided experience and start a new critic refinement experiment.
An existing output manifest is never overwritten.

The previous `python -m play_llb.qplan` command calls the same Puffer learner.
Its checkpoints and immutable replay remain readable. There is one training
loop and one prioritized critic trainer, not a subprocess launcher around the
old command.
