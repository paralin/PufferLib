# Native KLPO

`--train.learner=klpo` selects an experimental, critic-free native learner for
conditional discrete controls. PPO remains the default. Both learners use the
same recurrent encoder, MinGRU backpropagation, Muon optimizer, CUDA/HIP buffers,
graphs, prioritized rollout replay and paired checkpoints.

The implementation follows the token-regression objective in
[KLPO](https://github.com/yifanzhang-pro/KLPO), reference revision
`ecd92da48fd6ee7ddaf51cad7f6c9c6999343dc9`. It computes the sampler expectation
exactly instead of estimating it with auxiliary samples. It does not use the
unrelated Go heuristics described as KLPO in the Zero-Weight Q* report.

For the historical sampler `q`, current learner `p` and feedback `R`, each
decision contributes:

```
z = log p(action) - sum_control q(control) log p(control)
h = stop_gradient(R - beta * log(p(action) / q(action)))
loss = -h * z
```

The loss sums decisions and averages complete episodes. There is no PPO ratio,
clipping, advantage normalization, learned baseline or entropy bonus. All actor
layers train. The portable decoder retains its value row for weight-layout
compatibility, with zero value gradient. PPO-specific loss options have no effect.

The collector saves float32 normalized sampler conditionals. Each independent
control group sums probabilities of equivalent encodings before scoring a
schedule: time-zero switches and identical endpoints are constant controls.
The GPU kernel differentiates these sums and their conditional softmaxes
analytically. It does not enumerate the Cartesian product across control groups.

## Returns and replay

- `--train.klpo_returns=whole_match` assigns the sum of an episode's rewards to
  every decision: the paper's trajectory-return form. Gamma is ignored.
- `--train.klpo_returns=discounted` assigns Monte Carlo reward-to-go with the
  configured gamma. This is an explicit dense-reward adaptation, not a claim
  that the paper's whole-trajectory equivalence proof applies unchanged.
- `--train.klpo_beta=0.1` sets the historical-sampler KL penalty. Scale it with
  the reward units; it is not a coefficient against a permanent reference policy.

Replay is bounded by the current rollout. Row priority uses absolute valid
returns. Full inverse sampling correction, including the global episode count,
preserves the episode average over the retained data. The user-selected
`prio_beta0` schedule is overridden with 1 for KLPO. Replay must be enabled.

Only episodes whose start and end both lie inside one collection window train.
This guarantees a fixed recorded sampler and a real recurrent reset. Unfinished
edges, including the initial partial episode, have no loss. Empty windows do
not step optimizer momentum. This filtering favors shorter episodes; matches
longer than the horizon cannot contribute. It is a current collector limitation,
not an unbiased full-match sampling claim. Use a long horizon, inspect
`klpo/used_fraction`, and account for discarded interactions when comparing
sample efficiency. Do not substitute bootstrapped values to hide this loss.

`klpo/complete_episodes` and `klpo/used_decisions` are cumulative unique retained
data, not multiplied by replay reuse. `fresh_decisions` counts collection,
including asynchronous prefetch. `loss/kl` is exact mean schedule KL(q || p),
`loss/entropy` is mean schedule entropy, and `loss/policy` is the per-episode
surrogate; none is an independent gameplay score.

Current support is one GPU and conditional discrete controls with recurrent
carry. Continuous actions, arbitrary masked categories and multi-GPU episode
normalization are not implemented. Paired resume restores optimizer, RNG and
counters and starts fresh matches, as PPO does. It does not resume an interrupted
match or preserve discarded rollout edges. Old paired PPO checkpoints still load.

## Checks

```
python -m pytest -q --timeout=90 tests/test_klpo.py
KLPO_TEST_HIP=1 python -m pytest -q --timeout=90 tests/test_klpo.py
PUFFER_KLPO_BINARY=/absolute/path/build/hip/puffer \
  python -m pytest -q --timeout=120 tests/test_klpo_training.py
```

The arithmetic probe compiles the production score and target functions. Its
independent float64 autograd reference enumerates and merges schedules. The
native trainer check exercises eager/graph agreement, asynchronous collection,
complete-match and discounted targets, recurrent weight updates, inert value
weights, empty-window skipping, PPO and paired resume. The HIP build has been
exercised on an RX 7700 XT; CUDA execution and learning-quality superiority are
not established by these checks.
