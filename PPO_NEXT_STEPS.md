# PPO reaching: why the corrected policy plateaus, and what to change

Companion to [PPO debugging and evaluation](PPO_DEBUGGING.md). That document fixed
the physics and the reward exploit and established a reproducible benchmark. This
one diagnoses why the corrected policy still stops short of the analytic reference,
using measurements on the 512-case hold-out (seed 50721) rather than on training
curves.

Everything below is diagnosis plus one implemented change. No result here claims a
learned policy has met the criterion. The benchmark in `training/evaluate_ppo.py`
is unchanged and still reproduces the published table bit-for-bit
(`outputs/ppo_debug_physics/ppo_joint_delta.pt` → 363/512, mean final error
0.08203484024815733 m), so old and new checkpoints stay comparable.

## What the failures actually are

Failing episodes end at **0.000 rad/s**. The arm does not oscillate, hunt, or run
out of time — median time to success is 1.0–1.7 s of an 8 s budget. It drives to a
stable fixed point that is not the target and parks there. The velocity gate in the
success definition is never the binding constraint.

Two distinct failure populations, needing different fixes:

| | pure PPO (149 failures) | teacher-assisted (77 failures) |
| --- | ---: | ---: |
| never within 10 cm — wrong basin | 51.0% | 15.6% |
| within 5 cm, stopped short — precision | 37.6% | 80.5% |
| ended pinned at a joint limit (\|q\|>2.6) | 45.0% | 11.7% |
| median best error over the episode | 10.2 cm | 2.4 cm |

The teacher's contribution was almost entirely to the first mode. Its residual
failures stop dead a median of 4 mm outside tolerance.

## Finding 1: the training success criterion is inflated by exploration noise

Running `outputs/ppo_debug_teacher/ppo_joint_delta.pt` on the 512-case hold-out at
multiples of its own training noise:

| action noise | reached and settled (the training criterion) | still settled at 8 s |
| --- | ---: | ---: |
| ×0 (deployed mean) | 85.0% | 81.1% |
| ×0.5 | 90.6% | 65.2% |
| **×1.0 (training)** | **92.0%** | **10.2%** |
| ×2.0 | 46.5% | 0.4% |
| ×4.0 | 0.4% | 0.0% |

At training noise the policy scores **better** on the criterion PPO optimizes
(92.0% vs 85.0%) while being almost entirely unable to stay at the goal (10.2%).
Exploration noise dithers the end effector through the 2 cm ball long enough to
register a five-step streak; the episode then terminated and paid +10 before the
inability to hold was ever scored.

This is the same species of bug as the hovering exploit already fixed, but living
in the noise distribution rather than in the reward's shape. It was visible in the
logs only as an unexplained gap between `rollout/success_rate` (0.940 in the
teacher run) and `evaluation/success_rate` (0.82–0.89). Because the stochastic
objective was already near-satisfied, PPO had little gradient pressure left to fix
the deterministic failures — which is what both more training and more shaping ran
into.

### What was changed

In `training/train_ppo_joint_delta.py`:

1. **A settling streak no longer ends a training episode.** Episodes terminate on
   the time limit only, so every termination is a truncation that bootstraps the
   critic.
2. **The reward pays per step while settled** (`HOLD_BONUS = 1.0`, gated on the
   same `<2 cm` and `<0.25 rad/s` predicate) instead of paying +10 once for a
   five-step streak. It is the only positive term and it is unreachable outside the
   success region, so hovering still cannot earn it — but an arm dithered through
   the goal now collects it only on the steps it is actually inside.
3. **A reach counts only after `HOLD_STEPS = 25` (0.5 s) of continuous settling**,
   after which the arm is **re-targeted in place** rather than reset. Chaining
   reaches inside one episode keeps target throughput up and makes the settled
   state something to hold rather than an exit. The benchmark's own five-step
   criterion (`SETTLE_STREAK`) is untouched.
4. **`log_std` is annealed** under a ceiling falling linearly from
   `--initial-log-std` to `--final-log-std` (default −4.0) over
   `--log-std-anneal-fraction` (default 0.8) of training, so the stochastic
   objective converges to the deterministic mean action that is actually deployed.
   Left free, `log_std` settles wherever it maximizes stochastic return, which the
   table above shows is not where the mean action performs best.
5. **Both evaluations are now logged every eval interval** —
   `evaluation/success_rate` (mean action) and `evaluation/stochastic_success_rate`,
   plus `evaluation/exploration_gap` between them. A large positive gap means noise,
   not skill, is scoring. Rollouts also report `rollout/settled_fraction` and
   `rollout/reaches_completed`, which are dense and cannot be faked by dithering.

Checkpoint selection still keys on deterministic `final_settled_rate` first; that
was already correct. New checkpoints carry
`reward_version: hold-to-complete-settling-bonus` so they are distinguishable from
those trained under the old protocol.

Regression tests in `tests/test_ppo_task.py` cover each invariant: no positive
reward outside the success region, a streak not ending the episode, holding paying
more than dithering, re-targeting preserving arm state, and both evaluation modes
being reported.

**This is untested as a training result.** The mechanism is measured; the benefit
is not. It needs the multi-seed protocol below before any claim.

## Finding 2: pure PPO cannot touch a structurally identifiable subpopulation

Slicing the 512 hold-out cases by geometry, with base rates from replicating the
sampler exactly:

| case type | share of cases | pure PPO | teacher-assisted |
| --- | ---: | ---: | ---: |
| both elbow branches within joint limits | 87.9% | 80.0% | 86.9% |
| **only one branch within joint limits** | 12.1% | **4.8%** | 71.0% |
| **requires flipping elbow sign from start** | 7.2% | **8.1%** | 64.9% |
| target below 0.35 m | 15.8% | 13.6% | 74.1% |
| greedy Cartesian descent gets stuck | 11.7% | 38.3% | 80.0% |

Pure PPO solves 3 of 62 single-branch cases; that bucket alone is 40% of its
failures. The mechanism is direct: a resolved-rate IK controller — which is exactly
what `10 * progress` shaping rewards — gets stuck on 60 of 512 cases, driving into
a joint limit. Escaping requires temporarily *increasing* Cartesian distance, which
the progress term penalizes. With `log_std` collapsed to 0.03–0.15 and 20 ms steps,
randomly discovering a coordinated half-second detour has negligible probability.
This is a deceptive-reward problem, and the existing conclusion that shaping alone
will not fix it is supported.

Candidate changes, in order of leverage per unit of effort:

- **Oversample the hard geometry in `reset_one`.** Single-branch targets get 12% of
  the gradient and cause 40% of the failures. `solve_all` is already called there,
  so the branch count is free. Uses no privileged information at inference.
- **Put limit proximity in the observation.** `sin/cos(q)` is invertible over ±2.8
  but makes "how much room is left before the stop" something the network must
  reconstruct. Appending `q/2.8` is two dimensions and exposes the variable that
  45% of pure-PPO failures die on.
- **Make the progress term potential-based**: `10*(γ·(−d′) − (−d))` rather than
  `10*(d − d′)`. The current form is not policy-invariant under discounting; it
  weights early progress above late progress, which is the greedy bias that traps
  the flip cases.
- **DAgger rather than one-shot pretrain.** `--teacher-weight` already applies an
  imitation loss against `teacher_actions` relabelled on on-policy states. Annealing
  that weight to zero converts "a policy permanently anchored to the teacher" into
  "PPO that was shown the right basin" — a stronger and more defensible result.

## Finding 3: the action scale is set well above what the actuator can execute

Realized joint displacement per 20 ms control interval, measured:

| action | commanded offset | realized (steady) | fraction |
| --- | ---: | ---: | ---: |
| 1.0 | 0.400 rad | 0.037 rad | 9% |
| 0.5 | 0.200 rad | 0.020 rad | 10% |
| 0.1 | 0.040 rad | 0.004 rad | 9% |

With `kp=120` and the arm's inertia the position servo's bandwidth is roughly
10 rad/s, so `ctrl = qpos + 0.4·a` re-anchored every 20 ms is not a position delta —
it is a velocity command, and the top of the action range is saturated against the
servo rather than doing anything. `MAX_DELTA` therefore sets both the maximum speed
and the Cartesian noise floor and is tuned by neither. Worth a straight ablation
over 0.1 / 0.2 / 0.4: halving it roughly halves the noise while the median reach
still fits comfortably in the 8 s budget. A state-dependent standard-deviation head
is the more principled version of the same fix, since one global `log_std` currently
has to serve both a 1 m traverse and a 2 cm endgame.

## Finding 4: the measurement protocol inflates validation

`--eval-episodes 64` with best-of-16 checkpoint selection carries roughly ±6%
binomial noise, and the evaluation seed is fixed at `seed + 10000` across all
evaluations, so selection overfits it. That plausibly accounts for most of the
teacher run's 89.06% validation → 84.96% test drop. Before believing any change
above helped:

- evaluate on 512 cases, not 64;
- draw a fresh evaluation seed each evaluation, and keep a final hold-out seed that
  has never been inspected;
- run at least three training seeds per configuration.

There is far more compute available than is being used: 14 cores, but
`torch.set_num_threads(1)` and a serial Python loop over 64 `MjData` gives about
5.4k environment steps/s. Both prior runs were still improving when stopped — pure
PPO went 0.72 @ 2.4M steps → 0.836 @ 4.9M, and the teacher run was cut off at 160
updates while still climbing. Parallelizing the environment loop is worth roughly
10×, which buys 20–50M-step runs and a multi-seed protocol in the same wall clock.

## Checked and ruled out

Recorded so they are not re-investigated:

- **tanh log-probability numerics.** The `atanh` round-trip and `1e-6` epsilon in
  `evaluate_actions` would bias saturated actions, but p99 of `|pre-tanh mean|` is
  1.34 and the maximum is 2.23. The policy never saturates. Non-issue.
- **Episode horizon.** Median time to success is 1.0–1.7 s of 8 s.
- **Oscillation and the velocity gate.** Failures end at 0.000 rad/s.
- **Critic quality.** `explained_variance` is 0.995.

## Unrelated, but worth knowing

`artifacts/ppo_joint_delta.pt` — the checkpoint `controllers/registry.py` loads and
`web_controller.py` selects as the browser default — is a pre-fix checkpoint with no
`physics_version` tag and 0.78% evaluation success. The browser is serving a policy
that does not work, while the usable checkpoints sit under `outputs/`. Promoting one
is a deliberate act and has not been done here.
