# PPO reaching: diagnosis and reproducible checks

## Confirmed bugs

1. **Gravity compensation was disabled in training.** Changing `body_gravcomp`
   after MuJoCo compiled the model left `model.ngravcomp == 0`, so passive gravity
   compensation was skipped. A zero-action test over 64 starting poses drifted by
   0.721 rad on average (maximum 1.306 rad) in two seconds. Compiling the two link
   bodies with `gravcomp="1"` makes the same test hold to numerical precision.
   The browser already applied explicit compensation, so training and deployment
   previously had different dynamics. See `training/arm_task.py` and the parity test.
2. **The old reward could prefer hovering to success.** Positive bonuses outside
   the terminal success radius allowed repeatedly earning reward without finishing.
   Removed these bonuses; distance now has a non-positive cost and completing the
   task gets a one-off +10 reward.
3. **Timeouts were treated as terminal states.** Time-limit transitions now
   bootstrap the critic from the final observation before auto-reset; GAE still
   stops at the reset boundary. True success does not bootstrap.
4. **Success definitions disagreed.** Training used 4 cm and evaluation used 2 cm.
   Both now require <2 cm Cartesian error and <0.25 rad/s joint-velocity norm for
   five consecutive 50 Hz steps. Evaluation reports both reaching-and-settling at
   any time and remaining settled at the end. It does not reset successful cases.
5. **Browser policy frequency differed.** Inference now runs every 20 ms of
   simulation time independently of the 30 FPS renderer, matching training.
   Feedback stays enabled after reaching; a status request no longer disables it.

Other changes: separate actor/critic gradient clipping; unclipped critic MSE;
full-workspace target sampling instead of a stalled distance curriculum; persistent
per-update JSON metrics; isolated output directories; additional evaluation metrics.

## What “100%” means here

Use independently sampled, reachable, above-floor starting and goal configurations
inside the joint limits. The benchmark allows **8 seconds** (the old run allowed
3 seconds). Neither a looser tolerance nor an easier target subset is used to claim
success. An arbitrary browser click may be unreachable; moving targets, other speed
slider settings, collisions, and different start-state distributions need separate
tests. A finite benchmark score is not a universal guarantee.

The analytic reference is a feasibility check, **not the learned PPO policy**.
It achieved 256/256 reached-and-settled and 256/256 final-settled cases on seed 10719.
The first corrected, purely reward-trained PPO experiment reached 188/256 (73.44%)
on diagnostic seed 20719, with 71.88% still settled at the end. This establishes a
real improvement but does **not** establish 100% learned-policy success.

Final comparison on the same **512 new cases**, seed 50721:

| Controller | Reached and settled | Still settled at 8 s | Mean final error |
| --- | ---: | ---: | ---: |
| Corrected PPO | 363/512 (70.90%) | 69.92% | 82.03 mm |
| PPO + IK reward | 372/512 (72.66%) | 72.07% | 79.48 mm |
| Teacher-assisted PPO | 435/512 (84.96%) | 81.05% | 29.65 mm |
| Analytic reference (not PPO) | 512/512 (100%) | 100% | numerical precision |

Raw per-case results are in `outputs/ppo_comparison/*.json`. The teacher-assisted
run's best training validation was 89.06% on 128 cases; that is **not** its held-out
test score. Checkpoints remain experimental and have not been installed in the UI.

Additional reward shaping alone did not solve the tail of failures. The next
neural-only experiment should emphasize on-policy corrective demonstrations and
settling cases, and test changes across multiple seeds. If reliable reaching is
required before the neural policy meets the criterion, an explicitly labeled
PPO-plus-analytic-recovery controller is a practical option, but it must be reported
as a hybrid and benchmarked separately; its success must not be credited to PPO.

## Reproduce

Run these Python commands inside the project's `mujoco-thor:26.05` container,
with this repository mounted as `/workspace` and working directory `/workspace`:

```bash
python -m unittest discover -s tests -v
python -m training.evaluate_ppo --analytic-reference --episodes 256 --seed 10719
python -m training.evaluate_ppo \
  --checkpoint outputs/ppo_debug_physics/ppo_joint_delta.pt \
  --episodes 256 --seed 20719 --output outputs/diagnostic-evaluation.json
```

`training/evaluate_ppo.py` never trains or changes a checkpoint. Its JSON includes
failed start configurations, targets, final poses, and errors for reproduction.
Validation during training selects checkpoints; use a **new seed** for a final
hold-out test, especially after inspecting failures on an earlier test set.

Pure PPO configuration used for the corrected baseline:

```bash
python -u -m training.train_ppo_joint_delta \
  --updates 300 --envs 64 --rollout-steps 128 --epochs 10 --minibatch 1024 \
  --seed 719 --eval-episodes 64 --eval-interval 20 \
  --output-dir outputs/ppo_debug_physics \
  --wandb-project mujoco-debug --wandb-mode offline
```

Optional approaches remain explicitly opt-in:

- `--ik-reward-weight 2`: additional reward for approaching either valid IK
  solution in joint space. This is privileged reward shaping, not runtime IK.
- `--teacher-pretrain-steps 4000 --teacher-weight 1`: supervised actor warm-start
  and an auxiliary imitation loss during PPO. This is **teacher-assisted PPO**,
  not reinforcement learning entirely from scratch. The critic still learns from
  returns and PPO updates still run. The controller uses only its neural actor at
  inference. `loss/imitation` is logged separately.
- `--init-checkpoint PATH`: load corrected-physics model weights, with a **new**
  optimizer and reset exploration. This is not exact training resumption.

Training metrics include policy/value/imitation losses, KL, clipping fraction,
explained variance, exploration standard deviations, rollout return/success,
validation success and final-settled rate. Rollout success is over the most recent
100 *completed* episodes, so early numbers can be biased toward short/easy episodes.
Prefer the fixed-size deterministic evaluation for comparing policies.

Experiments are isolated under `outputs/`; they do not replace the live browser
checkpoint under `artifacts/`. W&B runs in these debugging experiments are offline;
cloud upload is a separate action.
