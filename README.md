# MuJoCo two-link controller lab

The runtime derives from `nvcr.io/nvidia/pytorch:26.05-py3` and supports three pluggable controllers:

- `analytic_ik`: closed-form baseline.
- `ik_mlp`: supervised PyTorch model.
- `ppo_joint_delta`: PPO policy producing continuous joint offsets.

## Train the IK MLP

```bash
docker build -t mujoco-thor:26.05 .
./train.sh
```

Training generates `artifacts/ik_training_data.npz`, writes validation history to `artifacts/ik_mlp_metrics.json`, and saves the best checkpoint as `artifacts/ik_mlp.pt`.

The six MLP inputs are `sin(q1), cos(q1), sin(q2), cos(q2), target_x, target_z`. The two outputs are desired joint angles. The objective is joint-angle MSE plus ten times Cartesian forward-kinematics MSE. Joint loss preserves the selected elbow branch; Cartesian loss directly encourages the end effector to reach the target.

## Run browser inference

```bash
./run-web.sh
```

From a Mac, forward the loopback-only port and open <http://127.0.0.1:8000>:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@THOR_HOST
```

The page supports controller selection, click-to-reach, command-speed control, training-sample playback, label/prediction comparison, validation metrics, and a loss curve.

The **Reload checkpoint** button reloads `artifacts/ik_mlp.pt` and
`artifacts/ppo_joint_delta.pt` without restarting the simulation. Checkpoints are
fully loaded before the live controller instances are replaced, so a failed or
incompatible load leaves the currently running models intact.

Adding another approach requires implementing `Controller.predict()` in `controllers/` and registering it in `controllers/registry.py`. The browser host dispatches according to the returned `ActionType`, allowing future joint-delta and torque policies to share the same observation and UI infrastructure.


## Train PPO with Weights & Biases

Rebuild the Thor image after changing dependencies:

```bash
docker build -t mujoco-thor:26.05 .
```

For live W&B monitoring, export your key in the Thor shell and enable a project. Do not put the key in this repository:

```bash
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
./train-ppo.sh \
  --wandb-project mujoco-two-link \
  --wandb-run-name ppo-joint-delta
```

Optionally add `--wandb-entity ENTITY`. Add `--wandb-watch` to collect gradient histograms; this has additional overhead. Without `--wandb-project`, training behaves exactly as before and does not import or initialize W&B.

To test logging without uploading anything:

```bash
./train-ppo.sh \
  --updates 2 --envs 8 --rollout-steps 32 --epochs 2 --minibatch 128 \
  --wandb-project mujoco-two-link --wandb-mode offline
```

W&B records rollout distance, success and return; policy/value losses; entropy, KL and clip fraction; critic explained variance; shoulder/elbow action standard deviations; and deterministic evaluation distance, success, and final-settled metrics.

See [PPO debugging and evaluation](PPO_DEBUGGING.md) for the corrected physics,
success definition, reproducible benchmarks, and optional teacher-assisted training.
See [PPO next steps](PPO_NEXT_STEPS.md) for the failure-mode breakdown of the
corrected policy, the exploration-noise objective mismatch and the hold-to-complete
training protocol that addresses it, and the remaining candidate changes.
