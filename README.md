# MuJoCo two-link controller lab

The runtime derives from `nvcr.io/nvidia/pytorch:26.05-py3` and supports two pluggable joint-position controllers:

- `analytic_ik`: closed-form baseline.
- `ik_mlp`: supervised PyTorch model.

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

Adding another approach requires implementing `Controller.predict()` in `controllers/` and registering it in `controllers/registry.py`. The browser host dispatches according to the returned `ActionType`, allowing future joint-delta and torque policies to share the same observation and UI infrastructure.
