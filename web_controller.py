"""Browser-based controller host for the MuJoCo two-link arm."""
from __future__ import annotations

import asyncio
import io
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import mujoco
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image

from controllers.analytic_ik import forward_kinematics
from controllers.base import ActionType, Observation
from controllers.registry import create_controllers

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
INDEX_PATH = ROOT / "static" / "index.html"
CHECKPOINT_PATH = ROOT / "artifacts" / "ik_mlp.pt"
DATASET_PATH = ROOT / "artifacts" / "ik_training_data.npz"
METRICS_PATH = ROOT / "artifacts" / "ik_mlp_metrics.json"
PPO_METRICS_PATH = ROOT / "artifacts" / "ppo_joint_delta_metrics.json"
FPS, WIDTH, HEIGHT = 30, 800, 600
SHOULDER_HEIGHT = 0.60


class Simulation:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.gravity_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.camera = mujoco.MjvCamera()
        self.controllers = create_controllers(CHECKPOINT_PATH)
        self.active_controller = (
            "ppo_joint_delta" if "ppo_joint_delta" in self.controllers
            else "ik_mlp" if "ik_mlp" in self.controllers
            else "analytic_ik"
        )
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.default_target = self.model.site_pos[self.target_site_id].copy()
        self.training_data = np.load(DATASET_PATH) if DATASET_PATH.exists() else None
        self.training_metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
        self.ppo_metrics = json.loads(PPO_METRICS_PATH.read_text()) if PPO_METRICS_PATH.exists() else None
        self.clients: set[asyncio.Queue[bytes]] = set()
        self.pressed: set[str] = set()
        self.paused = False
        self.motion_speed = 1.0
        self.joint_goal: np.ndarray | None = None
        self.policy_target_xz: np.ndarray | None = None
        self.next_policy_time = 0.0
        self.sample_info: dict[str, object] | None = None
        self.message = "Click the robot plane to choose a target"
        self.task: asyncio.Task[None] | None = None
        self.reset()

    def reset_camera(self) -> None:
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (0.0, 0.0, 0.85)
        self.camera.distance = 2.3
        self.camera.azimuth = 90.0
        self.camera.elevation = -8.0

    def reset(self) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "bent")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.model.site_pos[self.target_site_id] = self.default_target
        self.joint_goal = None
        self.policy_target_xz = None
        self.next_policy_time = 0.0
        self.sample_info = None
        self.message = "Robot reset"
        mujoco.mj_forward(self.model, self.data)
        self.reset_camera()

    def observation(self, target_xz: np.ndarray) -> Observation:
        return Observation(
            self.data.qpos.copy(), self.data.qvel.copy(), np.asarray(target_xz), self.data.ctrl.copy()
        )

    def set_controller(self, name: str) -> None:
        if name in self.controllers:
            self.active_controller = name
            self.joint_goal = None
            self.policy_target_xz = None
            self.message = f"Using {name}"
        else:
            self.message = f"Controller '{name}' is unavailable"

    def checkpoint_info(self) -> list[dict[str, object]]:
        paths = (CHECKPOINT_PATH, CHECKPOINT_PATH.with_name("ppo_joint_delta.pt"))
        return [
            {"name": path.name, "size_bytes": path.stat().st_size,
             "modified_ns": path.stat().st_mtime_ns}
            for path in paths if path.exists()
        ]

    def reload_checkpoints(self) -> dict[str, object]:
        """Load all controller files before atomically replacing live instances."""
        try:
            reloaded = create_controllers(CHECKPOINT_PATH)
            training_metrics = (
                json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
            )
            ppo_metrics = (
                json.loads(PPO_METRICS_PATH.read_text()) if PPO_METRICS_PATH.exists() else None
            )
            checkpoint_info = self.checkpoint_info()
        except Exception as error:
            self.message = f"Checkpoint reload failed: {type(error).__name__}: {error}"
            return {"ok": False, "message": self.message,
                    "controllers": list(self.controllers)}

        previous_active = self.active_controller
        if previous_active not in reloaded:
            previous_active = (
                "ppo_joint_delta" if "ppo_joint_delta" in reloaded
                else "ik_mlp" if "ik_mlp" in reloaded
                else "analytic_ik"
            )
            self.joint_goal = None
            self.policy_target_xz = None
        self.controllers = reloaded
        self.active_controller = previous_active
        self.training_metrics = training_metrics
        self.ppo_metrics = ppo_metrics
        loaded = ", ".join(item["name"] for item in checkpoint_info) or "no checkpoint files"
        self.message = f"Reloaded checkpoints: {loaded}"
        return {
            "ok": True, "message": self.message,
            "controllers": list(self.controllers),
            "active_controller": self.active_controller,
            "checkpoints": checkpoint_info,
        }

    def set_speed(self, speed: float) -> None:
        self.motion_speed = max(0.1, min(2.5, speed))

    def set_goal(self, target_xz: np.ndarray, controller_name: str | None = None) -> bool:
        name = controller_name or self.active_controller
        controller = self.controllers.get(name)
        if controller is None:
            self.message = f"Controller '{name}' is unavailable"
            return False
        try:
            action = controller.predict(self.observation(target_xz))
        except ValueError as error:
            self.message = str(error)
            return False
        if action.action_type is ActionType.JOINT_POSITION:
            self.policy_target_xz = None
            self.joint_goal = action.values.astype(float)
        elif action.action_type is ActionType.JOINT_DELTA:
            self.joint_goal = None
            self.policy_target_xz = np.asarray(target_xz, dtype=float)
            self.next_policy_time = self.data.time
        else:
            self.message = f"Unsupported action type: {action.action_type.value}"
            return False
        self.message = f"Moving with {name}"
        return True

    def handle_key(self, key: str, down: bool) -> None:
        key = key.lower()
        if down:
            if key in {"q", "a", "w", "s"}:
                self.joint_goal = None
                self.policy_target_xz = None
                self.sample_info = None
            if key == "r":
                self.reset()
                return
            if key == "c":
                self.reset_camera()
                return
            if key == " " and key not in self.pressed:
                self.paused = not self.paused
            self.pressed.add(key)
        else:
            self.pressed.discard(key)

    def pick_target(self, screen_x: float, screen_y: float) -> None:
        left_camera, right_camera = self.renderer.scene.camera
        camera_pos = (left_camera.pos + right_camera.pos) / 2.0
        forward = left_camera.forward.astype(float)
        forward /= np.linalg.norm(forward)
        up = left_camera.up.astype(float)
        up /= np.linalg.norm(up)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        near = float(left_camera.frustum_near)
        half_height = float(left_camera.frustum_top - left_camera.frustum_bottom) / 2.0
        center_height = float(left_camera.frustum_top + left_camera.frustum_bottom) / 2.0
        half_width = half_height * WIDTH / HEIGHT
        ray = near * forward + screen_x * half_width * right + (center_height + screen_y * half_height) * up
        ray /= np.linalg.norm(ray)
        if abs(ray[1]) < 1e-5:
            self.message = "Orbit toward a side view before selecting a target"
            return
        distance = -camera_pos[1] / ray[1]
        if distance <= 0:
            self.message = "The robot plane is behind the camera"
            return
        point = camera_pos + distance * ray
        if point[2] < 0.06:
            self.message = "Target is below the usable workspace"
            return
        target_xz = np.array([point[0], point[2] - SHOULDER_HEIGHT])
        if self.set_goal(target_xz):
            self.model.site_pos[self.target_site_id] = (point[0], 0.0, point[2])
            self.sample_info = None

    def show_training_sample(self, index: int, use_prediction: bool) -> None:
        if self.training_data is None:
            self.message = "No generated training data is available"
            return
        count = len(self.training_data["current_q"])
        index %= count
        current_q = self.training_data["current_q"][index].astype(float)
        target_xz = self.training_data["target_xz"][index].astype(float)
        label_q = self.training_data["label_q"][index].astype(float)
        self.data.qpos[:] = current_q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = current_q
        self.model.site_pos[self.target_site_id] = (target_xz[0], 0.0, target_xz[1] + SHOULDER_HEIGHT)
        mujoco.mj_forward(self.model, self.data)
        prediction_q = None
        prediction_error = None
        if "ik_mlp" in self.controllers:
            prediction_q = self.controllers["ik_mlp"].predict(self.observation(target_xz)).values
            prediction_error = float(np.linalg.norm(forward_kinematics(prediction_q) - target_xz))
        use_mlp = use_prediction and prediction_q is not None
        self.joint_goal = prediction_q.copy() if use_mlp else label_q.copy()
        mode = "MLP prediction" if use_mlp else "supervised label"
        self.sample_info = {
            "index": index, "count": count, "start_q": current_q.tolist(),
            "target_xz": target_xz.tolist(), "label_q": label_q.tolist(),
            "prediction_q": prediction_q.tolist() if prediction_q is not None else None,
            "prediction_cartesian_error_m": prediction_error, "playing": mode,
        }
        self.message = f"Training sample {index}: {mode}"

    def apply_policy_controls(self) -> None:
        # PPO was trained at 50 Hz. Rendering at 30 FPS must not change its dynamics.
        if self.policy_target_xz is None or self.data.time + 1e-9 < self.next_policy_time:
            return
        controller = self.controllers[self.active_controller]
        action = controller.predict(self.observation(self.policy_target_xz))
        if action.action_type is ActionType.JOINT_DELTA:
            self.data.ctrl[:] = np.clip(
                self.data.qpos + 0.40 * action.values * self.motion_speed,
                self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1],
            )
        self.next_policy_time = self.data.time + 0.02

    def apply_controls(self, dt: float) -> None:
        if self.joint_goal is not None:
            delta = self.joint_goal - self.data.ctrl
            max_step = self.motion_speed * dt
            self.data.ctrl[:] += np.clip(delta, -max_step, max_step)
            if np.max(np.abs(delta)) < 1e-3:
                self.data.ctrl[:] = self.joint_goal
        self.data.ctrl[0] += 1.2 * dt * (("q" in self.pressed) - ("a" in self.pressed))
        self.data.ctrl[1] += 1.2 * dt * (("w" in self.pressed) - ("s" in self.pressed))
        self.data.ctrl[:] = self.data.ctrl.clip(
            self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )
        self.camera.azimuth += 70.0 * dt * (
            ("arrowright" in self.pressed) - ("arrowleft" in self.pressed)
        )
        self.camera.elevation += 70.0 * dt * (
            ("arrowup" in self.pressed) - ("arrowdown" in self.pressed)
        )
        self.camera.elevation = max(-89.0, min(89.0, self.camera.elevation))
        zoom = ("-" in self.pressed) - ("+" in self.pressed or "=" in self.pressed)
        self.camera.distance *= max(0.2, 1.0 + 1.4 * dt * zoom)
        self.camera.distance = max(0.7, min(6.0, self.camera.distance))

    def apply_gravity_compensation(self) -> None:
        self.gravity_data.qpos[:] = self.data.qpos
        self.gravity_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.gravity_data)
        self.data.qfrc_applied[:] = self.gravity_data.qfrc_bias

    def render_jpeg(self) -> bytes:
        self.renderer.update_scene(self.data, camera=self.camera)
        buffer = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()

    async def run(self) -> None:
        frame_dt = 1.0 / FPS
        loop = asyncio.get_running_loop()
        next_frame = loop.time()
        while True:
            self.apply_controls(frame_dt)
            if not self.paused:
                target_time = self.data.time + frame_dt
                while self.data.time < target_time:
                    self.apply_policy_controls()
                    self.apply_gravity_compensation()
                    mujoco.mj_step(self.model, self.data)
            frame = self.render_jpeg()
            for queue in tuple(self.clients):
                if queue.full():
                    with suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                queue.put_nowait(frame)
            next_frame += frame_dt
            await asyncio.sleep(max(0.0, next_frame - loop.time()))

    def status(self) -> str:
        target = self.model.site_pos[self.target_site_id]
        error = float(np.linalg.norm(self.data.site("end_effector").xpos - target))
        if self.joint_goal is not None and error < 0.015:
            self.message = "Target reached"
        if self.policy_target_xz is not None and error < 0.02 and np.linalg.norm(self.data.qvel) < 0.1:
            self.message = "Target reached"
        elif self.policy_target_xz is not None:
            self.message = f"Moving with {self.active_controller}"
        return json.dumps({
            "type": "status", "qpos": self.data.qpos.tolist(),
            "ctrl": self.data.ctrl.tolist(), "paused": self.paused,
            "time": self.data.time, "target": target.tolist(), "error": error,
            "speed": self.motion_speed, "message": self.message,
            "controller": self.active_controller, "sample": self.sample_info,
        })

    def public_info(self) -> dict[str, object]:
        metrics = None
        if self.training_metrics:
            metrics = self.training_metrics
        return {
            "controllers": list(self.controllers),
            "active_controller": self.active_controller,
            "training_metrics": metrics,
            "ppo_metrics": self.ppo_metrics,
            "training_samples": len(self.training_data["current_q"]) if self.training_data is not None else 0,
        }

    def close(self) -> None:
        if self.training_data is not None:
            self.training_data.close()
        self.renderer.close()


simulation: Simulation | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global simulation
    simulation = Simulation()
    simulation.task = asyncio.create_task(simulation.run())
    try:
        yield
    finally:
        simulation.task.cancel()
        with suppress(asyncio.CancelledError):
            await simulation.task
        simulation.close()


app = FastAPI(title="MuJoCo Two-Link Arm", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/health")
async def health() -> dict[str, object]:
    assert simulation is not None
    return {"ok": True, "model": MODEL_PATH.name, "time": simulation.data.time}


@app.get("/api/info")
async def info() -> dict[str, object]:
    assert simulation is not None
    return simulation.public_info()


@app.post("/api/reload-checkpoints")
async def reload_checkpoints() -> dict[str, object]:
    assert simulation is not None
    return await asyncio.to_thread(simulation.reload_checkpoints)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    assert simulation is not None
    await websocket.accept()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    simulation.clients.add(queue)

    async def send_frames() -> None:
        status_counter = 0
        while True:
            await websocket.send_bytes(await queue.get())
            status_counter += 1
            if status_counter >= 6:
                await websocket.send_text(simulation.status())
                status_counter = 0

    sender = asyncio.create_task(send_frames())
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            if message_type == "key":
                simulation.handle_key(str(message.get("key", "")), bool(message.get("down")))
            elif message_type == "speed":
                simulation.set_speed(float(message.get("value", 1.0)))
            elif message_type == "target":
                simulation.pick_target(float(message.get("x", 0.0)), float(message.get("y", 0.0)))
            elif message_type == "controller":
                simulation.set_controller(str(message.get("name", "")))
            elif message_type == "training_sample":
                simulation.show_training_sample(
                    int(message.get("index", 0)), bool(message.get("prediction", True))
                )
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        simulation.clients.discard(queue)
        with suppress(asyncio.CancelledError):
            await sender
