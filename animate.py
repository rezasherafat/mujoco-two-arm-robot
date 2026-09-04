"""Simulate the two-link arm and save a remote-friendly MP4 animation."""

from pathlib import Path

import mediapy as media
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs"
VIDEO_PATH = OUTPUT_DIR / "two_link_arm.mp4"

FPS = 30
DURATION_SECONDS = 8
WIDTH = 960
HEIGHT = 720


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "bent")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    frames: list[np.ndarray] = []
    frame_interval = 1.0 / FPS
    next_frame_time = 0.0

    while data.time < DURATION_SECONDS:
        # Position actuator controls are desired joint angles in radians.
        data.ctrl[0] = -0.65 + 0.45 * np.sin(0.8 * data.time)
        data.ctrl[1] = 1.25 + 0.65 * np.sin(1.1 * data.time + 0.7)
        mujoco.mj_step(model, data)

        if data.time + 1e-9 >= next_frame_time:
            renderer.update_scene(data, camera="side")
            frames.append(renderer.render().copy())
            next_frame_time += frame_interval

    renderer.close()
    media.write_video(VIDEO_PATH, frames, fps=FPS)

    end_effector = data.site("end_effector").xpos
    print(f"Model: {MODEL_PATH}")
    print(f"Simulation: {data.time:.3f} seconds, {len(frames)} frames")
    print(f"Final joint angles: {data.qpos}")
    print(f"Final end-effector position: {end_effector}")
    print(f"Video: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
