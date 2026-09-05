"""Shared MuJoCo task definitions; gravity compensation is enabled at compile time."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "two_link_arm.xml"


def load_training_model() -> mujoco.MjModel:
    root = ET.parse(MODEL_PATH).getroot()
    for name in ("link1", "link2"):
        root.find(f".//body[@name='{name}']").set("gravcomp", "1")
    # Setting body_gravcomp after compilation leaves model.ngravcomp == 0.
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.ngravcomp == 2
    return model
