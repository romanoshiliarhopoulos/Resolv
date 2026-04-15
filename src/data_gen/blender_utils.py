"""
Utilities for invoking Blender as a subprocess.

Scene generation scripts run inside Blender's own Python (bpy), not the
Poetry venv. This module provides the bridge: Python code in the venv calls
run_blender_script() to dispatch a script into Blender's interpreter.
"""

import subprocess
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def blender_path() -> Path:
    config_path = PROJECT_ROOT / "pyproject.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    rel = config.get("tool", {}).get("resolv", {}).get("blender_path", ".blender/blender")
    return PROJECT_ROOT / rel


def run_blender_script(script: Path, *args: str, background: bool = True) -> subprocess.CompletedProcess:
    """
    Run a Python script inside Blender.

    Args:
        script:     Path to the .py script to execute inside Blender.
        *args:      Extra arguments passed after '--' to the script.
        background: Run headlessly (no GUI). Default True.

    Returns:
        CompletedProcess with stdout/stderr captured.

    Example:
        run_blender_script(Path("src/data_gen/scene_gen.py"), "--output", "data/renders")
    """
    cmd = [str(blender_path())]
    if background:
        cmd.append("--background")
    cmd += ["--python", str(script)]
    if args:
        cmd += ["--"] + list(args)

    return subprocess.run(cmd, check=True, capture_output=False)
