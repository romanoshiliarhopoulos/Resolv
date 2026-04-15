"""
Runner for the random scene generation engine.

Controls how many renders to produce and what parameters each scene uses.
The actual rendering is done by scene_gen.py inside Blender's Python interpreter.

Usage examples:
  # 10 renders at medium complexity, seeds 0–9
  poetry run python -m src.scene_gen.run --num-renders 10 --complexity medium

  # 5 fully randomized renders
  poetry run python -m src.scene_gen.run --num-renders 5 --randomize

  # 1 deterministic render: 4 objects, 256 samples, seed 42
  poetry run python -m src.scene_gen.run --seed 42 --num-objects 4 --samples 256

  # High-res complex scenes, custom output directory
  poetry run python -m src.scene_gen.run --num-renders 20 --complexity complex --output data/renders/complex
"""

import argparse
import random
from pathlib import Path

from src.data_gen.extractors import base

from ..data_gen.blender_utils import run_blender_script

SCENE_SCRIPT = Path(__file__).parent / "scene_gen.py"

# Preset bundles: (num_objects, samples, width, height)
COMPLEXITY_PRESETS = {
    "simple":  dict(num_objects=1, samples=64,  width=640,  height=480),
    "medium":  dict(num_objects=3, samples=128, width=1280, height=720),
    "complex": dict(num_objects=6, samples=256, width=1920, height=1080),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random scene generation engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output", type=Path, default=Path("data/renders"),
        help="Output directory for rendered PNGs",
    )
    parser.add_argument(
        "--num-renders", type=int, default=1,
        help="Number of renders to generate",
    )

    # Complexity / quality controls
    cx = parser.add_argument_group("Scene complexity")
    cx.add_argument(
        "--complexity", choices=list(COMPLEXITY_PRESETS), default=None,
        help="Complexity preset; overrides --num-objects / --samples / --width / --height",
    )
    cx.add_argument("--num-objects", type=int, default=None,
                    help="Objects to place per scene")
    cx.add_argument("--samples", type=int, default=None,
                    help="Cycles render samples (higher = slower but cleaner)")
    cx.add_argument("--width",  type=int, default=None, help="Render width in pixels")
    cx.add_argument("--height", type=int, default=None, help="Render height in pixels")

    # Randomization
    rnd = parser.add_argument_group("Randomization")
    rnd.add_argument(
        "--randomize", action="store_true",
        help="Fully randomize all scene parameters for every render",
    )
    rnd.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed; render i gets seed+i (ignored when --randomize is set)",
    )

    return parser.parse_args()


def resolve_params(args, render_index: int, master_rng: random.Random) -> dict:
    """Return the final parameter dict for one render."""
    if args.randomize:
        preset = master_rng.choice(list(COMPLEXITY_PRESETS.values()))
        return {
            "num_objects": master_rng.randint(1, 8),
            "samples":     master_rng.choice([64, 128, 256]),
            "width":       preset["width"],
            "height":      preset["height"],
            "seed":        master_rng.randint(0, 2**31 - 1),
        }

    # Start from complexity preset (or medium default) then apply overrides
    base = COMPLEXITY_PRESETS.get(args.complexity, COMPLEXITY_PRESETS["medium"]).copy()
    if args.num_objects is not None:
        base["num_objects"] = args.num_objects
    if args.samples is not None:
        base["samples"] = args.samples
    if args.width is not None:
        base["width"] = args.width
    if args.height is not None:
        base["height"] = args.height

    base_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    base["seed"] = base_seed + render_index
    
    return base


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(args.seed)

    print(f"Generating {args.num_renders} render(s) → {args.output}/")

    for i in range(args.num_renders):
        params = resolve_params(args, i, master_rng)
        output_path = str(args.output / f"render_{params['seed']:08d}.png")

        print(f"\n[{i + 1}/{args.num_renders}] {params}  →  {output_path}")

        run_blender_script(
            SCENE_SCRIPT,
            "--output",      output_path,
            "--seed",        str(params["seed"]),
            "--num-objects", str(params["num_objects"]),
            "--samples",     str(params["samples"]),
            "--width",       str(params["width"]),
            "--height",      str(params["height"]),
        )

    print(f"\nDone — {args.num_renders} render(s) in {args.output}/")


if __name__ == "__main__":
    main()
