"""
Runner for the random scene generation engine.

Controls how many render pairs to produce and what parameters each scene uses.
The actual rendering is done by scene_gen.py inside Blender's Python interpreter.

Each call to scene_gen.py produces:
  <output>/clean/<seed:08d>.png         — 1024 spp + OIDN denoiser ON (ground truth)
  <output>/noisy_0004spp/<seed:08d>.png — 4 spp,  denoiser OFF  (very noisy)
  <output>/noisy_0016spp/<seed:08d>.png — 16 spp, denoiser OFF  (moderately noisy)
  <output>/noisy_0064spp/<seed:08d>.png — 64 spp, denoiser OFF  (mildly noisy)

Usage examples:
  # 10 pairs at medium complexity, seeds 0–9
  poetry run python -m src.scene_gen.run --num-renders 10 --complexity medium

  # 5 fully randomized renders
  poetry run python -m src.scene_gen.run --num-renders 5 --randomize

  # 1 deterministic render: 4 objects, seed 42
  poetry run python -m src.scene_gen.run --seed 42 --num-objects 4

  # Custom noise levels (only very noisy + mildly noisy)
  poetry run python -m src.scene_gen.run --num-renders 10 --noise-levels 4 64

  # High-res complex scenes, custom output directory
  poetry run python -m src.scene_gen.run --num-renders 20 --complexity complex --output data/renders/complex
"""

import argparse
import random
from pathlib import Path

from ..data_gen.blender_utils import run_blender_script

SCENE_SCRIPT = Path(__file__).parent / "scene_gen.py"

# Preset bundles: (num_objects, width, height)
# samples_clean is always 1024 (fixed for the clean pass);
# noise_levels are passed separately.
COMPLEXITY_PRESETS = {
    "simple":  dict(num_objects=1, width=640,  height=480),
    "medium":  dict(num_objects=3, width=1280, height=720),
    "complex": dict(num_objects=6, width=1920, height=1080),
}

DEFAULT_NOISE_LEVELS = [4, 16, 64]
DEFAULT_SAMPLES_CLEAN = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random scene generation engine — produces clean/noisy render pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output", type=Path, default=Path("data/renders"),
        help="Base output directory; clean/ and noisy_XXXXspp/ subdirs created here",
    )
    parser.add_argument(
        "--num-renders", type=int, default=1,
        help="Number of render pairs to generate",
    )

    # Complexity / quality controls
    cx = parser.add_argument_group("Scene complexity")
    cx.add_argument(
        "--complexity", choices=list(COMPLEXITY_PRESETS), default=None,
        help="Complexity preset; overrides --num-objects / --width / --height",
    )
    cx.add_argument("--num-objects", type=int, default=None,
                    help="Objects to place per scene")
    cx.add_argument("--width",  type=int, default=None, help="Render width in pixels")
    cx.add_argument("--height", type=int, default=None, help="Render height in pixels")

    # Noise / sample controls
    ns = parser.add_argument_group("Noise levels")
    ns.add_argument(
        "--samples-clean", type=int, default=DEFAULT_SAMPLES_CLEAN,
        help="Sample count for the clean (ground-truth) render",
    )
    ns.add_argument(
        "--noise-levels", type=int, nargs="+", default=DEFAULT_NOISE_LEVELS,
        help="Sample counts for noisy renders, e.g. --noise-levels 4 16 64",
    )

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
    """Return the final parameter dict for one render pair."""
    if args.randomize:
        preset = master_rng.choice(list(COMPLEXITY_PRESETS.values()))
        return {
            "num_objects": master_rng.randint(1, 8),
            "width":       preset["width"],
            "height":      preset["height"],
            "seed":        master_rng.randint(0, 2**31 - 1),
        }

    base = COMPLEXITY_PRESETS.get(args.complexity, COMPLEXITY_PRESETS["medium"]).copy()
    if args.num_objects is not None:
        base["num_objects"] = args.num_objects
    if args.width is not None:
        base["width"] = args.width
    if args.height is not None:
        base["height"] = args.height

    base_seed  = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    base["seed"] = base_seed + render_index

    return base


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(args.seed)

    noise_str = " ".join(str(s) for s in args.noise_levels)
    print(f"Generating {args.num_renders} render pair(s) → {args.output}/")
    print(f"  clean: {args.samples_clean} spp + OIDN ON")
    print(f"  noisy: {noise_str} spp  (OIDN OFF each)")

    for i in range(args.num_renders):
        params = resolve_params(args, i, master_rng)

        print(f"\n[{i + 1}/{args.num_renders}] seed={params['seed']}  "
              f"objects={params['num_objects']}  "
              f"{params['width']}x{params['height']}")

        run_blender_script(
            SCENE_SCRIPT,
            "--output-dir",    str(args.output),
            "--seed",          str(params["seed"]),
            "--num-objects",   str(params["num_objects"]),
            "--samples-clean", str(args.samples_clean),
            "--noise-levels",  *[str(s) for s in args.noise_levels],
            "--width",         str(params["width"]),
            "--height",        str(params["height"]),
        )

    total_outputs = 1 + len(args.noise_levels)  # 1 clean + N noisy
    print(f"\nDone — {args.num_renders} scene(s), "
          f"{args.num_renders * total_outputs} total renders in {args.output}/")


if __name__ == "__main__":
    main()
