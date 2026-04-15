"""
Blender scene generation script — renders a clean/noisy pair for denoiser training.

Runs inside Blender's Python interpreter via blender_utils.run_blender_script().
Everything after -- on the command line is parsed as arguments.

Usage (via run.py / orchestrate.py — not called directly):
  blender --background --python scene_gen.py -- \
    --output-dir data/renders \
    --seed 42 \
    --num-objects 3 \
    --samples-clean 1024 \
    --noise-levels 4 16 64 \
    --width 1280 \
    --height 720

Outputs per render
------------------
  <output-dir>/clean/<seed:08d>.png          — 1024 spp + OIDN denoiser ON
  <output-dir>/noisy_0004spp/<seed:08d>.png  — 4 spp,  denoiser OFF
  <output-dir>/noisy_0016spp/<seed:08d>.png  — 16 spp, denoiser OFF
  <output-dir>/noisy_0064spp/<seed:08d>.png  — 64 spp, denoiser OFF

All renders share identical geometry, materials, HDRI, and camera.
Only sample count and denoiser flag differ — pixel-perfect alignment guaranteed.
"""

import hashlib
import bpy
import math
import mathutils
import random
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
RENDERS_DIR  = DATA_DIR / "renders"

DEFAULT_NOISE_LEVELS = [4, 16, 64]
SAMPLES_CLEAN        = 1024


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",    type=Path, default=RENDERS_DIR,
                        help="Base output directory; clean/ and noisy_XXXXspp/ subdirs created here")
    parser.add_argument("--seed",          type=int,  default=None)
    parser.add_argument("--num-objects",   type=int,  default=3)
    parser.add_argument("--samples-clean", type=int,  default=SAMPLES_CLEAN,
                        help="Sample count for the clean (ground-truth) render")
    parser.add_argument("--noise-levels",  type=int,  nargs="+", default=DEFAULT_NOISE_LEVELS,
                        help="Sample counts for noisy renders, e.g. --noise-levels 4 16 64")
    parser.add_argument("--width",         type=int,  default=1280)
    parser.add_argument("--height",        type=int,  default=720)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------

def discover_hdris() -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.hdr", "*.exr"):
        paths.extend(DATA_DIR.glob(f"hdri/**/{ext}"))
    return sorted(paths)


def discover_models() -> list[Path]:
    """Return Polyhaven GLTF models, falling back to .blend if none downloaded.

    Polyhaven .blend files are zstd/gzip-compressed and cannot be loaded via
    bpy.data.libraries.load() in headless mode.  GLTF is plain JSON+binary,
    always loadable.  Sketchfab models are excluded (watermark geometry).
    """
    gltf = sorted(DATA_DIR.glob("models/polyhaven/**/*.gltf"))
    if gltf:
        return gltf
    return sorted(DATA_DIR.glob("models/polyhaven/**/*.blend"))


def discover_texture_dirs() -> list[Path]:
    dirs: list[Path] = []
    for d in DATA_DIR.glob("textures/*/*"):
        if d.is_dir() and (any(d.glob("*.jpg")) or any(d.glob("*.png"))):
            dirs.append(d)
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes) + list(bpy.data.lights) + list(bpy.data.cameras):
        bpy.data.batch_remove([block])


def sample_exposure(rng: random.Random) -> float:
    """Sample HDRI strength with the brightness distribution from scene_generation.md.

    Target split:
      40 % dark   (EV 0.3–0.9)  — low-light is where denoising matters most
      40 % normal (EV 0.9–1.8)
      20 % bright (EV 1.8–2.5)
    """
    bucket = rng.random()
    if bucket < 0.40:
        return rng.uniform(0.3, 0.9)
    elif bucket < 0.80:
        return rng.uniform(0.9, 1.8)
    else:
        return rng.uniform(1.8, 2.5)


def setup_hdri(hdri_path: Path, rng: random.Random):
    """Load HDRI with random Z rotation and biased exposure."""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    coord   = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    env     = nodes.new("ShaderNodeTexEnvironment")
    bg      = nodes.new("ShaderNodeBackground")
    out     = nodes.new("ShaderNodeOutputWorld")

    # Random Z rotation: dominant light source hits from a different azimuth each render
    mapping.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)

    env.image = bpy.data.images.load(str(hdri_path))
    bg.inputs["Strength"].default_value = sample_exposure(rng)

    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"],  env.inputs["Vector"])
    links.new(env.outputs["Color"],       bg.inputs["Color"])
    links.new(bg.outputs["Background"],   out.inputs["Surface"])


# ---------------------------------------------------------------------------
# PBR material builder  (shared by ground plane and model objects)
# ---------------------------------------------------------------------------

def make_pbr_material(name: str, texture_dir: Path, tile: float) -> bpy.types.Material:
    """Build a Principled BSDF fully wired to PBR maps in texture_dir."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    uv      = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    bsdf    = nodes.new("ShaderNodeBsdfPrincipled")
    out     = nodes.new("ShaderNodeOutputMaterial")

    mapping.inputs["Scale"].default_value = (tile, tile, tile)
    links.new(uv.outputs["UV"],     mapping.inputs["Vector"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def try_tex(patterns: list[str], colorspace: str = "Non-Color"):
        for pattern in patterns:
            matches = list(texture_dir.glob(pattern))
            if matches:
                n = nodes.new("ShaderNodeTexImage")
                n.image = bpy.data.images.load(str(matches[0]))
                n.image.colorspace_settings.name = colorspace
                links.new(mapping.outputs["Vector"], n.inputs["Vector"])
                return n
        return None

    color = try_tex(["*diffuse*", "*Diffuse*", "*Color*", "*color*", "*albedo*"], "sRGB")
    rough = try_tex(["*rough*", "*Rough*", "*Roughness*", "*roughness*"])
    nrm   = try_tex(["*nor_gl*", "*NormalGL*", "*normalgl*", "*Normal*", "*normal*", "*_nrm*"])
    metal = try_tex(["*metallic*", "*Metallic*", "*Metalness*", "*metal*"])

    if color: links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    if rough: links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    if nrm:
        nm = nodes.new("ShaderNodeNormalMap")
        links.new(nrm.outputs["Color"], nm.inputs["Color"])
        links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if metal: links.new(metal.outputs["Color"], bsdf.inputs["Metallic"])

    return mat


def add_ground_plane(texture_dir: Path | None, rng: random.Random):
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground"

    if texture_dir is not None:
        tile = rng.uniform(2.0, 8.0)
        mat  = make_pbr_material("Ground", texture_dir, tile)
    else:
        mat = bpy.data.materials.new("Ground_grey")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.4, 0.4, 0.4, 1)
        bsdf.inputs["Roughness"].default_value  = 0.85

    plane.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Model import, placement, and materialisation
# ---------------------------------------------------------------------------

def _import_blend(model_path: Path) -> list:
    try:
        with bpy.data.libraries.load(str(model_path), link=False) as (data_from, data_to):
            data_to.objects = data_from.objects
        objs = [o for o in data_to.objects if o is not None]
        for obj in objs:
            bpy.context.collection.objects.link(obj)
        return objs
    except OSError as e:
        print(f"[scene_gen] WARNING: skipping {model_path.name} (compressed blend): {e}")
        return []


def _import_gltf(model_path: Path) -> list:
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    except Exception as e:
        print(f"[scene_gen] WARNING: GLTF import failed for {model_path.name}: {e}")
        return []
    return [o for o in bpy.data.objects if o not in before]


def import_model(model_path: Path) -> list:
    if model_path.suffix.lower() == ".gltf":
        return _import_gltf(model_path)
    return _import_blend(model_path)


def has_color_texture(objs: list) -> bool:
    """Return True if any mesh has a meaningful sRGB colour texture (> 100 KB).

    Filters out models with no texture or a solid white/uniform diffuse image —
    both cases get a random PBR material from our library instead.
    """
    for obj in objs:
        if obj.type != "MESH":
            continue
        for mat in obj.data.materials:
            if mat is None or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type != "TEX_IMAGE" or node.image is None:
                    continue
                if node.image.colorspace_settings.name != "sRGB":
                    continue
                fp = node.image.filepath_from_user()
                try:
                    if fp and Path(fp).stat().st_size > 100_000:
                        return True
                except OSError:
                    pass
    return False


def snap_to_ground(objs: list):
    """Shift objects upward so the lowest vertex sits exactly at Z = 0."""
    bpy.context.view_layer.update()
    min_z = float("inf")
    for obj in objs:
        if obj.type != "MESH":
            continue
        for v in obj.data.vertices:
            wz = (obj.matrix_world @ v.co).z
            if wz < min_z:
                min_z = wz
    if min_z != float("inf") and abs(min_z) > 1e-4:
        for obj in objs:
            obj.location.z -= min_z
    bpy.context.view_layer.update()


def normalize_scale(objs: list, target_size: float):
    """Uniformly scale a group so its longest bounding-box axis equals target_size."""
    bpy.context.view_layer.update()
    max_dim = 0.0
    for obj in objs:
        if obj.type == "MESH":
            d = obj.dimensions
            max_dim = max(max_dim, d.x, d.y, d.z)
    if max_dim > 1e-3:
        factor = target_size / max_dim
        for obj in objs:
            obj.scale = (obj.scale.x * factor,
                         obj.scale.y * factor,
                         obj.scale.z * factor)
    bpy.context.view_layer.update()


def assign_material(objs: list, mat: bpy.types.Material):
    for obj in objs:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)


def place_objects(model_paths: list[Path], texture_dirs: list[Path], rng: random.Random):
    """Import, scale-normalise, texture, ground-snap, and scatter each model."""
    # Shared scale anchor: all models vary ±30% around this, so they look like
    # they belong in the same scene rather than being independently random sizes.
    scene_target = rng.uniform(0.5, 1.5)

    for i, model_path in enumerate(model_paths):
        objs = import_model(model_path)
        if not objs:
            print(f"[scene_gen] WARNING: no objects imported from {model_path}")
            continue

        target_size = scene_target * rng.uniform(0.7, 1.3)
        normalize_scale(objs, target_size)

        radius = scene_target * rng.uniform(1.0, 2.5) if i > 0 else 0.0
        angle  = rng.uniform(0, 2 * math.pi)
        cx     = radius * math.cos(angle)
        cy     = radius * math.sin(angle)
        for obj in objs:
            obj.location = (cx, cy, obj.location.z)
            obj.rotation_euler.z = rng.uniform(0, 2 * math.pi)

        snap_to_ground(objs)

        # Only replace GLTF materials when they carry no real colour texture
        if texture_dirs and not has_color_texture(objs):
            tex_dir = rng.choice(texture_dirs)
            tile    = rng.uniform(1.0, 4.0)
            mat     = make_pbr_material(f"Model_{i}", tex_dir, tile)
            assign_material(objs, mat)


# ---------------------------------------------------------------------------
# Camera: orbit around scene bounding-box centre, with depth of field
# ---------------------------------------------------------------------------

def scene_bounds() -> tuple[mathutils.Vector, float]:
    """Return (centre, radius) of all non-ground mesh objects."""
    bpy.context.view_layer.update()
    lo = mathutils.Vector(( 1e9,  1e9,  1e9))
    hi = mathutils.Vector((-1e9, -1e9, -1e9))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name == "Ground":
            continue
        for corner in obj.bound_box:
            wco = obj.matrix_world @ mathutils.Vector(corner)
            for k in range(3):
                if wco[k] < lo[k]: lo[k] = wco[k]
                if wco[k] > hi[k]: hi[k] = wco[k]
            found = True
    if not found:
        return mathutils.Vector((0, 0, 0.5)), 1.0
    centre   = (lo + hi) / 2
    centre.z = max(centre.z, 0.3)  # aim at upper half, not the floor
    radius   = max((hi - lo).length / 2, 0.3)
    return centre, radius


def setup_camera(rng: random.Random, target: mathutils.Vector, scene_radius: float):
    """Place camera on a sphere around target with DoF matching real photography.

    Elevation: triangular distribution 15°–55°, mode 25°.
      - Avoids top-down surveillance angles and floor-level shots.
      - Biased toward natural table-top / product photography angles.

    Depth of field: f/2.8–f/11, focused on scene centre.
      - DoF is one of the clearest visual signals that an image came from a
        real camera; without it renders look like video-game screenshots.
    """
    azimuth   = rng.uniform(0, 2 * math.pi)
    elevation = rng.triangular(math.radians(15), math.radians(55), math.radians(25))
    distance  = scene_radius * rng.uniform(2.8, 5.0)

    x = target.x + distance * math.cos(elevation) * math.cos(azimuth)
    y = target.y + distance * math.cos(elevation) * math.sin(azimuth)
    z = max(target.z + distance * math.sin(elevation), 0.3)

    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object

    direction = target - mathutils.Vector((x, y, z))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # Focal length: 35–75 mm equivalent (covers wide-to-portrait range)
    cam.data.lens = rng.uniform(35, 75)

    # Depth of field — focused on scene centre
    cam.data.dof.use_dof          = True
    cam.data.dof.aperture_fstop   = rng.uniform(2.8, 11.0)
    cam.data.dof.focus_distance   = distance

    bpy.context.scene.camera = cam


# ---------------------------------------------------------------------------
# Render configuration and pair rendering
# ---------------------------------------------------------------------------

def setup_render(output_path: str, samples: int, use_denoiser: bool,
                 width: int, height: int):
    """Configure Cycles for one render pass.

    Changing only filepath / samples / denoiser between passes guarantees the
    scene geometry, materials, and camera are identical — i.e. pixel-perfect
    alignment between clean and every noisy variant.
    """
    scene = bpy.context.scene
    scene.render.engine                      = "CYCLES"
    scene.cycles.device                      = "CPU"  # change to "GPU" for CUDA/OptiX
    scene.cycles.samples                     = samples
    scene.cycles.use_denoising              = use_denoiser
    if use_denoiser:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"

    scene.render.resolution_x               = width
    scene.render.resolution_y               = height
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"   # 16-bit preserves full dynamic range
    scene.render.filepath                   = output_path

    # Filmic: matches real camera tone-curve; prevents blown-out highlights that
    # make renders instantly recognisable as CG.
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look           = "None"


def render_pair(output_dir: Path, seed: int,
                samples_clean: int, noise_levels: list[int],
                width: int, height: int):
    """Render one clean image and N noisy variants for the same frozen scene.

    Directory layout:
      output_dir/clean/<seed:08d>.png
      output_dir/noisy_0004spp/<seed:08d>.png
      output_dir/noisy_0016spp/<seed:08d>.png
      output_dir/noisy_0064spp/<seed:08d>.png

    The scene is NOT rebuilt between passes — only sample count and denoiser
    flag change.  This guarantees pixel-perfect alignment for supervised
    denoiser training.
    """
    filename = f"{seed:08d}.png"

    # --- clean pass ---
    clean_dir  = output_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_file = clean_dir / filename

    if not clean_file.exists():
        print(f"[scene_gen] CLEAN ({samples_clean} spp, OIDN ON)  → {clean_file}")
        setup_render(str(clean_file), samples_clean, use_denoiser=True, width=width, height=height)
        bpy.ops.render.render(write_still=True)
    else:
        print(f"[scene_gen] CLEAN already exists, skipping: {clean_file}")

    # --- noisy passes (one per noise level) ---
    for spp in sorted(noise_levels):
        noisy_dir  = output_dir / f"noisy_{spp:04d}spp"
        noisy_dir.mkdir(parents=True, exist_ok=True)
        noisy_file = noisy_dir / filename

        if not noisy_file.exists():
            print(f"[scene_gen] NOISY  ({spp:4d} spp, OIDN OFF) → {noisy_file}")
            setup_render(str(noisy_file), spp, use_denoiser=False, width=width, height=height)
            bpy.ops.render.render(write_still=True)
        else:
            print(f"[scene_gen] NOISY  {spp} spp already exists, skipping: {noisy_file}")


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def seed_from_dir(directory: Path = RENDERS_DIR) -> int:
    """Generate a seed by hashing the names+sizes of files in a directory."""
    if not directory.exists():
        return 0
    hasher = hashlib.md5()
    for entry in sorted(directory.iterdir()):
        stat = entry.stat()
        hasher.update(entry.name.encode())
        hasher.update(str(stat.st_size).encode())
    return int(hasher.hexdigest(), 16) % (2 ** 32)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else seed_from_dir(RENDERS_DIR)
    rng  = random.Random(seed)

    hdris        = discover_hdris()
    models       = discover_models()
    texture_dirs = discover_texture_dirs()

    print(f"[scene_gen] seed={seed}  num_objects={args.num_objects}  "
          f"samples_clean={args.samples_clean}  noise_levels={args.noise_levels}  "
          f"res={args.width}x{args.height}")
    print(f"[scene_gen] assets — {len(hdris)} HDRIs  {len(models)} models  "
          f"{len(texture_dirs)} texture sets")

    if not models:
        print("[scene_gen] ERROR: no models found — run the data pipeline first")
        return

    clear_scene()

    if hdris:
        hdri = rng.choice(hdris)
        print(f"[scene_gen] HDRI: {hdri.name}")
        setup_hdri(hdri, rng)
    else:
        print("[scene_gen] WARNING: no HDRIs found — scene will be unlit")

    chosen = [rng.choice(models) for _ in range(args.num_objects)]
    print(f"[scene_gen] models: {[m.parent.name for m in chosen]}")
    place_objects(chosen, texture_dirs, rng)

    tex_dir = rng.choice(texture_dirs) if texture_dirs else None
    if tex_dir:
        print(f"[scene_gen] ground texture: {tex_dir.name}")
    add_ground_plane(tex_dir, rng)

    target, radius = scene_bounds()
    print(f"[scene_gen] scene centre={tuple(round(v, 2) for v in target)}  radius={radius:.2f}m")
    setup_camera(rng, target, radius)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_pair(
        output_dir    = args.output_dir,
        seed          = seed,
        samples_clean = args.samples_clean,
        noise_levels  = args.noise_levels,
        width         = args.width,
        height        = args.height,
    )

    print("[scene_gen] done")


main()
