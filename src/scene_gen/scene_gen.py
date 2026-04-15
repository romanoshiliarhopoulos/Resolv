"""
Blender scene generation script.
Runs inside Blender's Python interpreter via blender_utils.run_blender_script().

Arguments (passed after --):
  --output PATH       Where to write the rendered PNG
  --seed INT          Random seed (default: 0)
  --num-objects INT   Number of objects to place (default: 3)
  --samples INT       Cycles render samples (default: 128)
  --width INT         Render width in pixels (default: 1280)
  --height INT        Render height in pixels (default: 720)
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
DATA_DIR = PROJECT_ROOT / "data"
RENDERS_DIR = DATA_DIR / "renders"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
    """Return Polyhaven models, preferring GLTF over blend.

    Polyhaven .blend downloads are zstd/gzip-compressed and cannot be loaded
    via bpy.data.libraries.load() in headless mode.  GLTF files are plain
    JSON + binary — always loadable.

    Sketchfab GLTF models are excluded: they contain watermark geometry.
    """
    gltf = sorted(DATA_DIR.glob("models/polyhaven/**/*.gltf"))
    if gltf:
        return gltf
    # Fall back to blend files if no GLTF has been downloaded yet
    return sorted(DATA_DIR.glob("models/polyhaven/**/*.blend"))


def discover_texture_dirs() -> list[Path]:
    dirs: list[Path] = []
    for d in DATA_DIR.glob("textures/*/*"):
        if d.is_dir() and (any(d.glob("*.jpg")) or any(d.glob("*.png"))):
            dirs.append(d)
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Scene setup helpers
# ---------------------------------------------------------------------------

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes) + list(bpy.data.lights) + list(bpy.data.cameras):
        bpy.data.batch_remove([block])


def setup_hdri(hdri_path: Path, rng: random.Random):
    """Load HDRI with random rotation and exposure."""
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

    # Rotate HDRI around Z so the dominant light source varies per render
    mapping.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)

    env.image = bpy.data.images.load(str(hdri_path))
    bg.inputs["Strength"].default_value = rng.uniform(0.8, 2.0)

    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"],  env.inputs["Vector"])
    links.new(env.outputs["Color"],       bg.inputs["Color"])
    links.new(bg.outputs["Background"],   out.inputs["Surface"])


# ---------------------------------------------------------------------------
# PBR material builder  (shared by ground plane and model objects)
# ---------------------------------------------------------------------------

def make_pbr_material(name: str, texture_dir: Path, tile: float) -> bpy.types.Material:
    """Build a Principled BSDF material fully wired to PBR maps in texture_dir."""
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
    links.new(uv.outputs["UV"],       mapping.inputs["Vector"])
    links.new(bsdf.outputs["BSDF"],   out.inputs["Surface"])

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

    color = try_tex(["*Color*", "*col*", "*Diffuse*", "*diffuse*", "*albedo*"], "sRGB")
    rough = try_tex(["*Roughness*", "*rough*", "*_rgh*"])
    nrm   = try_tex(["*NormalGL*", "*Normal*", "*_nrm*", "*normal*"])
    metal = try_tex(["*Metalness*", "*metallic*", "*Metal*"])

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
        mat = make_pbr_material("Ground", texture_dir, tile)
    else:
        mat = bpy.data.materials.new("Ground_grey")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.4, 0.4, 0.4, 1)

    plane.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Model import, placement, and materialisation
# ---------------------------------------------------------------------------

def _import_blend(model_path: Path) -> list:
    """Append objects from an uncompressed .blend file.

    Compressed blend files (zstd/gzip) produced by Polyhaven will raise an
    OSError here — they cannot be loaded via bpy.data.libraries in headless
    mode.  The caller should fall back to re-downloading as GLTF.
    """
    try:
        with bpy.data.libraries.load(str(model_path), link=False) as (data_from, data_to):
            data_to.objects = data_from.objects
        objs = [o for o in data_to.objects if o is not None]
        for obj in objs:
            bpy.context.collection.objects.link(obj)
        return objs
    except OSError as e:
        print(f"[scene_gen] WARNING: skipping {model_path.name} (compressed blend — re-download as GLTF): {e}")
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
    suffix = model_path.suffix.lower()
    if suffix == ".gltf":
        return _import_gltf(model_path)
    return _import_blend(model_path)


def has_valid_materials(objs: list) -> bool:
    """Return True if any mesh object already has a material with image textures.

    Polyhaven GLTF models carry their own PBR materials — we keep those and
    only apply a fallback material when the model has none.
    """
    for obj in objs:
        if obj.type != "MESH":
            continue
        for mat in obj.data.materials:
            if mat is None or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    return True
    return False


def snap_to_ground(objs: list):
    """Shift a group of objects upward so the lowest vertex sits at Z = 0."""
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
    """Replace all materials on mesh objects with mat."""
    for obj in objs:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)


def place_objects(model_paths: list[Path], texture_dirs: list[Path], rng: random.Random):
    """Import, scale-normalise, texture, ground-snap and scatter each model."""
    for i, model_path in enumerate(model_paths):
        objs = import_model(model_path)
        if not objs:
            print(f"[scene_gen] WARNING: no objects imported from {model_path}")
            continue

        # --- scale: randomise target size between 0.4 m and 2.0 m ---
        target_size = rng.uniform(0.4, 2.0)
        normalize_scale(objs, target_size)

        # --- position: scatter on a disc around origin ---
        radius = rng.uniform(0.3, 2.0) if i > 0 else 0.0
        angle  = rng.uniform(0, 2 * math.pi)
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        for obj in objs:
            obj.location = (cx, cy, obj.location.z)
            obj.rotation_euler.z = rng.uniform(0, 2 * math.pi)

        # --- snap to ground AFTER repositioning ---
        snap_to_ground(objs)

        # --- materialise ---
        # GLTF models from Polyhaven carry their own PBR materials — keep them.
        # If the model has no image textures (e.g. bare blend geometry), apply
        # a random material from our texture library as a fallback.
        if texture_dirs and not has_valid_materials(objs):
            tex_dir = rng.choice(texture_dirs)
            tile    = rng.uniform(1.0, 4.0)
            mat     = make_pbr_material(f"Model_{i}", tex_dir, tile)
            assign_material(objs, mat)


# ---------------------------------------------------------------------------
# Camera: aim at scene bounding-box centre
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
    centre = (lo + hi) / 2
    # Aim slightly above the true centre so we see the top half of objects
    centre.z = max(centre.z, 0.3)
    radius = max((hi - lo).length / 2, 0.3)
    return centre, radius


def setup_camera(rng: random.Random, target: mathutils.Vector, scene_radius: float):
    """Place camera on a sphere around target, guaranteeing objects stay in frame."""
    azimuth   = rng.uniform(0, 2 * math.pi)
    elevation = rng.uniform(math.radians(20), math.radians(55))
    # Keep a minimum clearance so the camera is never inside an object
    distance  = scene_radius * rng.uniform(2.8, 5.0)

    x = target.x + distance * math.cos(elevation) * math.cos(azimuth)
    y = target.y + distance * math.cos(elevation) * math.sin(azimuth)
    z = target.z + distance * math.sin(elevation)

    # Clamp so camera never goes below the ground plane
    z = max(z, 0.3)

    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object

    direction = target - mathutils.Vector((x, y, z))
    rot = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot.to_euler()
    cam.data.lens = rng.uniform(35, 70)  # focal length in mm

    bpy.context.scene.camera = cam


# ---------------------------------------------------------------------------
# Render configuration
# ---------------------------------------------------------------------------

def setup_render(output_path: str, samples: int, width: int, height: int):
    scene = bpy.context.scene
    scene.render.engine                       = "CYCLES"
    scene.cycles.device                       = "CPU"
    scene.cycles.samples                      = samples
    scene.cycles.use_denoising               = True
    scene.render.resolution_x                = width
    scene.render.resolution_y                = height
    scene.render.image_settings.file_format  = "PNG"
    scene.render.image_settings.color_depth  = "16"
    scene.render.filepath                    = output_path
    # Filmic: matches real camera tonecurve; prevents blown-out highlights
    scene.view_settings.view_transform       = "Filmic"
    scene.view_settings.look                 = "None"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def seed_from_dir(directory: Path = RENDERS_DIR) -> int:
    """Generate a seed by hashing the names + sizes of files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return 0

    hasher = hashlib.md5()
    for entry in sorted(directory.iterdir()):          # sorted = deterministic order
        stat = entry.stat()
        hasher.update(entry.name.encode())
        hasher.update(str(stat.st_size).encode())

    return int(hasher.hexdigest(), 16) % (2**32)

def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else seed_from_dir(RENDERS_DIR)

    rng  = random.Random(seed)
    
    output_path = args.output
    if output_path is None:
        RENDERS_DIR.mkdir(parents=True, exist_ok=True)

        output_path = str(RENDERS_DIR / f"render_{seed:08d}.png")

    hdris        = discover_hdris()
    models       = discover_models()
    texture_dirs = discover_texture_dirs()

    print(f"[scene_gen] seed={args.seed}  num_objects={args.num_objects}  "
          f"samples={args.samples}  res={args.width}x{args.height}")
    print(f"[scene_gen] assets — {len(hdris)} HDRIs  {len(models)} models  "
          f"{len(texture_dirs)} texture sets")

    if not models:
        print("[scene_gen] ERROR: no models found — run the data pipeline first")
        return

    clear_scene()

    if hdris:
        hdri = rng.choice(hdris)
        print(f"[scene_gen] HDRI: {hdri.parent.name}/{hdri.name}")
        setup_hdri(hdri, rng)
    else:
        print("[scene_gen] WARNING: no HDRIs — scene will be unlit")

    chosen = [rng.choice(models) for _ in range(args.num_objects)]
    print(f"[scene_gen] models: {[m.parent.name for m in chosen]}")
    place_objects(chosen, texture_dirs, rng)

    tex_dir = rng.choice(texture_dirs) if texture_dirs else None
    if tex_dir:
        print(f"[scene_gen] ground texture: {tex_dir.name}")
    add_ground_plane(tex_dir, rng)

    target, radius = scene_bounds()
    print(f"[scene_gen] scene centre={tuple(round(v,2) for v in target)}  radius={radius:.2f}m")
    setup_camera(rng, target, radius)

    setup_render(output_path, args.samples, args.width, args.height)

    print(f"[scene_gen] rendering → {output_path}")
    bpy.ops.render.render(write_still=True)
    print("[scene_gen] done")


main()
