# Scene Generation Pipeline

## Goal

Produce a large-scale paired image dataset — clean renders and their noisy counterparts — that is diverse and photorealistic enough to train an accurate image denoising model that generalises to real-world photographs.

Every design decision below is driven by two questions:

1. **Does this render look like a real photograph?** (photorealism)
2. **Does this render represent something the model will see in the wild?** (distribution coverage)

---

## Why Synthetic Data Works Here

Camera noise is a physical process: photon shot noise follows a Poisson distribution scaled by scene brightness, and read noise adds a Gaussian floor. Both are fully simulatable. Because we control the renderer, we can produce exact (pixel-aligned, zero-registration-error) clean/noisy pairs — something impossible to acquire from a real camera.

The catch is the **sim-to-real gap**: renders that look fake teach the model fake statistics. The pipeline is specifically designed to close that gap.

---

## Pair Generation Strategy

Each scene is rendered **twice with identical geometry, materials, and camera**. Only the sample count changes:

```
Scene composition (fixed)
  ├── render @ 1024 spp + OIDN denoiser ON  → clean/XXXXX.png   (ground truth)
  └── render @    4 spp + OIDN denoiser OFF → noisy/XXXXX.png   (model input)
```

The low-sample render inherits Monte Carlo variance from Cycles — the same physical noise structure as a real underexposed or high-ISO photograph. This is more realistic than adding synthetic Gaussian noise in post-processing because the noise is spatially correlated, signal-dependent, and affected by geometry (edges, shadows) exactly as it would be on a real sensor.

The OIDN denoiser is applied on the clean pass only to remove any residual variance from the 1024-sample render, giving a truly noiseless ground truth.

---

## Asset Foundation

All assets are CC0 (public domain). No licensing issues for commercial or research use.

| Source | Asset type | Count (approx.) | Why it matters |
|---|---|---|---|
| Poly Haven | HDRIs | ~580 | Photographed real environments → physically accurate global illumination |
| Poly Haven | Models | ~400 | Real-world objects at correct metric scale |
| Poly Haven | Textures | ~300 | Physically measured PBR maps |
| AmbientCG | Textures | ~700 | Additional PBR materials, high variety |

The HDRI library alone provides ~580 distinct lighting environments. Combined with random rotation and exposure, this gives effectively unlimited unique lighting conditions.

---

## File Structure

```
src/
  scene_gen/
    randomize.py      — pure Python, no Blender dependency
    scene_gen.py      — Blender Python script (runs inside bpy)
    render_job.py     — subprocess wrapper, one render pair per call
  orchestrate.py      — batch driver

data/
  hdri/               — .hdr files
  models/polyhaven/   — .blend files
  textures/           — per-material folders (ambientcg/, polyhaven/)
  renders/
    clean/            — ground truth renders
    noisy/            — low-sample renders
    manifest.jsonl    — per-pair metadata
```

---

## File Responsibilities

### `src/scene_gen/randomize.py`

Pure Python (no Blender). Called by the orchestrator to generate a configuration dict for a single render pair. Nothing is random at render time — all randomisation happens here, seeded, so every render is reproducible from its seed.

**Randomises:**

- **HDRI selection** — uniform sample from all `.hdr` files on disk
- **HDRI rotation** — 0°–360° around Z axis; this changes where the dominant light source falls in the scene
- **HDRI exposure** — sampled from a distribution biased toward realistic values (0.3–2.5 EV), with deliberate over-representation of dark scenes (EV < 0.8) because low-light is where denoising matters most and where models fail most often
- **Model selection** — 1 to 4 models drawn without replacement from the full model catalog
- **Ground texture** — random material folder from the texture catalog
- **Camera position** — spherical orbit around scene centre:
  - radius: 2.5–6.0 m (controls how far the camera is from the scene)
  - elevation: 15°–65° (avoids floor-level and top-down shots; both look unnatural)
  - azimuth: 0°–360°
- **Camera FOV** — 35–75 mm equivalent focal length
- **Depth of field** — f/2.8 to f/11, focus distance set to scene centre; provides the background blur present in real photographs
- **Output paths** — `clean/XXXXX.png` and `noisy/XXXXX.png` keyed by render index

Returns a single JSON-serialisable dict. This dict is the complete specification for one render pair and is written verbatim to `manifest.jsonl`.

---

### `src/scene_gen/scene_gen.py`

Runs **inside Blender's Python interpreter** via `blender_utils.run_blender_script()`. Receives the config dict as a JSON string passed after `--` on the command line.

This is where photorealism is enforced. Each function below is responsible for one aspect of the final image quality.

#### `clear_scene()`

Removes all objects, meshes, lights, and cameras from the default Blender scene. Exists already; no changes needed.

#### `setup_hdri(path, rotation_deg, exposure)`

Builds the world shader node tree:

```
TexEnvironment → Mapping (rotation_deg on Z) → Background (strength=exposure) → World Output
```

The `Mapping` node rotates the HDRI so the dominant light source falls at a different azimuth each render. Without this, every render with the same HDRI has identical lighting direction regardless of camera position. The `Background` strength directly controls scene-wide exposure.

**Why this produces photorealism:** Poly Haven HDRIs are latitude-longitude photographs of real environments. They carry accurate radiance, colour temperature, and light distribution. Cycles uses them as an infinite light source via importance sampling, so shadows, bounce light, and sky colour all come for free.

#### `append_model(blend_path, location, rotation_z)`

Loads a `.blend` file and links its objects into the scene. Extends the existing function to:
- Accept an explicit `location` tuple so multiple models can be placed without overlapping
- Accept `rotation_z` so objects face different directions
- Snap the model's lowest vertex to Z=0 so it sits on the ground plane rather than floating or clipping through it

#### `add_ground_plane(texture_dir, tile_scale)`

Creates a 16 m × 16 m ground plane and applies a full PBR material. The material node graph:

```
TexCoord → Mapping (tile_scale) → TexImage (Color, sRGB)    → Principled BSDF Base Color
                                 → TexImage (Roughness)      → Principled BSDF Roughness
                                 → TexImage (NormalGL)       → Normal Map → Principled BSDF Normal
                                 → TexImage (Displacement)   → Displacement → Material Output
```

All four texture maps are wired up. **This is the single most important photorealism fix over the current code**, which only connects the diffuse colour. Without roughness and normal maps, every surface looks like flat painted plastic regardless of what the texture represents.

The `tile_scale` parameter is drawn from randomise.py (range 2–8) so the same texture appears at different spatial frequencies across renders, preventing the model from memorising texture scale.

#### `apply_model_material(obj, texture_dir)`

Applies the same full PBR node graph (above) to imported model objects that have no material or a placeholder material. Models from Poly Haven usually carry their own correct materials; this function is a fallback for models that don't.

#### `setup_camera(position, target, fov_deg, aperture_fstop, focus_distance)`

Places a camera at `position`, points it at `target` using a track-to constraint, sets the focal length from `fov_deg`, and configures depth of field:

```python
cam.data.lens = focal_length_from_fov(fov_deg)
cam.data.dof.use_dof = True
cam.data.dof.aperture_fstop = aperture_fstop
cam.data.dof.focus_distance = focus_distance
```

DoF is one of the clearest visual signals that an image came from a real camera. Without it, renders look like video game screenshots.

#### `setup_render(output_path, samples, use_denoiser, resolution)`

Configures Cycles:

```python
scene.render.engine = "CYCLES"
scene.cycles.samples = samples          # 1024 for clean, 4 for noisy
scene.cycles.use_denoising = use_denoiser  # True for clean, False for noisy
scene.render.image_settings.color_depth = "16"   # 16-bit PNG preserves full dynamic range
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Filmic"    # matches real camera tonecurve
```

**Filmic colour management** is mandatory. The "Standard" view transform clips highlights and makes renders immediately recognisable as CG. Filmic compresses highlights the way film and camera sensors do, matching the tonal response of real photographs.

#### `render_pair(clean_path, noisy_path, samples_clean, samples_noisy)`

Calls `setup_render` and `bpy.ops.render.render(write_still=True)` twice:

1. `samples=samples_clean`, `use_denoiser=True`, output to `clean_path`
2. `samples=samples_noisy`, `use_denoiser=False`, output to `noisy_path`

The scene geometry, materials, camera, and lighting are unchanged between the two renders. The only difference is sample count and denoiser state. This guarantees pixel-perfect alignment between clean and noisy images.

#### `main(config_json)`

Entry point. Parses the JSON config string from `sys.argv`, calls the functions above in order, handles errors with informative messages.

---

### `src/scene_gen/render_job.py`

Called from the orchestrator for each render pair. Sits between the orchestrator and Blender.

**Responsibilities:**

- **Skip check** — if both `clean/XXXXX.png` and `noisy/XXXXX.png` already exist on disk, return immediately without launching Blender. This makes the pipeline fully resumable: kill it at any point and re-run from where it stopped.
- **Config serialisation** — writes the config dict to a temp JSON file, passes its path to `blender_utils.run_blender_script()`
- **Error handling** — captures Blender's exit code; logs failures with the render index and config for debugging without crashing the whole batch
- **Timing** — records wall-clock time per render pair, useful for estimating total dataset generation time

---

### `src/orchestrate.py`

Top-level batch driver. CLI entry point.

```bash
python -m src.orchestrate \
  --data-dir data/ \
  --output-dir data/renders/ \
  --n-renders 5000 \
  --seed 0 \
  --samples-clean 1024 \
  --samples-noisy 4
```

**Responsibilities:**

1. **Catalog scan** — walks `data/hdri/`, `data/models/`, `data/textures/` to build lists of available assets. Validates that texture folders contain at minimum a colour map and a roughness map; skips incomplete folders.
2. **Config generation** — calls `randomize.sample_render_config(seed=base_seed + i, ...)` for each render index. Configs are deterministic: the same `--seed` and `--n-renders` always produce the same sequence.
3. **Dispatch** — calls `render_job.render_pair(config)` for each config. Sequential by default; add `--jobs N` for multiprocessing (each Blender process is independent).
4. **Manifest writing** — appends one JSON line to `data/renders/manifest.jsonl` per successful render pair. Contains the full config plus output paths and render time. This is the dataset index used later by the PyTorch `Dataset` class.

---

## Achieving Distribution Coverage

A model trained on narrow data learns narrow statistics. The following axes must all be varied across the dataset:

| Axis | Mechanism | Why it matters |
|---|---|---|
| **Lighting environment** | 580 HDRIs × random rotation | Covers indoor, outdoor, overcast, sunset, studio, harsh sun |
| **Exposure / brightness** | EV range 0.3–2.5, biased toward dark | Noise is signal-dependent; dark scenes are hardest and most useful |
| **Material type** | ~1000 PBR textures across stone, metal, fabric, wood, plastic, organic | Each material has a different noise signature due to different roughness/specularity |
| **Object type** | ~400 models covering furniture, tools, plants, architecture, vehicles | Structural diversity; different edges, silhouettes, depth variation |
| **Camera distance** | Radius 2.5–6.0 m | Controls spatial frequency of noise relative to image features |
| **Camera angle** | Elevation 15°–65°, azimuth 0°–360° | Prevents angle bias; model must work on any viewpoint |
| **Focal length** | 35–75 mm equivalent | Different perspective distortion and background compression |
| **Depth of field** | f/2.8–f/11 | Trains model to handle blur correctly; blurry regions have different noise characteristics |
| **Scene complexity** | 1–4 models per scene | Sparse and cluttered scenes have different occlusion and shadow patterns |

### Brightness bias

The distribution of HDRI exposure is intentionally skewed toward low-light values. In the real world, denoising is applied most often to photos taken in dim conditions. If the dataset is dominated by well-lit scenes, the model will be under-trained on the cases it will encounter most.

Target distribution (approximate):

- 40% dark scenes (EV 0.3–0.9)
- 40% normal scenes (EV 0.9–1.8)
- 20% bright / high-key scenes (EV 1.8–2.5)

---

## Photorealism Checklist

Before running at scale, verify each of the following on a small test batch:

- [ ] PBR node graph: roughness and normal maps are connected (not just colour)
- [ ] Filmic colour management is active (`bpy.context.scene.view_settings.view_transform == "Filmic"`)
- [ ] Denoiser is OFF for the noisy pass
- [ ] Denoiser is ON for the clean pass
- [ ] Models sit on the ground plane (no floating, no clipping)
- [ ] Depth of field is enabled on the camera
- [ ] HDRI rotation varies between renders of the same environment
- [ ] 16-bit PNG output (preserves full dynamic range for training)
- [ ] Clean and noisy renders are pixel-aligned (same seed, same scene, only sample count differs)

---

## Dataset Output Format

```
data/renders/
  clean/
    00000.png, 00001.png, ...
  noisy/
    00000.png, 00001.png, ...
  manifest.jsonl
```

Each line of `manifest.jsonl` is a complete JSON object:

```json
{
  "index": 42,
  "seed": 42,
  "clean": "data/renders/clean/00042.png",
  "noisy": "data/renders/noisy/00042.png",
  "hdri": "data/hdri/sunflower_patio_4k.hdr",
  "hdri_rotation_deg": 214.7,
  "hdri_exposure": 0.85,
  "models": ["data/models/polyhaven/barrel.blend"],
  "ground_texture": "data/textures/ambientcg/Bricks104",
  "camera_position": [3.1, -2.4, 1.7],
  "camera_target": [0.0, 0.0, 0.5],
  "camera_fov_deg": 52.0,
  "aperture_fstop": 5.6,
  "samples_clean": 1024,
  "samples_noisy": 4,
  "render_time_s": 142.3
}
```

A PyTorch `Dataset` class reads this file, loads image pairs, and optionally filters by any metadata field (e.g. only dark scenes, only outdoor HDRIs) to create targeted training splits.

---

## Scale Targets

| Renders | Disk usage (est.) | Render time (est., CPU) |
|---|---|---|
| 1 000 | ~6 GB | ~40 hours |
| 5 000 | ~30 GB | ~200 hours |
| 20 000 | ~120 GB | ~800 hours |

GPU rendering (CUDA/OptiX) reduces per-render time by 5–10× depending on hardware. The `scene.cycles.device = "CPU"` line in `setup_render` should be changed to `"GPU"` when running on a machine with a supported GPU.
