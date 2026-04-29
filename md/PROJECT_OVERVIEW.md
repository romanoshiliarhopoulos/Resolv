# Resolv — Project Overview

## Goal

Resolv generates a large synthetic dataset of **clean/noisy image pairs** for training a neural denoiser. Each scene is rendered twice (or more) under identical geometry, materials, camera, and lighting — once at high sample count with OIDN denoising (ground truth) and once or more at low sample counts without denoising (training inputs). The model learns to map noisy Cycles output → clean reference.

---

## Pipeline Architecture

The project has two sequential phases:

```
Phase 1: Asset Downloading       Phase 2: Scene Generation
─────────────────────────────    ─────────────────────────────────────────
src/data_gen/run.py              src/scene_gen/run.py
        │                                 │
        ▼                                 ▼
src/data_gen/pipeline.py         src/scene_gen/scene_gen.py
  ├── PolyHavenExtractor              (runs inside Blender's Python)
  ├── AmbientCGExtractor
  └── SketchfabExtractor
        │
        ▼
    data/
     ├── hdri/          ← 414 environment maps (.hdr/.exr)
     ├── models/        ← 242 PolyHaven models + 15 Sketchfab glTF models
     └── textures/      ← 100 AmbientCG PBR texture sets
```

---

## Phase 1 — Asset Downloading

**Entry point:** `poetry run python -m src.data_gen.run [--limit N]`

Downloads free 3D assets from public APIs into `data/`:

| Source | Format | Content |
|--------|--------|---------|
| PolyHaven | `.blend` (GLTF planned) | HDRIs, PBR textures, 3D models |
| AmbientCG | directory of PBR maps (JPG/PNG) | Tileable surface textures |
| Sketchfab | `scene.gltf` + `scene.bin` + textures | Miscellaneous 3D objects |

Asset counts currently in `data/`: 414 HDRIs, 242 models, 100 texture sets, 15 Sketchfab models.

Requires `SKETCHFAB_API_TOKEN` in `.env` (project root) for Sketchfab downloads.

---

## Phase 2 — Scene Generation

**Entry point:** `poetry run python -m src.scene_gen.run [options]`

The runner (`src/scene_gen/run.py`) calls Blender headlessly in a loop via `src/data_gen/blender_utils.py`. Each iteration launches `src/scene_gen/scene_gen.py` inside Blender's Python interpreter with a set of parameters.

### What `scene_gen.py` does per render

1. **Asset discovery** — scans `data/` for all HDRIs, models, and texture dirs at runtime
2. **HDRI** — picks one randomly; applies random Z-rotation and biased exposure (40% dark / 40% normal / 20% bright)
3. **Objects** — imports N randomly chosen models; normalises each to a shared scene scale (±30% jitter); applies random PBR material from texture library if the model has no colour texture; ground-snaps each model
4. **Ground plane** — 20×20 m plane with a random tiled PBR material (or neutral grey fallback)
5. **Camera** — orbits the scene bounding-box centre; elevation drawn from triangular distribution (15°–55°, mode 25°); random focal length 35–75 mm; depth of field (f/2.8–f/11) focused on scene centre
6. **Render pair** — renders clean + N noisy passes without rebuilding the scene, guaranteeing pixel-perfect alignment

### Output layout

```
data/renders/
 ├── clean/                  ← 1024 spp + OIDN ON  (ground truth)
 │    ├── 00000042.png
 │    └── ...
 ├── noisy_0002spp/          ← 2 spp,  OIDN OFF  (very noisy)
 ├── noisy_0004spp/          ← 4 spp,  OIDN OFF
 ├── noisy_0016spp/          ← 16 spp, OIDN OFF
 └── noisy_0064spp/          ← 64 spp, OIDN OFF
```

Filenames are the zero-padded seed (`{seed:08d}.png`), so clean and noisy files with the same name are guaranteed to be pixel-aligned.

### Runner CLI

```bash
# 10 medium-complexity render pairs (seed 0–9)
poetry run python -m src.scene_gen.run --num-renders 10 --complexity medium

# 5 fully randomised renders (objects, resolution all vary per render)
poetry run python -m src.scene_gen.run --num-renders 5 --randomize

# 1 deterministic render: 4 objects, seed 42
poetry run python -m src.scene_gen.run --seed 42 --num-objects 4

# Custom noise levels only
poetry run python -m src.scene_gen.run --num-renders 10 --noise-levels 4 64

# High-res complex batch
poetry run python -m src.scene_gen.run --num-renders 20 --complexity complex --output data/renders/complex
```

| Flag | Default | Effect |
|------|---------|--------|
| `--num-renders N` | 1 | How many scenes to generate |
| `--complexity simple\|medium\|complex` | medium | Preset for objects/resolution |
| `--num-objects N` | preset | Override objects per scene |
| `--width / --height` | preset | Override render resolution |
| `--samples-clean N` | 1024 | Sample count for clean pass |
| `--noise-levels N [N ...]` | 2 4 16 64 | Sample counts for noisy passes |
| `--randomize` | off | Fully randomise all params per render |
| `--seed N` | random | Base seed; render i gets seed+i |

Complexity presets:

| Preset | Objects | Resolution |
|--------|---------|------------|
| `simple` | 1 | 640×480 |
| `medium` | 3 | 1280×720 |
| `complex` | 6 | 1920×1080 |

---

## Tech Stack

| Component | Version / Detail |
|-----------|-----------------|
| Python | 3.11 (Poetry venv) |
| Blender | 4.2 (headless, at `.blender/blender`) |
| Render engine | Cycles (CPU; switch `scene.cycles.device = "GPU"` for CUDA/OptiX) |
| Denoiser | OpenImageDenoise (OIDN), built into Blender |
| Tone mapping | Filmic (matches real-camera response) |
| Output format | 16-bit PNG |
| Dependency manager | Poetry (`pyproject.toml`) |

---

## Key Source Files

| File | Runs in | Purpose |
|------|---------|---------|
| `src/data_gen/run.py` | Poetry venv | CLI entry for asset downloading |
| `src/data_gen/pipeline.py` | Poetry venv | Orchestrates all extractors |
| `src/data_gen/extractors/` | Poetry venv | Per-source download logic |
| `src/data_gen/blender_utils.py` | Poetry venv | Subprocess bridge to Blender |
| `src/scene_gen/run.py` | Poetry venv | CLI entry for scene generation |
| `src/scene_gen/scene_gen.py` | Blender Python (`bpy`) | Full scene build + render logic |

---

## Dataset Intent

Each `(noisy_XXXXspp, clean)` image pair is one training sample. The dataset is designed for **supervised denoising**: given a low-spp render as input, predict the high-spp+OIDN reference. Scene variety (diverse HDRIs, object types, textures, camera angles, lighting) is maximised through randomisation so the model generalises rather than memorising specific geometries.
