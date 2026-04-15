"""
Resolv pipeline status report.

Shows a live snapshot of available assets and rendered output.

Usage:
  poetry run python -m src.status
  poetry run python -m src.status --warnings   # show only problem rows
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
DATA_DIR     = PROJECT_ROOT / "data"

# Texture maps we expect every complete set to carry
REQUIRED_TEXTURE_MAPS = {"diffuse", "rough"}   # minimum viable PBR
OPTIONAL_TEXTURE_MAPS = {"nor_gl", "displacement", "metallic"}


# ── ANSI colours (disabled if stdout is not a tty) ─────────────────────────

def _color(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _color("32", t)
def yellow(t): return _color("33", t)
def red(t):    return _color("31", t)
def bold(t):   return _color("1",  t)
def dim(t):    return _color("2",  t)


# ── Formatting helpers ──────────────────────────────────────────────────────

def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_size(path: Path) -> int:
    """Recursively sum file sizes under path."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def count_files(path: Path, *patterns: str) -> int:
    if not path.exists():
        return 0
    total = 0
    for pat in patterns:
        total += sum(1 for _ in path.rglob(pat))
    return total


def section(title: str):
    width = 60
    print()
    print(bold(f"{'─' * width}"))
    print(bold(f"  {title}"))
    print(bold(f"{'─' * width}"))


def row(label: str, value: str, status: str = "ok"):
    colour = {"ok": green, "warn": yellow, "error": red, "info": str}.get(status, str)
    print(f"  {label:<36} {colour(value)}")


# ── Asset audit helpers ─────────────────────────────────────────────────────

def audit_texture_source(source_dir: Path) -> dict:
    """Return counts for a single texture source (ambientcg/ or polyhaven/)."""
    if not source_dir.exists():
        return {"total": 0, "complete": 0, "partial": 0, "empty": 0, "size": 0}

    total = complete = partial = empty = 0
    size  = 0

    for td in source_dir.iterdir():
        if not td.is_dir():
            continue
        total += 1
        size  += dir_size(td)

        stems = {f.stem.lower() for f in td.iterdir() if f.is_file()}
        if REQUIRED_TEXTURE_MAPS.issubset(stems):
            complete += 1
        elif stems:
            partial += 1
        else:
            empty += 1

    return {
        "total":    total,
        "complete": complete,
        "partial":  partial,
        "empty":    empty,
        "size":     size,
    }


def audit_models() -> dict:
    """Return per-source model counts."""
    result = {}
    models_dir = DATA_DIR / "models"
    if not models_dir.exists():
        return result

    for source in sorted(models_dir.iterdir()):
        if not source.is_dir():
            continue
        gltf  = count_files(source, "*.gltf")
        blend = count_files(source, "*.blend")
        size  = dir_size(source)
        result[source.name] = {"gltf": gltf, "blend": blend, "size": size}

    return result


def audit_renders() -> dict:
    """Return render counts and pair-completeness stats."""
    renders_dir = DATA_DIR / "renders"
    if not renders_dir.exists():
        return {}

    clean_dir = renders_dir / "clean"
    clean_files = set(f.name for f in clean_dir.glob("*.png")) if clean_dir.exists() else set()

    noisy_dirs = sorted(d for d in renders_dir.iterdir()
                        if d.is_dir() and d.name.startswith("noisy_"))

    noisy_counts  = {}
    missing_pairs = {}  # noisy_dir_name → count of renders missing from clean

    for nd in noisy_dirs:
        noisy_files = set(f.name for f in nd.glob("*.png"))
        noisy_counts[nd.name] = len(noisy_files)
        # renders that exist in this noisy dir but lack a clean counterpart
        orphans = noisy_files - clean_files
        if orphans:
            missing_pairs[nd.name] = len(orphans)

    # clean renders that have no noisy counterpart in ANY noisy dir
    all_noisy = set()
    for nd in noisy_dirs:
        all_noisy |= set(f.name for f in nd.glob("*.png"))
    unpaired_clean = clean_files - all_noisy if noisy_dirs else set()

    return {
        "clean":         len(clean_files),
        "noisy":         noisy_counts,
        "missing_pairs": missing_pairs,
        "unpaired_clean": len(unpaired_clean),
        "size":          dir_size(renders_dir),
    }


# ── Report sections ─────────────────────────────────────────────────────────

def report_hdris():
    section("HDRIs")
    hdri_dir = DATA_DIR / "hdri"
    count = count_files(hdri_dir, "*.hdr", "*.exr")
    size  = dir_size(hdri_dir)

    if count == 0:
        row("Polyhaven HDRIs", "none downloaded", "error")
    else:
        row("Polyhaven HDRIs", f"{count} files", "ok")

    row("Disk usage", fmt_bytes(size), "info")


def report_textures(show_warnings: bool):
    section("Textures")
    tex_dir = DATA_DIR / "textures"

    total_sets = complete = partial = empty = total_size = 0

    for source_name in ("polyhaven", "ambientcg"):
        src = tex_dir / source_name
        s   = audit_texture_source(src)

        total_sets += s["total"]
        complete   += s["complete"]
        partial    += s["partial"]
        empty      += s["empty"]
        total_size += s["size"]

        label = f"{source_name.capitalize()} sets"
        if s["total"] == 0:
            row(label, "none downloaded", "error" if not show_warnings else "error")
        else:
            row(label, f"{s['total']} total  ({s['complete']} complete, "
                       f"{s['partial']} partial, {s['empty']} empty)",
                "ok" if s["partial"] == 0 and s["empty"] == 0 else "warn")

        row(f"  └─ disk usage", fmt_bytes(s["size"]), "info")

    print()
    row("Total texture sets",   str(total_sets), "ok" if total_sets > 0 else "error")
    row("  Complete (diff+rough)", str(complete),  "ok" if complete > 0 else "warn")
    row("  Partial (missing maps)", str(partial),  "ok" if partial == 0 else "warn")
    row("  Empty",                  str(empty),    "ok" if empty == 0 else "warn")
    row("Total disk usage",     fmt_bytes(total_size), "info")

    if (partial > 0 or empty > 0) and show_warnings:
        print()
        print(yellow("  Partial / empty texture sets lack required maps."))
        print(dim (f"  Expected per set: {', '.join(sorted(REQUIRED_TEXTURE_MAPS))}"))
        print(dim (f"  Run: poetry run python -m src.data_gen.run --output data/ --type textures"))


def report_models(show_warnings: bool):
    section("Models")
    info = audit_models()
    total_gltf = total_blend = total_size = 0

    for source, counts in info.items():
        gltf  = counts["gltf"]
        blend = counts["blend"]
        size  = counts["size"]
        total_gltf  += gltf
        total_blend += blend
        total_size  += size

        if gltf > 0:
            status = "ok"
            detail = f"{gltf} GLTF"
            if blend > 0:
                detail += f"  + {blend} blend (ignored)"
        elif blend > 0:
            status = "warn"
            detail = f"{blend} blend only  (compressed — may fail to load)"
        else:
            status = "error"
            detail = "none downloaded"

        row(source.capitalize(), detail, status)
        row(f"  └─ disk usage", fmt_bytes(size), "info")

    print()
    row("Total GLTF (loadable)",    str(total_gltf),  "ok" if total_gltf > 0 else "error")
    row("Total blend (compressed)", str(total_blend), "warn" if total_blend > 0 else "ok")
    row("Total disk usage",         fmt_bytes(total_size), "info")

    if total_gltf == 0 and show_warnings:
        print()
        print(yellow("  No GLTF models found — scenes will have no objects."))
        print(dim (  "  Run: poetry run python -m src.data_gen.run --output data/ --type models"))


def report_renders(show_warnings: bool):
    section("Renders")
    info = audit_renders()

    if not info:
        row("Renders directory", "not found", "error")
        return

    clean_count = info.get("clean", 0)
    noisy       = info.get("noisy", {})
    missing     = info.get("missing_pairs", {})
    unpaired    = info.get("unpaired_clean", 0)
    size        = info.get("size", 0)

    row("Clean renders (ground truth)", str(clean_count),
        "ok" if clean_count > 0 else "warn")

    if noisy:
        for name, count in sorted(noisy.items()):
            spp   = name.replace("noisy_", "").replace("spp", "").lstrip("0") or "0"
            paired = count == clean_count
            row(f"Noisy {spp} spp", str(count),
                "ok" if paired else "warn")
    else:
        row("Noisy renders", "none", "warn")

    print()

    # Pair completeness
    if clean_count > 0 and noisy:
        all_paired = all(c == clean_count for c in noisy.values()) and unpaired == 0
        row("Pairs complete", "yes" if all_paired else "no",
            "ok" if all_paired else "warn")
    if unpaired:
        row("  Unpaired clean renders", str(unpaired), "warn")
    for nd_name, cnt in missing.items():
        row(f"  Orphaned in {nd_name}", str(cnt), "warn")

    # Noise level coverage
    levels = sorted(noisy.keys())
    row("Noise levels available", str(len(levels)) if levels else "none",
        "ok" if len(levels) >= 2 else "warn")
    for lvl in levels:
        print(dim(f"    {lvl}"))

    print()
    row("Total disk usage", fmt_bytes(size), "info")

    if clean_count == 0 and show_warnings:
        print()
        print(yellow("  No renders yet."))
        print(dim(   "  Run: poetry run python -m src.scene_gen.run --num-renders 5"))


def report_summary():
    section("Summary")
    hdri_count  = count_files(DATA_DIR / "hdri", "*.hdr", "*.exr")
    gltf_count  = count_files(DATA_DIR / "models", "*.gltf")
    tex_count   = sum(
        1 for src in (DATA_DIR / "textures").glob("*/*")
        if src.is_dir() and any(src.glob("*.jpg"))
    ) if (DATA_DIR / "textures").exists() else 0
    clean_count = count_files(DATA_DIR / "renders" / "clean", "*.png")
    noisy_dirs  = sorted(
        d for d in (DATA_DIR / "renders").glob("noisy_*") if d.is_dir()
    ) if (DATA_DIR / "renders").exists() else []
    total_size  = dir_size(DATA_DIR)

    row("HDRIs",             str(hdri_count),  "ok" if hdri_count  > 0 else "warn")
    row("GLTF models",       str(gltf_count),  "ok" if gltf_count  > 0 else "warn")
    row("Texture sets",      str(tex_count),   "ok" if tex_count   > 0 else "warn")
    row("Clean renders",     str(clean_count), "ok" if clean_count > 0 else "info")
    row("Noise levels",      str(len(noisy_dirs)), "ok" if noisy_dirs else "info")
    row("Total data on disk", fmt_bytes(total_size), "info")

    # Readiness verdict
    ready = hdri_count > 0 and gltf_count > 0 and tex_count > 0
    print()
    if ready:
        print(green("  Pipeline ready — run scene_gen.run to start rendering."))
    else:
        missing = []
        if hdri_count == 0:  missing.append("HDRIs")
        if gltf_count == 0:  missing.append("GLTF models")
        if tex_count  == 0:  missing.append("textures")
        print(yellow(f"  Not ready — missing: {', '.join(missing)}"))
        print(dim(   "  Run: poetry run python -m src.data_gen.run --output data/"))


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Resolv pipeline status report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--warnings", action="store_true",
        help="Print extra guidance for any problems found",
    )
    args = parser.parse_args()

    print(bold("\nResolv — pipeline status"))
    print(dim(f"  data root: {DATA_DIR}"))

    report_hdris()
    report_textures(args.warnings)
    report_models(args.warnings)
    report_renders(args.warnings)
    report_summary()
    print()


if __name__ == "__main__":
    main()
