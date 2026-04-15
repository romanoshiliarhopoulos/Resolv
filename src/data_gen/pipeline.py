from pathlib import Path
from .extractors import (
    PolyHavenExtractor,
    AmbientCGExtractor,
    SketchfabExtractor,
    BlenderKitExtractor,
)


def run(
    output_dir: Path,
    sketchfab_token: str = None,
    limit: int = None,
    asset_type: str = "all",
):
    output_dir = Path(output_dir)

    all_extractors = {
        "hdris":     lambda: PolyHavenExtractor(output_dir, asset_type="hdris"),
        "textures":  lambda: PolyHavenExtractor(output_dir, asset_type="textures"),
        "models":    lambda: PolyHavenExtractor(output_dir, asset_type="models"),
        "ambientcg": lambda: AmbientCGExtractor(output_dir),
    }

    if asset_type == "all":
        keys = list(all_extractors)
    else:
        keys = [asset_type]

    extractors = [all_extractors[k]() for k in keys]

    if asset_type == "all" and sketchfab_token:
        extractors.append(SketchfabExtractor(output_dir, api_token=sketchfab_token))

    for extractor in extractors:
        name = type(extractor).__name__
        print(f"[{name}] starting...")
        results = extractor.run(limit=limit)
        print(f"[{name}] done — {len(results)} assets")