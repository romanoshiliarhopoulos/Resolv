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
):
    output_dir = Path(output_dir)
    extractors = [
        PolyHavenExtractor(output_dir, asset_type="hdris"),
        PolyHavenExtractor(output_dir, asset_type="textures"),
        PolyHavenExtractor(output_dir, asset_type="models"),
        AmbientCGExtractor(output_dir),
        BlenderKitExtractor(output_dir),
    ]

    if sketchfab_token:
        extractors.append(SketchfabExtractor(output_dir, api_token=sketchfab_token))

    for extractor in extractors:
        name = type(extractor).__name__
        print(f"[{name}] starting...")
        results = extractor.run(limit=limit)
        print(f"[{name}] done — {len(results)} assets")