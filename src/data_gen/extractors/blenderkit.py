import shutil
from pathlib import Path
from .base import AssetExtractor

# Default BlenderKit local cache path (Linux)
DEFAULT_CACHE = Path.home() / ".local/share/blenderkit_data"


class BlenderKitExtractor(AssetExtractor):
    """
    BlenderKit has no public API for bulk download.
    This extractor reads from the local BlenderKit addon cache
    and copies assets into the project data directory.

    Assets must be downloaded manually via the Blender addon first.
    Cache path: ~/.local/share/blenderkit_data/
    """

    def __init__(self, output_dir: Path, cache_dir: Path = DEFAULT_CACHE):
        self.cache_dir = Path(cache_dir)
        super().__init__(output_dir / "models/blenderkit")

    def fetch_index(self) -> list[str]:
        if not self.cache_dir.exists():
            return []
        # Each asset is a subdirectory in the cache
        return [p.name for p in self.cache_dir.iterdir() if p.is_dir()]

    def download(self, asset_id: str) -> Path | None:
        src = self.cache_dir / asset_id
        dest = self.output_dir / asset_id

        if dest.exists():
            return dest
        if not src.exists():
            return None

        shutil.copytree(src, dest)
        return dest
