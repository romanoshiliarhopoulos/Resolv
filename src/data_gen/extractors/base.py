from abc import ABC, abstractmethod
from pathlib import Path


class AssetExtractor(ABC):
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_index(self) -> list[str]:
        """Return list of all available asset IDs from the source."""

    @abstractmethod
    def download(self, asset_id: str) -> Path | None:
        """Download a single asset. Return its local path, or None if skipped."""

    def run(self, limit: int = None) -> list[Path]:
        ids = self.fetch_index()
        if limit:
            ids = ids[:limit]

        results = []
        for asset_id in ids:
            path = self.download(asset_id)
            if path:
                results.append(path)

        return results
