import requests
import zipfile
import io
from pathlib import Path
from .base import AssetExtractor

BASE_URL = "https://ambientcg.com/api/v2/full_json"
DOWNLOAD_URL = "https://ambientcg.com/get"
PAGE_SIZE = 100
RESOLUTION = "2K"


class AmbientCGExtractor(AssetExtractor):
    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "textures/ambientcg")

    def fetch_index(self) -> list[str]:
        ids = []
        offset = 0
        while True:
            resp = requests.get(
                BASE_URL,
                params={"type": "Material", "sort": "Latest", "limit": PAGE_SIZE, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = [a["assetId"] for a in data.get("foundAssets", [])]
            if not batch:
                break
            ids.extend(batch)
            offset += PAGE_SIZE
        return ids

    def download(self, asset_id: str) -> Path | None:
        asset_dir = self.output_dir / asset_id
        if asset_dir.exists():
            return asset_dir

        url = f"{DOWNLOAD_URL}?file={asset_id}_{RESOLUTION}-JPG.zip"
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None

        asset_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(asset_dir)

        return asset_dir
