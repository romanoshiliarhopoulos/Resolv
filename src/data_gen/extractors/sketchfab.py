import requests
import zipfile
import io
from pathlib import Path
from .base import AssetExtractor

BASE_URL = "https://api.sketchfab.com/v3"
PAGE_SIZE = 24


class SketchfabExtractor(AssetExtractor):
    def __init__(self, output_dir: Path, api_token: str):
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {api_token}"})
        super().__init__(output_dir / "models/sketchfab")

    def fetch_index(self) -> list[str]:
        ids = []
        url = f"{BASE_URL}/models"
        params = {"downloadable": "true", "count": PAGE_SIZE}

        while url:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            ids.extend(m["uid"] for m in data.get("results", []))
            url = data.get("next")
            params = {}  # next URL already contains pagination params

        return ids

    def download(self, asset_id: str) -> Path | None:
        asset_dir = self.output_dir / asset_id
        if asset_dir.exists():
            return asset_dir

        resp = self.session.get(f"{BASE_URL}/models/{asset_id}/download", timeout=30)
        if resp.status_code != 200:
            return None

        data = resp.json()
        url = data.get("gltf", {}).get("url") or data.get("source", {}).get("url")
        if not url:
            return None

        file_resp = requests.get(url, stream=True, timeout=120)
        if file_resp.status_code != 200:
            return None

        asset_dir.mkdir(parents=True, exist_ok=True)
        content = file_resp.content

        if file_resp.headers.get("Content-Type", "").startswith("application/zip") or url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(asset_dir)
        else:
            suffix = ".gltf" if "gltf" in url else ".glb"
            (asset_dir / f"{asset_id}{suffix}").write_bytes(content)

        return asset_dir
