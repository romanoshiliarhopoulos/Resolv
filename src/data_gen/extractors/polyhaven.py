import requests
from pathlib import Path
from .base import AssetExtractor

BASE_URL = "https://api.polyhaven.com"

# Asset types to fetch and their output subdirectories
ASSET_TYPES = {
    "hdris": "hdri",
    "textures": "textures/polyhaven",
    "models": "models/polyhaven",
}

# Preferred resolution per type
RESOLUTION = {
    "hdris": "4k",
    "textures": "2k",
    "models": "2k",
}

# File keys to pull per texture asset
TEXTURE_KEYS = ["diffuse", "rough", "nor_gl", "disp"]


class PolyHavenExtractor(AssetExtractor):
    def __init__(self, output_dir: Path, asset_type: str = "hdris"):
        assert asset_type in ASSET_TYPES, f"asset_type must be one of {list(ASSET_TYPES)}"
        self.asset_type = asset_type
        self.resolution = RESOLUTION[asset_type]
        super().__init__(output_dir / ASSET_TYPES[asset_type])

    def fetch_index(self) -> list[str]:
        resp = requests.get(f"{BASE_URL}/assets?type={self.asset_type}", timeout=30)
        resp.raise_for_status()
        return list(resp.json().keys())

    def download(self, asset_id: str) -> Path | None:
        asset_dir = self.output_dir / asset_id

        # For models: only skip if a GLTF already exists (not just any file).
        # Old downloads may contain only a compressed .blend that can't be loaded.
        if self.asset_type == "models":
            if (asset_dir / f"{asset_id}.gltf").exists():
                return asset_dir
        elif asset_dir.exists():
            return asset_dir  # already downloaded

        resp = requests.get(f"{BASE_URL}/files/{asset_id}", timeout=30)
        resp.raise_for_status()
        files = resp.json()

        asset_dir.mkdir(parents=True, exist_ok=True)

        if self.asset_type == "hdris":
            url = files.get("hdri", {}).get(self.resolution, {}).get("hdr", {}).get("url")
            if not url:
                return None
            self._fetch_file(url, asset_dir / f"{asset_id}.hdr")

        elif self.asset_type == "textures":
            for key in TEXTURE_KEYS:
                url = files.get(key, {}).get(self.resolution, {}).get("jpg", {}).get("url")
                if url:
                    self._fetch_file(url, asset_dir / f"{key}.jpg")

        elif self.asset_type == "models":
            # Prefer GLTF: portable, uncompressed, PBR textures included.
            # Polyhaven blend files are zstd/gzip-compressed and cannot be
            # loaded via bpy.data.libraries.load() in headless Blender.
            #
            # The GLTF entry looks like:
            #   files["gltf"]["2k"]["gltf"] = {
            #     "url": "...model_2k.gltf",
            #     "include": {
            #       "textures/model_diff_2k.jpg": {"url": "...", ...},
            #       "textures/model_nor_gl_2k.jpg": {"url": "...", ...},
            #       "textures/model_arm_2k.jpg":  {"url": "...", ...},
            #       "model.bin":                   {"url": "...", ...},
            #     }
            #   }
            gltf_info = files.get("gltf", {}).get(self.resolution, {}).get("gltf", {})
            gltf_url  = gltf_info.get("url")

            if gltf_url:
                self._fetch_file(gltf_url, asset_dir / f"{asset_id}.gltf")
                # Download every companion file (textures + .bin) at its
                # relative path so Blender's GLTF importer can find them.
                for rel_path, info in gltf_info.get("include", {}).items():
                    url = info.get("url")
                    if url:
                        dest = asset_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        self._fetch_file(url, dest)
            else:
                # Last resort: blend (may fail to load if compressed)
                blend_url = files.get("blend", {}).get(self.resolution, {}).get("blend", {}).get("url")
                if not blend_url:
                    return None
                self._fetch_file(blend_url, asset_dir / f"{asset_id}.blend")

        return asset_dir

    def _fetch_file(self, url: str, dest: Path):
        if dest.exists():
            return
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
