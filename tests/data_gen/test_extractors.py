import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_gen.extractors.polyhaven import PolyHavenExtractor
from src.data_gen.extractors.ambientcg import AmbientCGExtractor
from src.data_gen.extractors.sketchfab import SketchfabExtractor
from src.data_gen.extractors.blenderkit import BlenderKitExtractor


# --- Helpers ---

def make_response(json_data=None, content=b"", status=200, content_type="application/octet-stream"):
    mock = MagicMock()
    mock.status_code = status
    mock.content = content
    mock.headers = {"Content-Type": content_type}
    mock.json.return_value = json_data or {}
    mock.iter_content = lambda chunk_size: [content]
    mock.raise_for_status = MagicMock()
    return mock


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


# --- PolyHaven ---

class TestPolyHavenExtractor:
    def test_fetch_index_returns_asset_ids(self, tmp_path):
        extractor = PolyHavenExtractor(tmp_path, asset_type="hdris")
        index_response = make_response(json_data={"sky_01": {}, "forest_02": {}})

        with patch("src.data_gen.extractors.polyhaven.requests.get", return_value=index_response):
            ids = extractor.fetch_index()

        assert set(ids) == {"sky_01", "forest_02"}

    def test_download_hdri_creates_file(self, tmp_path):
        extractor = PolyHavenExtractor(tmp_path, asset_type="hdris")
        files_response = make_response(json_data={
            "hdri": {"4k": {"hdr": {"url": "https://example.com/sky_01_4k.hdr"}}}
        })
        file_response = make_response(content=b"hdr_data")

        with patch("src.data_gen.extractors.polyhaven.requests.get", side_effect=[files_response, file_response]):
            result = extractor.download("sky_01")

        assert result is not None
        assert (result / "sky_01.hdr").read_bytes() == b"hdr_data"

    def test_download_skips_existing(self, tmp_path):
        extractor = PolyHavenExtractor(tmp_path, asset_type="hdris")
        existing = extractor.output_dir / "sky_01"
        existing.mkdir(parents=True)

        with patch("src.data_gen.extractors.polyhaven.requests.get") as mock_get:
            result = extractor.download("sky_01")
            mock_get.assert_not_called()

        assert result == existing

    def test_download_returns_none_on_missing_url(self, tmp_path):
        extractor = PolyHavenExtractor(tmp_path, asset_type="hdris")
        files_response = make_response(json_data={"hdri": {}})  # no URL

        with patch("src.data_gen.extractors.polyhaven.requests.get", return_value=files_response):
            result = extractor.download("sky_01")

        assert result is None

    def test_fetch_index_textures(self, tmp_path):
        extractor = PolyHavenExtractor(tmp_path, asset_type="textures")
        index_response = make_response(json_data={"concrete_01": {}, "wood_floor": {}})

        with patch("src.data_gen.extractors.polyhaven.requests.get", return_value=index_response):
            ids = extractor.fetch_index()

        assert "concrete_01" in ids and "wood_floor" in ids


# --- AmbientCG ---

class TestAmbientCGExtractor:
    def test_fetch_index_paginates(self, tmp_path):
        extractor = AmbientCGExtractor(tmp_path)

        page1 = make_response(json_data={"foundAssets": [{"assetId": "Concrete001"}, {"assetId": "Wood002"}]})
        page2 = make_response(json_data={"foundAssets": []})  # signals end of pagination

        with patch("src.data_gen.extractors.ambientcg.requests.get", side_effect=[page1, page2]):
            ids = extractor.fetch_index()

        assert ids == ["Concrete001", "Wood002"]

    def test_download_extracts_zip(self, tmp_path):
        extractor = AmbientCGExtractor(tmp_path)
        zip_content = make_zip({"Color.jpg": b"color", "Roughness.jpg": b"rough"})
        zip_response = make_response(content=zip_content, status=200)

        with patch("src.data_gen.extractors.ambientcg.requests.get", return_value=zip_response):
            result = extractor.download("Concrete001")

        assert result is not None
        assert (result / "Color.jpg").read_bytes() == b"color"
        assert (result / "Roughness.jpg").read_bytes() == b"rough"

    def test_download_skips_existing(self, tmp_path):
        extractor = AmbientCGExtractor(tmp_path)
        existing = extractor.output_dir / "Concrete001"
        existing.mkdir(parents=True)

        with patch("src.data_gen.extractors.ambientcg.requests.get") as mock_get:
            result = extractor.download("Concrete001")
            mock_get.assert_not_called()

        assert result == existing

    def test_download_returns_none_on_404(self, tmp_path):
        extractor = AmbientCGExtractor(tmp_path)
        bad_response = make_response(status=404)

        with patch("src.data_gen.extractors.ambientcg.requests.get", return_value=bad_response):
            result = extractor.download("NonExistent")

        assert result is None


# --- Sketchfab ---

class TestSketchfabExtractor:
    def test_fetch_index_paginates(self, tmp_path):
        extractor = SketchfabExtractor(tmp_path, api_token="test_token")

        page1 = make_response(json_data={"results": [{"uid": "abc123"}, {"uid": "def456"}], "next": "page2_url"})
        page2 = make_response(json_data={"results": [{"uid": "ghi789"}], "next": None})

        with patch.object(extractor.session, "get", side_effect=[page1, page2]):
            ids = extractor.fetch_index()

        assert ids == ["abc123", "def456", "ghi789"]

    def test_download_extracts_zip(self, tmp_path):
        extractor = SketchfabExtractor(tmp_path, api_token="test_token")
        zip_content = make_zip({"scene.gltf": b"gltf_data", "scene.bin": b"bin_data"})

        download_response = make_response(json_data={"gltf": {"url": "https://example.com/model.zip"}})
        file_response = make_response(content=zip_content, content_type="application/zip")

        with patch.object(extractor.session, "get", return_value=download_response):
            with patch("src.data_gen.extractors.sketchfab.requests.get", return_value=file_response):
                result = extractor.download("abc123")

        assert result is not None
        assert (result / "scene.gltf").exists()

    def test_download_skips_existing(self, tmp_path):
        extractor = SketchfabExtractor(tmp_path, api_token="test_token")
        existing = extractor.output_dir / "abc123"
        existing.mkdir(parents=True)

        with patch.object(extractor.session, "get") as mock_get:
            result = extractor.download("abc123")
            mock_get.assert_not_called()

        assert result == existing

    def test_download_returns_none_on_failed_request(self, tmp_path):
        extractor = SketchfabExtractor(tmp_path, api_token="test_token")
        bad_response = make_response(status=403)

        with patch.object(extractor.session, "get", return_value=bad_response):
            result = extractor.download("abc123")

        assert result is None

    def test_download_returns_none_when_no_url(self, tmp_path):
        extractor = SketchfabExtractor(tmp_path, api_token="test_token")
        empty_response = make_response(json_data={})  # no gltf or source key

        with patch.object(extractor.session, "get", return_value=empty_response):
            result = extractor.download("abc123")

        assert result is None


# --- BlenderKit ---

class TestBlenderKitExtractor:
    def test_fetch_index_returns_empty_when_no_cache(self, tmp_path):
        extractor = BlenderKitExtractor(tmp_path, cache_dir=tmp_path / "nonexistent_cache")
        assert extractor.fetch_index() == []

    def test_fetch_index_lists_cache_dirs(self, tmp_path):
        cache = tmp_path / "bk_cache"
        (cache / "asset_a").mkdir(parents=True)
        (cache / "asset_b").mkdir(parents=True)
        (cache / "file.txt").write_text("not a dir")  # should be ignored

        extractor = BlenderKitExtractor(tmp_path, cache_dir=cache)
        ids = extractor.fetch_index()

        assert set(ids) == {"asset_a", "asset_b"}

    def test_download_copies_from_cache(self, tmp_path):
        cache = tmp_path / "bk_cache"
        src = cache / "asset_a"
        src.mkdir(parents=True)
        (src / "model.blend").write_bytes(b"blend_data")

        extractor = BlenderKitExtractor(tmp_path, cache_dir=cache)
        result = extractor.download("asset_a")

        assert result is not None
        assert (result / "model.blend").read_bytes() == b"blend_data"

    def test_download_skips_existing(self, tmp_path):
        cache = tmp_path / "bk_cache"
        src = cache / "asset_a"
        src.mkdir(parents=True)

        extractor = BlenderKitExtractor(tmp_path, cache_dir=cache)
        dest = extractor.output_dir / "asset_a"
        dest.mkdir(parents=True)

        result = extractor.download("asset_a")
        assert result == dest

    def test_download_returns_none_when_not_in_cache(self, tmp_path):
        cache = tmp_path / "bk_cache"
        cache.mkdir()

        extractor = BlenderKitExtractor(tmp_path, cache_dir=cache)
        result = extractor.download("missing_asset")

        assert result is None
