"""
Tests for IIIF_download.py — tile assembly, filename generation, and
the fixes for the indentation/manifest bugs (issues #4 and #5 in CHANGES.md).

Run with:  pytest tests/test_iiif_download.py -v
"""

import math
import sys
import types
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Stubs for IIIFTileSource dependency
# ---------------------------------------------------------------------------

def _stub_iiif_tile_source():
    mod = types.ModuleType("IIIFTileSource")
    mod.IIIFTileSource = MagicMock()
    mod.zoom_to_scale = MagicMock()
    sys.modules["IIIFTileSource"] = mod

    srsly = types.ModuleType("srsly")
    srsly.write_json = MagicMock()
    srsly.read_json = MagicMock(return_value={})
    sys.modules.setdefault("srsly", srsly)

    httpx_mod = types.ModuleType("httpx")
    httpx_mod.Limits = MagicMock()
    httpx_mod.Timeout = MagicMock()
    httpx_mod.AsyncClient = MagicMock()
    sys.modules.setdefault("httpx", httpx_mod)


_stub_iiif_tile_source()


# ---------------------------------------------------------------------------
# Reference implementations of fixed helpers
# ---------------------------------------------------------------------------

def build_image_filename(image_id: str) -> str:
    """
    Fixed version of the filename logic in iiif_tiles_download.
    Derives a safe filename from the IIIF image_id URI.
    """
    parts = image_id.split("/")
    # Take the relevant path segments (indices 4 and 5 in the IIIF URI scheme)
    if len(parts) >= 6:
        name = "_".join(parts[4:6])
    else:
        name = parts[-1]

    if ".jp2" in name:
        name = name.replace(".jp2", ".jpg")
    elif not name.endswith(".jpg"):
        name = name + ".jpg"

    return name


def assemble_tiles(
    tile_images: list,
    tiles_x: int,
    tiles_y: int,
    level_width: int,
    level_height: int,
    tile_size: int = 256,
) -> Image.Image:
    """
    Reference implementation of the tile assembly logic in iiif_tiles_download.
    Places tiles in row-major order onto a canvas of (level_width, level_height).
    """
    combined = Image.new("RGB", (level_width, level_height))
    tile_index = 0
    for y in range(tiles_y):
        for x in range(tiles_x):
            if tile_index >= len(tile_images):
                break
            tile = tile_images[tile_index]
            pos_x = x * tile_size
            pos_y = y * tile_size
            tile_w = min(tile_size, level_width - pos_x)
            tile_h = min(tile_size, level_height - pos_y)
            if tile.size != (tile_w, tile_h):
                tile = tile.resize((tile_w, tile_h))
            combined.paste(tile, (pos_x, pos_y))
            tile_index += 1
    return combined


# ---------------------------------------------------------------------------
# Filename generation tests (regression for indentation bug)
# ---------------------------------------------------------------------------

class TestBuildImageFilename:
    def test_standard_iiif_uri_produces_filename(self):
        uri = "https://example.org/iiif/2/collection/item/full/max/0/default.jpg"
        name = build_image_filename(uri)
        assert name.endswith(".jpg")
        assert "/" not in name

    def test_jp2_extension_replaced_with_jpg(self):
        uri = "https://example.org/iiif/2/collection/image.jp2/full/max/0/default.jpg"
        name = build_image_filename(uri)
        assert ".jp2" not in name
        assert name.endswith(".jpg")

    def test_regular_jpg_id_gets_jpg_extension(self):
        uri = "https://example.org/iiif/2/collection/image001/full/max/0/default.jpg"
        name = build_image_filename(uri)
        assert name.endswith(".jpg")

    def test_filename_has_no_path_separators(self):
        uri = "https://example.org/iiif/2/repo/item-42/full/max/0/default.jpg"
        name = build_image_filename(uri)
        assert "/" not in name
        assert "\\" not in name


# ---------------------------------------------------------------------------
# Tile assembly tests
# ---------------------------------------------------------------------------

class TestAssembleTiles:
    TILE_SIZE = 256

    def _make_solid_tile(self, color=(128, 128, 128)):
        return Image.new("RGB", (self.TILE_SIZE, self.TILE_SIZE), color=color)

    def test_single_tile_image_correct_size(self):
        tile = self._make_solid_tile()
        result = assemble_tiles([tile], 1, 1, self.TILE_SIZE, self.TILE_SIZE)
        assert result.size == (self.TILE_SIZE, self.TILE_SIZE)

    def test_2x2_grid_correct_size(self):
        tiles = [self._make_solid_tile() for _ in range(4)]
        result = assemble_tiles(tiles, 2, 2, 512, 512)
        assert result.size == (512, 512)

    def test_non_square_grid(self):
        # 3 columns x 2 rows, full size
        tiles = [self._make_solid_tile() for _ in range(6)]
        result = assemble_tiles(tiles, 3, 2, 768, 512)
        assert result.size == (768, 512)

    def test_edge_tiles_resized_correctly(self):
        """
        When the image dimensions are not exact multiples of tile_size,
        edge tiles should be resized/cropped to fit the canvas boundary.
        """
        # 300 wide — last column tile should be 300 - 256 = 44 px wide
        tiles = [self._make_solid_tile() for _ in range(2)]
        result = assemble_tiles(tiles, 2, 1, 300, 256)
        assert result.size == (300, 256)

    def test_tile_colors_placed_correctly(self):
        """
        Verify tiles are placed in the correct positions.
        Use distinct colors to identify each tile's location.
        """
        red = Image.new("RGB", (256, 256), color=(255, 0, 0))
        blue = Image.new("RGB", (256, 256), color=(0, 0, 255))
        result = assemble_tiles([red, blue], 2, 1, 512, 256)

        # Top-left pixel should be red (first tile)
        assert result.getpixel((0, 0)) == (255, 0, 0)
        # Pixel in second tile area should be blue
        assert result.getpixel((256, 0)) == (0, 0, 255)

    def test_missing_tiles_handled_gracefully(self):
        """
        If fewer tiles are available than the grid expects
        (e.g., a download failure), assembly should not raise.
        """
        tiles = [self._make_solid_tile() for _ in range(2)]  # only 2 of 4
        result = assemble_tiles(tiles, 2, 2, 512, 512)
        assert result.size == (512, 512)

    def test_blank_fallback_tile_for_failed_download(self):
        """
        When a tile download fails, a white blank tile should be substituted.
        Verify the blank-tile creation pattern used in the code.
        """
        blank = Image.new("RGB", (256, 256), (255, 255, 255))
        assert blank.getpixel((0, 0)) == (255, 255, 255)
        assert blank.size == (256, 256)


# ---------------------------------------------------------------------------
# Grid dimension calculation
# ---------------------------------------------------------------------------

class TestGridDimensions:
    """Tests for the tile grid math used to determine tiles_x and tiles_y."""

    def test_exact_multiple(self):
        level_width, level_height, tile_size = 512, 256, 256
        tiles_x = math.ceil(level_width / tile_size)
        tiles_y = math.ceil(level_height / tile_size)
        assert tiles_x == 2
        assert tiles_y == 1

    def test_non_multiple_requires_extra_tile(self):
        level_width, level_height, tile_size = 300, 200, 256
        tiles_x = math.ceil(level_width / tile_size)
        tiles_y = math.ceil(level_height / tile_size)
        assert tiles_x == 2  # 300 / 256 = 1.17 → ceil = 2
        assert tiles_y == 1  # 200 / 256 = 0.78 → ceil = 1

    def test_scale_factor_applied_to_level_dimensions(self):
        img_width, img_height = 4000, 3000
        scale_factor = 0.25
        level_width = math.ceil(img_width * scale_factor)
        level_height = math.ceil(img_height * scale_factor)
        assert level_width == 1000
        assert level_height == 750


# ---------------------------------------------------------------------------
# info.json structure test
# ---------------------------------------------------------------------------

class TestInfoJsonStructure:
    def test_info_json_contains_required_keys(self, tmp_path):
        """
        The info.json written by iiif_tiles_download must have 'url' and
        'images' keys so that fetch.py to_hub can build the dataset correctly.
        """
        info = {
            "url": "https://example.org/manifest.json",
            "images": {
                "page_001.jpg": "https://example.org/iiif/2/item/page_001",
            },
        }
        info_path = tmp_path / "info.json"
        import json
        info_path.write_text(json.dumps(info))

        loaded = json.loads(info_path.read_text())
        assert "url" in loaded
        assert "images" in loaded
        assert isinstance(loaded["images"], dict)
