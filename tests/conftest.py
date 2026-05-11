"""
Shared pytest fixtures for the vlms-on-hpc test suite.
"""

import io
import pytest
from PIL import Image


@pytest.fixture
def fake_pil_image():
    """A small RGB PIL image suitable for testing make_ocr_message."""
    return Image.new("RGB", (100, 80), color=(200, 100, 50))


@pytest.fixture
def fake_pdf_page(fake_pil_image):
    """A page dict in the format produced by pymupdf in main_pdf.py."""
    return {"image": fake_pil_image, "page": 1}


@pytest.fixture
def image_bytes(fake_pil_image):
    """PNG bytes of a PIL image, for testing the dict-with-bytes path."""
    buf = io.BytesIO()
    fake_pil_image.save(buf, format="PNG")
    return buf.getvalue()
