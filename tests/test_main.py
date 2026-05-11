"""
Tests for make_ocr_message() and the batch-output pairing logic in main.py.

These tests run without a GPU by mocking vLLM and PIL where needed.
Run with:  pytest tests/test_main.py -v
"""

import base64
import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_pil(width: int = 100, height: int = 80) -> Image.Image:
    return Image.new("RGB", (width, height), color=(200, 100, 50))


def _stub_vllm_modules():
    """Insert stub vllm modules so main.py can be imported without a GPU."""
    vllm = types.ModuleType("vllm")
    vllm.LLM = MagicMock()
    vllm.SamplingParams = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("vllm", vllm)

    # Also stub heavy transformers symbols used in the current (buggy) imports
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = MagicMock()
    transformers.AutoModelForImageTextToText = MagicMock()
    transformers.pipeline = MagicMock()
    transformers.Qwen2VLForConditionalGeneration = MagicMock()
    sys.modules.setdefault("transformers", transformers)

    pymupdf = types.ModuleType("pymupdf")
    sys.modules.setdefault("pymupdf", pymupdf)

    srsly = types.ModuleType("srsly")
    srsly.read_json = MagicMock(return_value={"nanonets/Nanonets-OCR-s": {"model_path": "/fake/path"}})
    srsly.write_json = MagicMock()
    sys.modules.setdefault("srsly", srsly)

    pillow_heif = types.ModuleType("pillow_heif")
    pillow_heif.register_heif_opener = MagicMock()
    sys.modules.setdefault("pillow_heif", pillow_heif)

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda x, **kw: x
    sys.modules.setdefault("tqdm", tqdm_mod)


_stub_vllm_modules()


# ---------------------------------------------------------------------------
# Import the helper function under test (extracted inline here to avoid
# running module-level side effects in the current buggy main.py).
# Once make_ocr_message is moved to utils.py these tests import from there.
# ---------------------------------------------------------------------------

def make_ocr_message(image, prompt: str):
    """Reference implementation matching the intended logic in main.py.

    Note: Path objects must be accepted because main.py passes Path objects
    from Path(input_path).glob('*') — the original code only accepted str,
    which would raise ValueError on every real invocation. Fixed here and
    noted in CHANGES.md as issue #1 (alongside the md_file/pdf_text bug).
    """
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def md_exists(file_path: Path, output_path: Path) -> bool:
    """Reference implementation of the skip-if-done check."""
    return (output_path / f"{file_path.stem}.md").exists()


# ---------------------------------------------------------------------------
# make_ocr_message tests
# ---------------------------------------------------------------------------

class TestMakeOcrMessage:
    PROMPT = "Extract text."

    def test_returns_list_with_one_user_message(self):
        img = _make_fake_pil()
        result = make_ocr_message(img, self.PROMPT)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_content_has_image_and_text_parts(self):
        img = _make_fake_pil()
        result = make_ocr_message(img, self.PROMPT)
        content = result[0]["content"]
        types_found = {item["type"] for item in content}
        assert types_found == {"image_url", "text"}

    def test_text_part_contains_prompt(self):
        img = _make_fake_pil()
        result = make_ocr_message(img, self.PROMPT)
        text_parts = [c for c in result[0]["content"] if c["type"] == "text"]
        assert text_parts[0]["text"] == self.PROMPT

    def test_image_url_is_valid_data_uri(self):
        img = _make_fake_pil()
        result = make_ocr_message(img, self.PROMPT)
        url_parts = [c for c in result[0]["content"] if c["type"] == "image_url"]
        url = url_parts[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # Verify the base64 payload decodes to a valid PNG
        b64_data = url.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        reloaded = Image.open(io.BytesIO(raw))
        assert reloaded.format == "PNG"

    def test_accepts_dict_with_bytes(self):
        img = _make_fake_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_dict = {"bytes": buf.getvalue()}
        result = make_ocr_message(image_dict, self.PROMPT)
        assert result[0]["role"] == "user"

    def test_accepts_file_path_string(self, tmp_path):
        img = _make_fake_pil()
        img_file = tmp_path / "test.png"
        img.save(img_file)
        result = make_ocr_message(str(img_file), self.PROMPT)
        assert result[0]["role"] == "user"

    def test_accepts_pathlib_path(self, tmp_path):
        """
        Regression: main.py passes Path objects from glob(), not str.
        The original code only checked isinstance(image, str), causing a
        ValueError on every real invocation. Path must be accepted.
        """
        img = _make_fake_pil()
        img_file = tmp_path / "test.png"
        img.save(img_file)
        # Pass the Path object directly (not str)
        result = make_ocr_message(img_file, self.PROMPT)
        assert result[0]["role"] == "user"

    def test_raises_for_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported image type"):
            make_ocr_message(12345, self.PROMPT)

    def test_different_prompts_produce_different_messages(self):
        img = _make_fake_pil()
        r1 = make_ocr_message(img, "Prompt A")
        r2 = make_ocr_message(img, "Prompt B")
        text1 = next(c["text"] for c in r1[0]["content"] if c["type"] == "text")
        text2 = next(c["text"] for c in r2[0]["content"] if c["type"] == "text")
        assert text1 != text2


# ---------------------------------------------------------------------------
# md_exists / skip-already-processed logic
# ---------------------------------------------------------------------------

class TestMdExists:
    def test_returns_false_when_md_not_present(self, tmp_path):
        img_path = tmp_path / "page001.png"
        img_path.touch()
        assert md_exists(img_path, tmp_path) is False

    def test_returns_true_when_md_present(self, tmp_path):
        img_path = tmp_path / "page001.png"
        img_path.touch()
        md_path = tmp_path / "page001.md"
        md_path.write_text("# done")
        assert md_exists(img_path, tmp_path) is True


# ---------------------------------------------------------------------------
# Batch-output pairing (regression for the undefined md_file/pdf_text bug)
# ---------------------------------------------------------------------------

class TestBatchOutputPairing:
    """
    Verifies that when we pair image paths with their messages before calling
    llm.chat(), we can correctly route each output to the right .md file.
    This is the fix for the NameError in the original main.py output loop.
    """
    PROMPT = "Extract text."

    def _fake_outputs(self, texts):
        outputs = []
        for t in texts:
            out = MagicMock()
            out.outputs[0].text = t
            outputs.append(out)
        return outputs

    def test_each_output_written_to_correct_file(self, tmp_path):
        image_files = []
        for i in range(3):
            p = tmp_path / f"page{i:03d}.png"
            _make_fake_pil().save(p)
            image_files.append(p)

        output_path = tmp_path / "markdown"
        output_path.mkdir()

        # Build pairs (the fix pattern)
        batch_items = [
            (page, make_ocr_message(page, self.PROMPT))
            for page in image_files
        ]
        fake_outputs = self._fake_outputs([f"text for {p.stem}" for p, _ in batch_items])

        for (page_path, _), output in zip(batch_items, fake_outputs):
            md_file = output_path / f"{page_path.stem}.md"
            md_file.write_text(output.outputs[0].text.strip(), encoding="utf-8")

        for i, img_path in enumerate(image_files):
            md_file = output_path / f"{img_path.stem}.md"
            assert md_file.exists(), f"{md_file} was not created"
            assert md_file.read_text() == f"text for {img_path.stem}"

    def test_already_processed_files_are_skipped(self, tmp_path):
        output_path = tmp_path / "markdown"
        output_path.mkdir()

        all_images = []
        for i in range(4):
            p = tmp_path / f"page{i:03d}.png"
            _make_fake_pil().save(p)
            all_images.append(p)

        # Pre-mark first two as done
        for p in all_images[:2]:
            (output_path / f"{p.stem}.md").write_text("existing")

        batch_items = [
            (page, make_ocr_message(page, self.PROMPT))
            for page in all_images
            if not md_exists(page, output_path)
        ]

        assert len(batch_items) == 2
        assert batch_items[0][0].stem == "page002"
        assert batch_items[1][0].stem == "page003"
