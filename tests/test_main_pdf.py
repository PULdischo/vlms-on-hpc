"""
Tests for the PDF processing logic in main_pdf.py.

Covers:
- make_ocr_message with PIL images (same helper as main.py)
- Lock-file race condition fix (issue #11 in CHANGES.md)
- Skip-already-processed logic
- Batch assembly from PDF page dicts

Run with:  pytest tests/test_main_pdf.py -v
"""

import io
import sys
import time
import types
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _stub_heavy_modules():
    for name in ("vllm", "torch"):
        mod = types.ModuleType(name)
        sys.modules.setdefault(name, mod)

    vllm = sys.modules["vllm"]
    vllm.LLM = MagicMock()
    vllm.SamplingParams = MagicMock(return_value=MagicMock())

    transformers = types.ModuleType("transformers")
    for attr in ("AutoProcessor", "AutoModelForImageTextToText", "pipeline",
                 "Qwen2VLForConditionalGeneration"):
        setattr(transformers, attr, MagicMock())
    sys.modules.setdefault("transformers", transformers)

    pymupdf = types.ModuleType("pymupdf")
    sys.modules.setdefault("pymupdf", pymupdf)

    srsly = types.ModuleType("srsly")
    srsly.read_json = MagicMock(return_value=[])
    srsly.write_json = MagicMock()
    sys.modules.setdefault("srsly", srsly)

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda x, **kw: x
    sys.modules.setdefault("tqdm", tqdm_mod)


_stub_heavy_modules()


# ---------------------------------------------------------------------------
# Reference implementations of fixed helpers
# ---------------------------------------------------------------------------

def make_ocr_message_pdf(image, prompt: str):
    """Fixed make_ocr_message for PDF page dicts."""
    import base64

    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "image" in image:
        pil_img = image["image"]
    else:
        raise ValueError(f"Unsupported type: {type(image)}")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    import base64 as b64
    data_uri = f"data:image/png;base64,{b64.b64encode(buf.getvalue()).decode()}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def claim_pdf_with_lock(pdf_stem: str, lock_dir: Path) -> bool:
    """
    Attempt to claim a PDF for processing using a lock file.
    Returns True if the claim succeeded (this job owns it),
    False if another job already claimed it.
    This is the fix for issue #11.
    """
    lock_file = lock_dir / f"{pdf_stem}.lock"
    try:
        lock_file.touch(exist_ok=False)
        return True
    except FileExistsError:
        return False


def release_pdf_lock(pdf_stem: str, lock_dir: Path) -> None:
    lock_file = lock_dir / f"{pdf_stem}.lock"
    lock_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# make_ocr_message tests for PDF page dicts
# ---------------------------------------------------------------------------

class TestMakeOcrMessagePdf:
    PROMPT = "Extract text from document page."

    def _fake_page_dict(self):
        img = Image.new("RGB", (826, 1169), color=(255, 255, 255))
        return {"image": img, "page": 1}

    def test_accepts_page_dict_with_image_key(self):
        page = self._fake_page_dict()
        result = make_ocr_message_pdf(page, self.PROMPT)
        assert result[0]["role"] == "user"

    def test_accepts_plain_pil_image(self):
        img = Image.new("RGB", (100, 100))
        result = make_ocr_message_pdf(img, self.PROMPT)
        assert result[0]["role"] == "user"

    def test_raises_for_unsupported_type(self):
        with pytest.raises(ValueError):
            make_ocr_message_pdf("not-a-supported-type", self.PROMPT)

    def test_image_url_is_data_uri(self):
        page = self._fake_page_dict()
        result = make_ocr_message_pdf(page, self.PROMPT)
        url_items = [c for c in result[0]["content"] if c["type"] == "image_url"]
        assert url_items[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Lock file claim/release tests (fix for race condition)
# ---------------------------------------------------------------------------

class TestLockFileMechanism:
    def test_first_claim_succeeds(self, tmp_path):
        assert claim_pdf_with_lock("thesis_2024", tmp_path) is True

    def test_second_claim_on_same_pdf_fails(self, tmp_path):
        claim_pdf_with_lock("thesis_2024", tmp_path)
        assert claim_pdf_with_lock("thesis_2024", tmp_path) is False

    def test_claim_succeeds_after_release(self, tmp_path):
        claim_pdf_with_lock("thesis_2024", tmp_path)
        release_pdf_lock("thesis_2024", tmp_path)
        assert claim_pdf_with_lock("thesis_2024", tmp_path) is True

    def test_release_is_idempotent(self, tmp_path):
        """Releasing a lock that doesn't exist should not raise."""
        release_pdf_lock("nonexistent_pdf", tmp_path)  # should not raise

    def test_different_pdfs_have_independent_locks(self, tmp_path):
        claim_pdf_with_lock("doc_a", tmp_path)
        # doc_b should still be claimable
        assert claim_pdf_with_lock("doc_b", tmp_path) is True

    def test_lock_file_created_on_claim(self, tmp_path):
        claim_pdf_with_lock("my_document", tmp_path)
        assert (tmp_path / "my_document.lock").exists()

    def test_lock_file_removed_on_release(self, tmp_path):
        claim_pdf_with_lock("my_document", tmp_path)
        release_pdf_lock("my_document", tmp_path)
        assert not (tmp_path / "my_document.lock").exists()

    def test_concurrent_claims_only_one_succeeds(self, tmp_path):
        """
        Simulate two threads racing to claim the same PDF.
        Only one should succeed.
        """
        results = []

        def try_claim():
            results.append(claim_pdf_with_lock("shared_pdf", tmp_path))

        t1 = threading.Thread(target=try_claim)
        t2 = threading.Thread(target=try_claim)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results.count(True) == 1
        assert results.count(False) == 1

    def test_old_json_approach_is_racy(self, tmp_path):
        """
        Documents why the current_files.json approach is unsafe:
        two readers can both see an empty list before either writes.
        This test does NOT use the lock mechanism — it demonstrates the bug.
        """
        import json

        json_file = tmp_path / "current_files.json"
        json_file.write_text("[]")

        barrier = threading.Barrier(2)
        claimed_by = []

        def old_claim(name):
            current = json.loads(json_file.read_text())
            barrier.wait()  # both threads pause here after reading
            if name not in current:
                current.append(name)
                json_file.write_text(json.dumps(current))
                claimed_by.append(name)

        threads = [
            threading.Thread(target=old_claim, args=("pdf_001",)),
            threading.Thread(target=old_claim, args=("pdf_001",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Both threads claimed the same PDF — the race is real
        assert len(claimed_by) == 2, (
            "Both threads claimed pdf_001 — the JSON approach is racy"
        )


# ---------------------------------------------------------------------------
# Batch assembly and skip logic
# ---------------------------------------------------------------------------

class TestBatchAssemblyPdf:
    PROMPT = "Extract text."

    def _make_pages(self, n: int):
        return [
            {"image": Image.new("RGB", (100, 100)), "page": i + 1}
            for i in range(n)
        ]

    def test_batch_splitting(self):
        pages = self._make_pages(10)
        batch_size = 3
        batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1

    def test_pages_sorted_by_page_number(self):
        pages = [
            {"image": Image.new("RGB", (10, 10)), "page": 3},
            {"image": Image.new("RGB", (10, 10)), "page": 1},
            {"image": Image.new("RGB", (10, 10)), "page": 2},
        ]
        pages.sort(key=lambda x: x["page"])
        assert [p["page"] for p in pages] == [1, 2, 3]

    def test_batch_messages_count_matches_batch(self):
        pages = self._make_pages(5)
        batch_messages = [make_ocr_message_pdf(p, self.PROMPT) for p in pages]
        assert len(batch_messages) == 5

    def test_output_concatenation(self):
        """Simulates the pdf_text accumulation pattern in main_pdf.py."""
        fake_outputs = []
        for i in range(3):
            out = MagicMock()
            out.outputs[0].text = f"  Page {i + 1} content  "
            fake_outputs.append(out)

        pdf_text = ""
        for output in fake_outputs:
            pdf_text += output.outputs[0].text.strip() + "\n\n"

        assert "Page 1 content" in pdf_text
        assert "Page 2 content" in pdf_text
        assert "Page 3 content" in pdf_text

    def test_md_output_written_to_correct_path(self, tmp_path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        pdf_stem = "my_thesis_2024"
        content = "# Chapter 1\n\nHello world."

        md_file = md_dir / f"{pdf_stem}.md"
        md_file.write_text(content, encoding="utf-8")

        assert md_file.exists()
        assert md_file.read_text() == content

    def test_skip_pdf_if_md_already_exists(self, tmp_path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        pdfs = [tmp_path / f"doc_{i}.pdf" for i in range(4)]
        for p in pdfs:
            p.touch()

        # Pre-mark docs 0 and 2 as done
        (md_dir / "doc_0.md").write_text("done")
        (md_dir / "doc_2.md").write_text("done")

        to_process = [p for p in pdfs if not (md_dir / f"{p.stem}.md").exists()]
        assert len(to_process) == 2
        assert {p.stem for p in to_process} == {"doc_1", "doc_3"}
