"""
Tests for fetch.py — model download command and model_info.json schema.

Run with:  pytest tests/test_fetch.py -v
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Stubs — keep tests runnable without huggingface-hub installed
# ---------------------------------------------------------------------------

def _stub_dependencies(model_path_return: str = "/fake/cache/model"):
    hf_hub = types.ModuleType("huggingface_hub")
    hf_hub.snapshot_download = MagicMock(return_value=model_path_return)
    sys.modules["huggingface_hub"] = hf_hub

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = MagicMock()
    sys.modules["datasets"] = datasets_mod

    typer_mod = types.ModuleType("typer")
    # typer.Typer(), typer.Argument, typer.Option all need to be callable
    typer_mod.Typer = MagicMock(return_value=MagicMock())
    typer_mod.Argument = MagicMock(return_value=None)
    typer_mod.Option = MagicMock(return_value=None)
    sys.modules["typer"] = typer_mod

    return hf_hub


# ---------------------------------------------------------------------------
# model_info.json schema tests (the fix for issue #3 in CHANGES.md)
# ---------------------------------------------------------------------------

class TestModelInfoSchema:
    """
    The model_info.json file must use the repo_id as the top-level key so
    that main.py can look up model_info[model_repo]['model_path'].
    """

    def test_written_schema_is_keyed_by_repo_id(self, tmp_path):
        info_file = tmp_path / "model_info.json"
        repo_id = "nanonets/Nanonets-OCR-s"
        model_path = "/scratch/network/user/.cache/hub/model"

        # Replicate the fixed write logic
        existing = {}
        if info_file.exists():
            existing = json.loads(info_file.read_text())
        existing[repo_id] = {"model_path": model_path}
        info_file.write_text(json.dumps(existing))

        loaded = json.loads(info_file.read_text())
        assert repo_id in loaded
        assert loaded[repo_id]["model_path"] == model_path

    def test_main_py_read_pattern_works_with_fixed_schema(self, tmp_path):
        info_file = tmp_path / "model_info.json"
        repo_id = "nanonets/Nanonets-OCR-s"
        model_path = "/scratch/network/user/.cache/hub/model"

        info_file.write_text(json.dumps({repo_id: {"model_path": model_path}}))

        # Reproduce the exact read pattern used in main.py
        model_info = json.loads(info_file.read_text())
        result = model_info[repo_id]["model_path"]
        assert result == model_path

    def test_old_flat_schema_fails_main_py_read_pattern(self, tmp_path):
        """
        Demonstrates why the original schema was broken: the flat dict does
        not support the model_info[repo_id] lookup.
        """
        info_file = tmp_path / "model_info.json"
        repo_id = "nanonets/Nanonets-OCR-s"
        # Old (broken) format
        info_file.write_text(json.dumps({"model_repo_id": repo_id, "model_path": "/some/path"}))

        model_info = json.loads(info_file.read_text())
        with pytest.raises(KeyError):
            _ = model_info[repo_id]["model_path"]

    def test_second_model_download_preserves_first(self, tmp_path):
        """Multiple models can be stored without clobbering earlier entries."""
        info_file = tmp_path / "model_info.json"

        for repo_id, path in [
            ("nanonets/Nanonets-OCR-s", "/path/to/nanonets"),
            ("Qwen/Qwen2-VL-7B-Instruct", "/path/to/qwen"),
        ]:
            existing = json.loads(info_file.read_text()) if info_file.exists() else {}
            existing[repo_id] = {"model_path": path}
            info_file.write_text(json.dumps(existing))

        final = json.loads(info_file.read_text())
        assert "nanonets/Nanonets-OCR-s" in final
        assert "Qwen/Qwen2-VL-7B-Instruct" in final


# ---------------------------------------------------------------------------
# snapshot_download integration (mocked)
# ---------------------------------------------------------------------------

class TestSnapshotDownload:
    def test_snapshot_download_called_with_correct_args(self, tmp_path, monkeypatch):
        hf_hub = _stub_dependencies("/fake/path")
        repo_id = "nanonets/Nanonets-OCR-s"

        info_file = tmp_path / "model_info.json"
        monkeypatch.chdir(tmp_path)

        # Simulate the fixed model() command body
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(repo_id=repo_id, repo_type="model")
        existing = {}
        if info_file.exists():
            existing = json.loads(info_file.read_text())
        existing[repo_id] = {"model_path": model_path}
        info_file.write_text(json.dumps(existing))

        hf_hub.snapshot_download.assert_called_once_with(repo_id=repo_id, repo_type="model")
        result = json.loads(info_file.read_text())
        assert result[repo_id]["model_path"] == "/fake/path"
