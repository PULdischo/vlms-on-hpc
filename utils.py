"""Shared utilities for vLLM OCR pipeline."""

from PIL import Image
from vllm import LLM, SamplingParams
from pathlib import Path
from typing import Any, Dict, List, Union
import io
import base64


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str, Path],
    prompt: str,
) -> List[Dict]:
    """Convert an image to a vLLM chat message with a base64 data URI.

    Accepts:
      - PIL.Image.Image
      - dict with a "bytes" key (e.g. from a HuggingFace dataset)
      - str or pathlib.Path pointing to an image file
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


def build_llm(
    model_path: str,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
) -> LLM:
    """Construct a vLLM engine with settings tuned for OCR workloads."""
    return LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,
    )


def build_sampling_params(max_tokens: int = 4096) -> SamplingParams:
    """Deterministic sampling parameters for OCR."""
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)
