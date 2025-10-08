from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline, Qwen2VLForConditionalGeneration
from vllm import LLM, SamplingParams
from pathlib import Path 
import pymupdf
from tqdm import tqdm
from typing import Any, Dict, List, Union
import io
import base64
import srsly

img_path = "img"
md_path = "markdown"
model: str = '/scratch/network/aj7878/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct-FP8/snapshots/888324b140e2cbbc2b780e0cb79ed4e13ed916f5'
batch_size = 32
max_tokens: int = 4096
max_model_len: int = 8192
gpu_memory_utilization: float = 0.9
prompt = Path('prompt.txt').read_text()

llm = LLM(
    model=model,
    trust_remote_code=True,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    limit_mm_per_prompt={"image": 1},
)

sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for OCR
    max_tokens=max_tokens,
)

def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str,
) -> List[Dict]:
    """Create chat message for OCR processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

# Load images, ignore if already processed, and split into batches
images = []
image_paths = Path(img_path).glob('*.jpg')
for img_path in image_paths:
    md_file = Path(md_path) / f"{img_path.stem}.md"
    if md_file.exists():
        continue
    images.append(img_path)
    
    current_files = srsly.read_json("current_files.json")
    if img_path.stem in current_files:
        continue
    else:
        current_files.append(img_path.stem)
        srsly.write_json("current_files.json", current_files)
   
image_batches = [
    images[i:i + batch_size] for i in range(0, len(images), batch_size)
]

for batch in image_batches:
    
    batch_messages = [make_ocr_message(str(page), prompt) for page in batch]
            
    # Process with vLLM
    outputs = llm.chat(batch_messages, sampling_params)

    # Extract markdown from outputs
    for output, img_path in zip(outputs, batch):
        markdown_text = output.outputs[0].text.strip()
        md_file = Path(md_path) / f"{img_path.stem}.md"
        md_file.write_text(markdown_text, encoding='utf-8')

        current_files = srsly.read_json("current_files.json")
        if img_path.stem in current_files:
            current_files.remove(img_path.stem)
            srsly.write_json("current_files.json", current_files)

print("thank you, that's all")
