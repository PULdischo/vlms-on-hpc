from PIL import Image
from vllm import LLM, SamplingParams
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Union
import io
import base64
import srsly
from pillow_heif import register_heif_opener

# Register the HEIF opener
register_heif_opener()


input_path = "images"
output_path = "markdown"
model_repo = "nanonets/Nanonets-OCR-s"
# Read model path written by `fetch model`
model_info = srsly.read_json("model_info.json")
model_path = model_info[model_repo]['model_path']
batch_size = 32
max_tokens: int = 4096
max_model_len: int = 8192
gpu_memory_utilization: float = 0.9

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    limit_mm_per_prompt={"image": 1},
    enable_prefix_caching=True,
)

sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for OCR
    max_tokens=max_tokens,
)

def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str
) -> List[Dict]:
    """Create chat message for OCR processing."""
    
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, (str, Path)):
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


def md_exists(file_path: Path) -> bool:
    return (Path(output_path) / f"{file_path.stem}.md").exists()

images_paths = list(Path(input_path).glob('*'))

image_batches = [
    images_paths[i:i + batch_size] for i in range(0, len(images_paths), batch_size)
]
prompt = Path("ocr_prompt.txt").read_text()

Path(output_path).mkdir(parents=True, exist_ok=True)

for batch in tqdm(image_batches, desc="Processing images"):
    pending = [(page, make_ocr_message(page, prompt)) for page in batch if not md_exists(page)]
    if not pending:
        continue
    pages, batch_messages = zip(*pending)

    # Process with vLLM
    outputs = llm.chat(list(batch_messages), sampling_params)

    # Write each output to its corresponding markdown file
    for page_path, output in zip(pages, outputs):
        markdown_text = output.outputs[0].text.strip()
        md_file = Path(output_path) / f"{page_path.stem}.md"
        md_file.write_text(markdown_text, encoding='utf-8')
