from PIL import Image
from vllm import LLM, SamplingParams
from pathlib import Path
import pymupdf
from tqdm import tqdm
from typing import Any, Dict, List, Union
import io
import base64
import srsly
from pillow_heif import register_heif_opener

# Register the HEIF opener
register_heif_opener()

pdf_path = "pdfs"
md_path = "markdown"
model_repo = "nanonets/Nanonets-OCR-s"
# Read model path written by `fetch model`
model_info = srsly.read_json("model_info.json")
model: str = model_info[model_repo]['model_path']
batch_size = 32
max_tokens: int = 4096
max_model_len: int = 8192
gpu_memory_utilization: float = 0.9

llm = LLM(
    model=model,
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
    prompt: str = "Extract the text from the above document as if you were reading it naturally. Return the tables in markdown format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
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


pdfs = Path(pdf_path).glob('*')
Path(md_path).mkdir(parents=True, exist_ok=True)

for pdf in pdfs:

    md_file = Path(md_path) / f"{pdf.stem}.md"
    if md_file.exists():
        continue

    # Atomic claim using a .lock file (NFS-safe: O_CREAT|O_EXCL is atomic)
    lock_file = pdf.with_suffix('.lock')
    try:
        lock_file.touch(exist_ok=False)
    except FileExistsError:
        # Another job is already processing this PDF
        continue
    
    pdf_images = []
    try:
        doc = pymupdf.open(pdf)
        for i, page in tqdm(enumerate(doc)):  # iterate through the pages
            pix = page.get_pixmap(dpi=150)
            img = pix.pil_image()
            pdf_images.append({
                "image": img,
                "page": i + 1,
            })

        pdf_images.sort(key=lambda x: x["page"])
        image_batches = [
            pdf_images[i:i + batch_size] for i in range(0, len(pdf_images), batch_size)
        ]
        
        pdf_text = """"""
        for batch in tqdm(image_batches, desc=f"Processing {pdf.stem}"):
            batch_messages = [make_ocr_message(page["image"]) for page in batch]
            
            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract markdown from outputs
            for output in outputs:
                markdown_text = output.outputs[0].text.strip()
                pdf_text += markdown_text + "\n\n"     
        md_file.write_text(pdf_text, encoding='utf-8')
        lock_file.unlink(missing_ok=True)

    except Exception as e:
        print(f"Error opening {pdf}: {e}")
        lock_file.unlink(missing_ok=True)
        continue
