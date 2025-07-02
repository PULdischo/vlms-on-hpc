from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline, Qwen2VLForConditionalGeneration
from pathlib import Path 

processor = AutoProcessor.from_pretrained(
    "nanonets/Nanonets-OCR-s", local_files_only=True
)
model = AutoModelForImageTextToText.from_pretrained(
    "nanonets/Nanonets-OCR-s", torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
)
pipe = pipeline(
    "image-text-to-text",
    model=model,
    processor=processor,
)

images = Path('img').glob('*.jpg')
for img_path in images:
    if img_path.with_suffix(".md").exists():
        continue

    img = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {
                            "type": "text",
                            "text": "Extract and return all the text from this image. Include all text elements and maintain the reading order. If there are tables, convert them to markdown format. If there are mathematical equations, convert them to LaTeX format.",
                        },
                    ],
        }
    ]

    results = pipe(messages, max_new_tokens=8096)
    generated_content = results[0]["generated_text"]
    text = next(msg["content"] for msg in reversed(generated_content) if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg)
    img_path.with_suffix(".md").write_text(text)
