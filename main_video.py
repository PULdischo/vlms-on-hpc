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
prompt = Path('video_prompt.txt').read_text()

llm = LLM(
    model=model,
    trust_remote_code=True,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
)

sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for OCR
    max_tokens=max_tokens,
)


# Load images, ignore if already processed, and split into batches
image_paths =[f"file://{str(i)}" for i in Path(img_path).glob('*.jpg')]
messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": image_paths,
                    "fps": 1.0, 
                },
                {"type": "text", "text": prompt},
            ],
        }
]   
    
        
# Process with vLLM
outputs = llm.chat(messages, sampling_params)

Path('output.txt').write_text(outputs[0]['message']['content'])
print("thank you, that's all")
