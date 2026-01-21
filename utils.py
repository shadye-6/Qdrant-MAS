import os
import base64
import fitz
import torch
from io import BytesIO
from PIL import Image
import requests

from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# Image Captioning (VLM)
# =========================

_processor = None
_model = None

def initialize_vlm():
    global _processor, _model
    if _processor is None or _model is None:
        model_id = "Salesforce/blip-image-captioning-base"
        _processor = BlipProcessor.from_pretrained(model_id)
        _model = BlipForConditionalGeneration.from_pretrained(model_id)
    return _processor, _model


def describe_image(image_content):
    """
    Convert an image into a natural language caption using BLIP.
    """
    processor, model = initialize_vlm()

    image = Image.open(BytesIO(image_content)).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# =========================
# Optional Graph Detection
# =========================

def is_graph(image_content):
    return False


def prof(label):
    print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)

def process_graph(image_content, llm):
    prof("Graph processing start")
    deplot_description = process_graph_deplot(image_content)
    response = llm.complete(
        "Explain the following chart data:\n" + deplot_description
    )
    prof("Graph processing end")
    return response.text


# =========================
# NVIDIA Deplot (optional)
# =========================

def get_b64_image_from_content(image_content):
    img = Image.open(BytesIO(image_content)).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_graph_deplot(image_content):
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/google/deplot"
    image_b64 = get_b64_image_from_content(image_content)

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    payload = {
        "messages": [{
            "role": "user",
            "content": f'Generate underlying data table: <img src="data:image/png;base64,{image_b64}" />'
        }],
        "max_tokens": 512
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


# =========================
# PDF Helpers
# =========================

def extract_text_around_item(text_blocks, bbox, page_height, threshold_percentage=0.1):
    before_text, after_text = "", ""
    vertical_threshold_distance = page_height * threshold_percentage

    for block in text_blocks:
        block_bbox = fitz.Rect(block[:4])
        vertical_distance = min(abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))

        if vertical_distance <= vertical_threshold_distance:
            if block_bbox.y1 < bbox.y0 and not before_text:
                before_text = block[4]
            elif block_bbox.y0 > bbox.y1 and not after_text:
                after_text = block[4]
                break

    return before_text, after_text


def process_text_blocks(text_blocks, char_count_threshold=500):
    current_group = []
    grouped_blocks = []
    current_char_count = 0

    for block in text_blocks:
        if block[-1] == 0:
            block_text = block[4]
            block_char_count = len(block_text)

            if current_char_count + block_char_count <= char_count_threshold:
                current_group.append(block)
                current_char_count += block_char_count
            else:
                grouped_content = "\n".join([b[4] for b in current_group])
                grouped_blocks.append((current_group[0], grouped_content))
                current_group = [block]
                current_char_count = block_char_count

    if current_group:
        grouped_content = "\n".join([b[4] for b in current_group])
        grouped_blocks.append((current_group[0], grouped_content))

    return grouped_blocks


def save_uploaded_file(uploaded_file):
    temp_dir = os.path.join(os.getcwd(), "vectorstore", "ppt_references", "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    return temp_file_path
