import re
import torch
from diffusers import FluxPipeline
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import uuid
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

check_min_version("0.30.2")

# text = """[SCENE-1] TOK mushroom folds the cozy quilt with care.
#     [SCENE-2] TOK mushroom smooths the sheets and fluffs the pillow.
#     [SCENE-3] TOK mushroom wipes the wooden nightstand and adjusts a tiny lamp.
#     [SCENE-4] TOK mushroom lights a scented candle and smiles with satisfaction."""
# width=512
# height=1024
# seed=42
# mask_scale=1.3

output_path="output"


def parse_aspect_ratio(ratio_str):
    parts = ratio_str.replace(" ", "").lower().split("x")
    if len(parts) != 2:
        raise ValueError("aspect_ratio format error")
    width = int(parts[0])
    height = int(parts[1])
    if not (0 < width <= 1024 and 0 < height <= 2048):
        raise ValueError("width or height exceed limit")
    return width, height

def process_prompt(text):
    """
    处理形如 [标签] 内容的文本，提取各段标签后的内容为列表。
    标签必须是 [xxx] 格式，不限定内容。
    """
    print("Processing prompt...")
    pattern = r"\s*\[.*?\]\s*(.*?)\s*(?=\[.*?\]|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    return [match.strip() for match in matches]


transformer = None
generator = None
basepipe = None
controlnet = None
inpaintpipe = None

def init(seed):
    transformer = FluxTransformer2DModel.from_pretrained(
        "model/FLUX.1-dev",
        subfolder='transformer',
        torch_dtype=torch.bfloat16
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Base generation pipeline
    basepipe = FluxPipeline.from_pretrained(
        "model/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    basepipe.load_lora_weights('model/tok-mushroom-1-2-simple-set-simple-loss-lora-rank-16-bs-32-v-1-0-8-step-800.safetensors')
    
    # Inpainting pipeline
    controlnet = FluxControlNetModel.from_pretrained(
        "model/alimama-creative",
        torch_dtype=torch.bfloat16
    )
    inpaintpipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "model/FLUX.1-dev",
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    inpaintpipe.load_lora_weights('model/tok-mushroom-1-2-simple-set-simple-loss-lora-rank-16-bs-32-v-1-0-8-step-800.safetensors')

# # Build basepipeline
# basepipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
# basepipe.load_lora_weights('tok-mushroom-1-2-simple-set-simple-loss-lora-rank-16-bs-32-v-1-0-8-step-800.safetensors')
# generator = torch.Generator(device="cuda").manual_seed(seed)

# # Build inpaintpipeline
# controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
# transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16)
# inpaintpipe = FluxControlNetInpaintingPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev",
#     controlnet=controlnet,
#     transformer=transformer,
#     torch_dtype=torch.bfloat16
# ).to("cuda")
# inpaintpipe.load_lora_weights('tok-mushroom-1-2-simple-set-simple-loss-lora-rank-16-bs-32-v-1-0-8-step-800.safetensors')



def base_generation(prompt, width, height, output_path="output"):
    print("Generate first image...")
    os.makedirs(output_path, exist_ok=True)
    
    image = basepipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=28,
        max_sequence_length=512,
        generator=generator
    ).images[0]
    
    random_name = f"{uuid.uuid4().hex[:8]}.png"
    image_path = os.path.join(output_path, random_name)
    image.save(image_path)
    return image_path

def detect_and_generate_mask(image_path, mask_scale=1.0, model_path="model/best.pt", output_path="output"):
    print("Detect_and_generate_mask...")
    model = YOLO(model_path)
    os.makedirs(output_path, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Fail to read when generate_mask: {image_path}")
        return

    h, w, _ = image.shape
    results = model(image_path)[0]
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)
        box_w = x2 - x1
        box_h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        new_w = box_w * mask_scale
        new_h = box_h * mask_scale
        new_x1 = int(max(cx - new_w / 2, 0))
        new_y1 = int(max(cy - new_h / 2, 0))
        new_x2 = int(min(cx + new_w / 2, w))
        new_y2 = int(min(cy + new_h / 2, h))
        cv2.rectangle(mask, (new_x1, new_y1), (new_x2, new_y2), color=255, thickness=-1)

    image_name = os.path.basename(image_path)
    base, _ = os.path.splitext(image_name)
    mask_name = f"{base}_mask.png"
    mask_path = os.path.join(output_path, mask_name)
    cv2.imwrite(mask_path, mask)

    return image_path, mask_path

def inpaint(prompts, image_path, mask_path, output_path="output"):
    print("Inpaint other images...")
    os.makedirs(output_path, exist_ok=True)

    image_name = os.path.basename(image_path)
    image = load_image(image_path).convert("RGB")
    mask = load_image(mask_path).convert("RGB")
    w, h = image.size

    save_list = []
    save_list.append(image_path)
    for idx, prompt in enumerate(prompts):
        result = inpaintpipe(
            prompt=prompt,
            height=h,
            width=w,
            control_image=image,
            control_mask=mask,
            num_inference_steps=28,
            generator=generator,
            controlnet_conditioning_scale=0.9,
            guidance_scale=3.5,
            negative_prompt="",
            true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
        ).images[0]

        save_name=f'{image_name}_inpaint_{idx}.png'
        save_path=os.path.join(output_path, save_name)
        result.save(save_path)
        save_list.append(save_path)

    return save_list

       
# prompt = process_prompt(text)
# image_path = base_generation(prompt[0],width,height)
# _, mask_path = detect_and_generate_mask(image_path, mask_scale=mask_scale)
# image_list = inpaint(prompt[1:4], image_path, mask_path)