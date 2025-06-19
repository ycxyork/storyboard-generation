import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
import os

check_min_version("0.30.2")

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16)
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")



generator = torch.Generator(device="cuda").manual_seed(24)
pipe.load_lora_weights('tok-mushroom-1-2-simple-set-simple-loss-lora-rank-16-bs-32-v-1-0-8-step-800.safetensors')





def inpaint(image_path, mask_path, inpaint_path):
    filename = os.path.basename(image_path)
    
    # Inpaint
    prompts = [
        "TOK mushroom looks up with wide, curious eyes.",
        "TOK mushroom crouches down to examine something on the ground.",
        "TOK mushroom gently holds a tiny treasure in both hands.",
        "TOK mushroom lifts both hands with a bright smile of joy."
    ]
    # Load image and mask
    size = (512, 1024)
    image = load_image(image_path).convert("RGB").resize(size)
    mask = load_image(mask_path).convert("RGB").resize(size)
    
    for idx,prompt in enumerate(prompts):
        result = pipe(
            prompt=prompt,
            height=size[1],
            width=size[0],
            control_image=image,
            control_mask=mask,
            num_inference_steps=28,
            generator=generator,
            controlnet_conditioning_scale=0.9,
            guidance_scale=3.5,
            negative_prompt="",
            true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
        ).images[0]

        save_path=os.path.join(inpaint_path,f'{filename}_impaint_{idx}.png')
        result.save(save_path)
    print("Successfully inpaint image")

image_folder="images"
inpaint_folder = os.path.join(image_folder, "inpaint")
os.makedirs(inpaint_folder, exist_ok=True)
mask_folder = os.path.join(image_folder, "mask")

for filename in os.listdir(image_folder):
    if filename.endswith(".png") and not filename.endswith("_mask.png"):
        image_path = os.path.join(image_folder, filename)
            
        base, _ = os.path.splitext(filename)
        mask_filename = f"{base}_mask.png"
        mask_path = os.path.join(mask_folder, mask_filename)

        inpaint(image_path, mask_path, inpaint_folder)

                