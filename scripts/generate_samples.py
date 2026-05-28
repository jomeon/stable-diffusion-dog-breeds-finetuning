import torch
from diffusers import StableDiffusionPipeline
import os


MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "models/lora_separate/corgi" 
PROMPT = "a photo of corgi dog, highly detailed, realistic, 4k"
OUTPUT_DIR = "outputs"

def generate_image():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading base model (takes a while)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        safety_checker=None
    )
    pipe.to("cuda")

    print(f"Applying your LoRA scales from: {LORA_PATH}...")
    pipe.load_lora_weights(LORA_PATH)

    print(f"Generating an image for the prompt: '{PROMPT}'...")

    image = pipe(PROMPT, num_inference_steps=30, guidance_scale=7.5).images[0]
    
    path_parts = LORA_PATH.split('/')


    dynamic_filename = f"{path_parts[1]}_{path_parts[2]}_test.png"
    
    save_path = os.path.join(OUTPUT_DIR, dynamic_filename)
    image.save(save_path)
    print(f"Success! Your generated dog is waiting in the file: {save_path}")

if __name__ == "__main__":
    generate_image()