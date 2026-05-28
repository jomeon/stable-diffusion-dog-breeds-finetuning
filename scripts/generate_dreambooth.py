import torch
from diffusers import StableDiffusionPipeline
import os


MODEL_PATH = "models/dreambooth_corgi" 
PROMPT = "a photo of sks dog, highly detailed, realistic, 4k"
OUTPUT_DIR = "outputs"

def generate_image():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading full DreamBooth model from: {MODEL_PATH}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        safety_checker=None
    )
    pipe.to("cuda")

    print(f"Generating image for prompt: '{PROMPT}'...")
    image = pipe(PROMPT, num_inference_steps=30, guidance_scale=7.5).images[0]
    
    save_path = os.path.join(OUTPUT_DIR, "dreambooth_corgi_test.png")
    image.save(save_path)
    print(f"Success! The image is saved at: {save_path}")

if __name__ == "__main__":
    generate_image()