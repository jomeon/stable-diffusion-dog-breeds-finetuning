import os
import json
import shutil

SOURCE_DIR = "data/train_data"
TARGET_BASE = "data/train_separate"

def split_data():
    if not os.path.exists(TARGET_BASE):
        os.makedirs(TARGET_BASE)

    metadata_path = os.path.join(SOURCE_DIR, "metadata.jsonl")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)
        img_name = data["file_name"]
        prompt = data["text"]
        
        breed = prompt.replace("photo of ", "").replace(" dog", "").replace(" ", "_")
        breed_dir = os.path.join(TARGET_BASE, breed)
        
        if not os.path.exists(breed_dir):
            os.makedirs(breed_dir)
            
        shutil.copy2(os.path.join(SOURCE_DIR, img_name), os.path.join(breed_dir, img_name))
        
        with open(os.path.join(breed_dir, "metadata.jsonl"), "a", encoding="utf-8") as out_f:
            out_f.write(line)
            
    print(f"Success! Data separated into 5 races in the folder: {TARGET_BASE}")

if __name__ == "__main__":
    split_data()