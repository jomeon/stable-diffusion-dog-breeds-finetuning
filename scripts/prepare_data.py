import os
import shutil
import json

SOURCE_IMAGES_DIR = "data/images/Images"
TARGET_DATASET_DIR = "data/train_data"

# Format: "keyword_to_search_in_folder_name": "prompt_concept_name"
DESIRED_BREEDS = {
    "bluetick": "bluetick",         
    "pembroke": "corgi",              
    "pug": "pug",                     
    "samoyed": "samoyed",             
    "chow": "chow chow"               
}

def find_breed_folder(images_dir, keyword):
    """
    Searches the dataset directory for a folder containing the specific keyword.
    """
    if not os.path.exists(images_dir):
        return None
        
    for folder in os.listdir(images_dir):
        if keyword.lower() in folder.lower():
            return folder
            
    return None

def prepare_dataset():
    """
    Extracts images of the selected breeds and generates the metadata.jsonl file
    required by the Hugging Face Diffusers library.
    """
    if not os.path.exists(TARGET_DATASET_DIR):
        os.makedirs(TARGET_DATASET_DIR)
        
    metadata = []
    
    for keyword, breed_name in DESIRED_BREEDS.items():
        folder_name = find_breed_folder(SOURCE_IMAGES_DIR, keyword)
        
        if not folder_name:
            print(f"Error: Could not find a folder for '{breed_name}' (Keyword: {keyword})")
            continue
            
        breed_dir = os.path.join(SOURCE_IMAGES_DIR, folder_name)
        print(f"Processing breed: {breed_name} (Folder: {folder_name})...")
        
        images = os.listdir(breed_dir)
        
        for img_name in images:
            
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                new_img_name = f"{breed_name.replace(' ', '_')}_{img_name}"
                
                src_path = os.path.join(breed_dir, img_name)
                dst_path = os.path.join(TARGET_DATASET_DIR, new_img_name)
                
                shutil.copy2(src_path, dst_path)
                
                metadata.append({
                    "file_name": new_img_name,
                    "text": f"photo of {breed_name} dog"
                })

    metadata_path = os.path.join(TARGET_DATASET_DIR, "metadata.jsonl")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
            
    print(f"\nSuccess! Dataset prepared in: {TARGET_DATASET_DIR}")
    print(f"Total image-prompt pairs generated: {len(metadata)}")

if __name__ == "__main__":
    prepare_dataset()