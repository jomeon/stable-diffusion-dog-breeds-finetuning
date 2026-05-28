# Ścieżki
$MODEL_NAME="runwayml/stable-diffusion-v1-5"
$BASE_DATA_DIR="data/train_separate"
$BASE_OUTPUT_DIR="models/lora_separate"

# Pobieramy listę folderów (każdy folder to jedna rasa)
$breeds = Get-ChildItem -Path $BASE_DATA_DIR -Directory

foreach ($breed in $breeds) {
    $breed_name = $breed.Name
    Write-Host "=================================================" -ForegroundColor Cyan
    Write-Host "Starting LoRA training for the breed: $breed_name" -ForegroundColor Green
    Write-Host "=================================================" -ForegroundColor Cyan
    
    $DATASET_DIR = "$BASE_DATA_DIR/$breed_name"
    $OUTPUT_DIR = "$BASE_OUTPUT_DIR/$breed_name"
    
    accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py `
      --pretrained_model_name_or_path=$MODEL_NAME `
      --train_data_dir=$DATASET_DIR `
      --dataloader_num_workers=0 `
      --resolution=512 `
      --random_flip `
      --train_batch_size=1 `
      --gradient_accumulation_steps=4 `
      --max_train_steps=1000 `
      --learning_rate=1e-04 `
      --lr_scheduler="constant" `
      --lr_warmup_steps=0 `
      --seed=42 `
      --output_dir=$OUTPUT_DIR `
      --mixed_precision="fp16" `
      --gradient_checkpointing
}