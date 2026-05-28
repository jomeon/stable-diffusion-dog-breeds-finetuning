$MODEL_NAME="runwayml/stable-diffusion-v1-5"
$DATASET_DIR="data/train_data"
$OUTPUT_DIR="models/lora_all_breeds"

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