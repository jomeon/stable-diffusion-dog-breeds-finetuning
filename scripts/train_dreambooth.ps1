$MODEL_NAME="runwayml/stable-diffusion-v1-5"
$INSTANCE_DIR="data/train_separate/corgi"
$OUTPUT_DIR="models/dreambooth_corgi"


accelerate launch diffusers/examples/dreambooth/train_dreambooth.py `
  --pretrained_model_name_or_path=$MODEL_NAME  `
  --instance_data_dir=$INSTANCE_DIR `
  --output_dir=$OUTPUT_DIR `
  --instance_prompt="a photo of sks dog" `
  --resolution=512 `
  --train_batch_size=1 `
  --gradient_accumulation_steps=4 `
  --learning_rate=2e-6 `
  --lr_scheduler="constant" `
  --lr_warmup_steps=0 `
  --max_train_steps=200 `
  --mixed_precision="fp16" `
  --gradient_checkpointing