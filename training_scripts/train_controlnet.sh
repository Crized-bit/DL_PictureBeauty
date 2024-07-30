#!/bin/bash

OUTPUT_PATH="/home/controlnet_versions/output_controlnet_test_fixed_prompt"

cp "$0" "$OUTPUT_PATH"/

CUDA_VISIBLE_DEVICES=3 accelerate launch --mixed_precision bf16 train_controlnet.py \
 --pretrained_model_name_or_path="/home/web_ui/stable-diffusion-webui/models/Stable-diffusion/juggernautXL_juggXIByRundiffusion.safetensors" \
 --output_dir=$OUTPUT_PATH \
 --train_data_dir="/home/Final_dataset_Games" \
 --conditioning_image_column=control_image \
 --image_column=initial_picture \
 --caption_column=caption \
 --resolution=1024 \
 --random_flip \
 --learning_rate=1e-4 \
 --gradient_accumulation_steps=2 \
 --validation_image "/home/dataset/val/1.jpg" \
 --snr_gamma=5 \
 --validation_prompt ""\
 --train_batch_size=2 \
 --num_train_epochs=150 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --checkpoints_total_limit=5 \
 --tracker_project_name="controlnet_training_fixed_prompt" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=1000 \
 --validation_steps=200 \
 --mixed_precision=bf16 \
 --report_to tensorboard \
 --seed 1337 \
 --controlnet_model_name_or_path="/home/Diffusion_VK/models/ControlNet_Tile" \
#  --max_train_samples=1 \
#  --resume_from_checkpoint latest \