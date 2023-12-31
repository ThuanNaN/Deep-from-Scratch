#!/usr/bin/bash
python scripts/hf_image_classification_no_trainer.py \
    --dataset_name "cifar10" \
    --train_val_split 0.1 \
    --max_eval_samples 10 \
    --model_name_or_path "google/vit-base-patch16-224-in21k" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 0.0001 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type "linear" \
    --num_warmup_steps 0 \
    --output_dir "output" \
    --seed 0 
