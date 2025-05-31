#!/bin/bash

# Define variables
GPU_ID=0
TS="0,39"
BATCH_SIZE=1
GENERATOR="determ-indiv"
TRAINING_MODE="consistency_distillation"
SAMPLER="multistep_bsr_v2"
STEPS=40
MODEL_PATH="/weights/synth/cm_synth_burst8_sigma003_lpips.pt"
ATTENTION_RESOLUTIONS="32,16,8"
CLASS_COND=False
USE_SCALE_SHIFT_NORM=True
DROPOUT=0.0
IMAGE_SIZE=256
NUM_CHANNELS=128
NUM_HEAD_CHANNELS=64
NUM_RES_BLOCKS=2
NUM_SAMPLES=100
RESBLOCK_UPDOWN=True
USE_FP16=False
WEIGHT_SCHEDULE="uniform"
LEARN_SIGMA=True
BURST_SIZE=8
NUM_COND_FEATURES=48
TIMESTEP_RESPACING=1
INPUT_PATH="/data/synth/BurstSR/synth/SyntheticBurstVal"
CROP_SIZE=32
CALLBACK="save_image_callback"
SIGMA_MIN=0.00000001
SIGMA_MAX=0.03
ATTENTION_RESOLUTIONS=32,16,8


# Execute Python script
python sample_synth_BSR_consistency.py \
    --gpu_id $GPU_ID \
    --ts $TS \
    --batch_size $BATCH_SIZE \
    --generator $GENERATOR \
    --training_mode $TRAINING_MODE \
    --sampler $SAMPLER \
    --steps $STEPS \
    --model_path $MODEL_PATH \
    --attention_resolutions $ATTENTION_RESOLUTIONS \
    --class_cond $CLASS_COND \
    --use_scale_shift_norm $USE_SCALE_SHIFT_NORM \
    --dropout $DROPOUT \
    --image_size $IMAGE_SIZE \
    --num_channels $NUM_CHANNELS \
    --num_head_channels $NUM_HEAD_CHANNELS \
    --num_res_blocks $NUM_RES_BLOCKS \
    --num_samples $NUM_SAMPLES \
    --resblock_updown $RESBLOCK_UPDOWN \
    --use_fp16 $USE_FP16 \
    --weight_schedule $WEIGHT_SCHEDULE \
    --learn_sigma $LEARN_SIGMA \
    --burst_size $BURST_SIZE \
    --num_cond_features $NUM_COND_FEATURES \
    --timestep_respacing $TIMESTEP_RESPACING \
    --input_path $INPUT_PATH \
    --crop_size $CROP_SIZE \
    --callback $CALLBACK \
    --sigma_min $SIGMA_MIN \
    --sigma_max $SIGMA_MAX
