#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "$REPO_ROOT"

export WANDB_SILENT="${WANDB_SILENT:-true}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-30000}"

# Data paths
# FRAME_FOLDER, IMAGE_FOLDER, VIDEO_FOLDER can all point to spatial_mosaic_dataset/scannetpp.
FRAME_FOLDER="path_to_data"
IMAGE_FOLDER="path_to_data"
VIDEO_FOLDER="path_to_data"
DATA_YAML="${SCRIPT_DIR}/indoor.yaml"

SUFFIX="suffix" 
NUM_TRAIN_EPOCHS=5
SAVE_TOTAL_LIMIT=5
SPATIAL_TOWER="vggt"
FUSION_BLOCK="cross_attention" 
SPATIAL_TOWER_SELECT_FEATURE="all"
SPATIAL_FEATURE_DIM="2048" #
TUNE_MM_MLP_ADAPTER=True
GRADIENT_ACCUMULATION_STEPS=128

################ Arnold Jobs ################
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llava_video_7b_qwen2_${SUFFIX}"
BASE_MODEL="lmms-lab/LLaVA-Video-7B-Qwen2"

TRAIN_SCRIPT="${REPO_ROOT}/llava/train/train_mem.py"
DEEPSPEED_CONFIG="${REPO_ROOT}/scripts/zero2_torch_adam.json"
OUTPUT_DIR="${REPO_ROOT}/work_dirs_auto_eval/${MID_RUN_NAME}"


torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path $BASE_MODEL \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --spatial_tower $SPATIAL_TOWER \
    --spatial_tower_select_feature $SPATIAL_TOWER_SELECT_FEATURE \
    --spatial_feature_dim $SPATIAL_FEATURE_DIM \
    --fusion_block $FUSION_BLOCK \
    --tune_spatial_tower False \
    --tune_fusion_block True \
    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --frame_folder $FRAME_FOLDER \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $SUFFIX \
    --output_dir work_dirs_auto_eval/$MID_RUN_NAME \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0; 