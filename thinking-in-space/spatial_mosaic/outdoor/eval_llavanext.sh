#export CUDA_VISIBLE_DEVICES="0,1" # If you have multiple GPUs, you can set the actual GPU IDs, e.g., "0,1,2"
export LMMS_EVAL_LAUNCHER="accelerate"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THINKING_IN_SPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${THINKING_IN_SPACE_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${THINKING_IN_SPACE_DIR}:${PYTHONPATH:-}"
# If you have multiple GPUs, you can set --num_processes=the number of GPUs to use
accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model spatialmosaic \
    --model_args pretrained=path_to_model,model_base=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32 \
    --tasks ${THINKING_IN_SPACE_DIR}/lmms_eval/tasks/spatial_mosaic/outdoor \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_next_outdoor \
    --output_path logs/llava_next_outdoor
