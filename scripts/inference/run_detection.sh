#!/bin/bash

# Exit on any error
set -e

# Constants
DATA_DIR="..."
SET_NAME="test"
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
CACHE_DIR="..."
OUTPUT_BASE_DIR="..."
SEED="42"
MAX_JOBS=1  # Total jobs running concurrently
GPU_COUNT=1  # Number of GPUs
JOBS_PER_GPU=1  # Number of jobs per GPU

# Languages to process
LANGUAGES=(
    "arabic" "chinese" "german" "turkish" "russian" 
)

# Model details
MODEL="..."
CHECKPOINT_BASE_DIR="..."
CHECKPOINT="${CHECKPOINT_BASE_DIR}/${MODEL}"

# Job counter initialization
job_counter=0

# Function to run evaluation
run_evaluation() {
    local language=$1
    local gpu_id=$2
    local output_log=$3

    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python evaluation_script.py \
        --data_file "$DATA_DIR/${language}/${language}_${SET_NAME}_data.json" \
        --checkpoint "$CHECKPOINT" \
        --save_mode "$MODEL" \
        --language "$language" \
        --set_name "$SET_NAME" \
        --model_name "$MODEL_NAME" \
        --cache_dir "$CACHE_DIR" \
        --output_base_dir "$OUTPUT_BASE_DIR" \
        --seed "$SEED" \
        > "$output_log" 2>&1 &
}

for LANGUAGE in "${LANGUAGES[@]}"; do
    echo "Processing language $LANGUAGE with model $MODEL"
    OUTPUT_LOG="${OUTPUT_BASE_DIR}/${LANGUAGE}/${MODEL}_output.log"

    mkdir -p "$(dirname "$OUTPUT_LOG")"

    GPU_ID=$(( (job_counter / JOBS_PER_GPU) % GPU_COUNT ))
    echo "Using GPU $GPU_ID for language $LANGUAGE"

    run_evaluation "$LANGUAGE" "$GPU_ID" "$OUTPUT_LOG"

    job_counter=$(( job_counter + 1 ))
    if (( job_counter % MAX_JOBS == 0 )); then
        wait
    fi
done
