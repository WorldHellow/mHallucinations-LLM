#!/bin/bash

# Define a list of languages


languages=(
    'arabic' 'bangla' 'basque' 'cantonese' 'catalan' 'chinese' 'czech' 'esperanto' 'french'
    'finnish' 'german' 'hebrew' 'hindi' 'hungarian'
    )


# Define models
models=("qwen-small" "qwen" "aya" "llama" "mistral")

# Define an expanded array of GPU IDs, repeating each GPU 3 times
gpus=(
    0 0
    1 1
    2 2
    3 3
)

# Total number of GPU slots
gpu_count=${#gpus[@]}  # Should be 12

# Define the log directory
log_dir="logs"

# Create the log directory if it doesn't exist
if [ ! -d "${log_dir}" ]; then
    mkdir -p "${log_dir}"
    echo "Created log directory: ${log_dir}"
else
    echo "Log directory already exists: ${log_dir}"
fi

# Function to run inference for a specific language and model on a specific GPU
run_inference() {
    local language=$1
    local model=$2
    local gpu=$3
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${log_dir}/${language}_${model}_inference_${timestamp}.log"
    
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting inference for '${language}' using '${model}' model on GPU ${gpu}..."
    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${gpu} python inference.py --language "${language}" --model "${model}" > "${log_file}" 2>&1
    local exit_status=$?
    
    if [ ${exit_status} -eq 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Inference for '${language}' using '${model}' completed successfully. Log: ${log_file}"
    else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Inference for '${language}' using '${model}' failed with exit status ${exit_status}. Check log: ${log_file}"
    fi
}

# Initialize an array to keep track of PIDs
declare -a pids

# Initialize a counter for the tasks
task_counter=0

# Loop over languages and models
for language in "${languages[@]}"; do
    for model in "${models[@]}"; do
        # Get the GPU ID for this task
        gpu_index=$((task_counter % gpu_count))
        gpu_id=${gpus[$gpu_index]}

        # Run the inference in the background
        run_inference "${language}" "${model}" "${gpu_id}" &
        pid=$!
        pids+=($pid)
        echo "Started process with PID ${pid} on GPU ${gpu_id} for '${language}' and '${model}'"

        # Increment the task counter
        task_counter=$((task_counter + 1))

        # If we've started a number of tasks equal to the number of GPU slots, wait for any one to finish
        if [ ${#pids[@]} -ge ${gpu_count} ]; then
            echo "Maximum concurrent tasks reached (${gpu_count}). Waiting for a task to finish..."
            wait -n  # Wait for any one process to finish

            # Remove finished PIDs from the array
            updated_pids=()
            for current_pid in "${pids[@]}"; do
                if kill -0 "$current_pid" 2>/dev/null; then
                    updated_pids+=("$current_pid")
                else
                    echo "Process with PID ${current_pid} has finished."
                fi
            done
            pids=("${updated_pids[@]}")
        fi
    done
done

# Wait for any remaining background processes to finish
if [ ${#pids[@]} -gt 0 ]; then
    echo "Waiting for remaining processes to finish..."
    wait
fi

echo "Inference for all specified languages and models has completed."
