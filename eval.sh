#!/usr/bin/env bash
set -euo pipefail

# User parameters
HAB_GPU_ID=8
RUN_GPU_IDS="0,1,2,3,4,5,6,7"
SAVE_PATHS="\
/path/to/dir1\
"
MODEL_IDS="\
/path/to/model1,\
"

# --- helpers ---
split_csv_strip_ws() {  # usage: split_csv_strip_ws "var" "csv_string"
  local __out=$1 s=$2
  s=${s//[[:space:]]/}
  IFS=',' read -r -a "$__out" <<< "$s"
}

# Generate other parameters
split_csv_strip_ws RUNS     "$RUN_GPU_IDS"
split_csv_strip_ws SAVE_ARR "$SAVE_PATHS"
split_csv_strip_ws MODEL_ARR "$MODEL_IDS"
NUM_GPUS=${#RUNS[@]}

if [[ ${#SAVE_ARR[@]} -ne ${#MODEL_ARR[@]} ]]; then
  echo "ERROR: SAVE_PATHS and MODEL_IDS must have the same length (got ${#SAVE_ARR[@]} and ${#MODEL_ARR[@]})." >&2
  exit 1
fi

today=$(date +%m%d)

for idx in "${!SAVE_ARR[@]}"; do
  save_path="${SAVE_ARR[$idx]}"
  model_id="${MODEL_ARR[$idx]}"
  mkdir -p "$save_path"

  echo "==> Launching training for save_path='${save_path}' model='${model_id}'"
  pids=()

  for i in "${!RUNS[@]}"; do
    subset_id=$((i + 1))
    gpu="${RUNS[$i]}"
    export CUDA_VISIBLE_DEVICES="${gpu},${HAB_GPU_ID}"

    log="${save_path}/log_${today}_subset_${subset_id}.txt"

    nohup python lhvln/run.py \
      --cfg_file lhvln/configs/infer.yaml \
      --subset_id "$subset_id" \
      --num_gpus "$NUM_GPUS" \
      --save_path "$save_path" \
      --model_id "$model_id" > "$log" 2>&1 &

    pids+=("$!")
    echo "launched: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} --subset_id ${subset_id} -> ${log} (pid=${pids[-1]})"
  done

  echo "==> Waiting for training (save_path='${save_path}') to finish..."
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "==> Training finished for save_path='${save_path}'."

  echo "==> Running compute_metrics for save_path='${save_path}' ..."
  nohup python lhvln/compute_metrics.py --res_dir "$save_path" > "${save_path}/res.txt" 2>&1 &
  echo "==> Metrics done for save_path='${save_path}'."
done
