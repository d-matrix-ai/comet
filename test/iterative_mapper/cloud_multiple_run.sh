#!/bin/bash

config_folder=${1:-"config"}
output_folder=${2:-"outputs"}
iter=${3:-"5"}

dims_list=(
  "M-256 N-256 K-512"
  "M-2048 N-2048 K-256"
  "M-1 N-4096 K-512"
  "M-1 N-8192 K-128"
  "M-1 N-16384 K-128"
  "M-1024 N-1024 K-512"
)

for dims in "${dims_list[@]}"; do
  echo "Running for dims: $dims"
  python generate_constants_cloud.py \
  	--constraints_file constraints_cloud.yaml\
    --config_folder_name "$config_folder" \
    --output_folder_name "$output_folder" \
    --num_iterations "$iter" \
    --dims "$dims"
done

