#!/bin/bash

config_folder=${1:-"config"}
output_folder=${2:-"outputs"}
iter=${3:-"5"}

dims_list=(
  "M-256 N-256 K-128"
#  "M-1024 N-1024 K-256"
#  "M-1 N-1024 K-128"
#  "M-1 N-512 K-256"
#  "M-1 N-2048 K-256"
#  "M-512 N-256 K-128"
)

for dims in "${dims_list[@]}"; do
  echo "Running for dims: $dims"
  python generate_constants.py \
    --constraints_file constraints.yaml\
    --config_folder_name "$config_folder" \
    --output_folder_name "$output_folder" \
    --num_iterations "$iter" \
    --dims "$dims"
done

