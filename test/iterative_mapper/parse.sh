#!/bin/bash

cloud=${1:-false}

if [ "$cloud" == "true" ]; then
    pattern="simulation_cloud_"
    log_file="out_parse_cloud.log"
else
    pattern="simulation_edge_"
    log_file="out_parse_edge.log"
fi

for folder in */ ; do
  folder_name="${folder%/}"

  if [[ -d "$folder_name" && "$folder_name" == *"$pattern"* ]]; then
    echo "Processing folder: $folder_name" | tee -a "$log_file"
    python parse_results.py --folder_name "$folder_name" 2>&1 | tee -a "$log_file"
    python parse_breakdowns.py --folder_name "$folder_name" 2>&1 | tee -a "$log_file"
  fi
done
