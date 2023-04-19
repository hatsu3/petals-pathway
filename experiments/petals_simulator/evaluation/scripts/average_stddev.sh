#!/bin/bash

conda activate petals
parent_dir="evaluation/data"
python_script="evaluation/scripts/end2end_latency.py"

for exp_dir in "$parent_dir"/*/; do
  echo "$exp_dir"
  for run_dir in "$exp_dir"/*/; do
    # echo "$run_dir"
    python3 "$python_script" "$run_dir" >> "${exp_dir%?}".csv
  done

done
