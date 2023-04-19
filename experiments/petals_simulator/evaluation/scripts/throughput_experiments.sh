#!/bin/bash


conda activate petals

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
mkdir "evaluation/data/${timestamp}"

stage_assignment=(AllToAllStageAssignmentPolicy UniformStageAssignmentPolicy RequestRateStageAssignmentPolicy)
routing=(RandomRoutingPolicy QueueLengthRoutingPolicy RequestRateRoutingPolicy)
scheduling=(RandomSchedulingPolicy FIFOSchedulingPolicy LatencyAwareSchedulingPolicy)

for stage_assignment_policy in "${stage_assignment[@]}"; do
  for routing_policy in "${routing[@]}"; do
    for scheduling_policy in "${scheduling[@]}"; do
      for i in {10..100..10}; do
        directory="evaluation/data/${timestamp}/${stage_assignment_policy}_${routing_policy}_${scheduling_policy}_${i}"
        echo ${directory}
        mkdir "${directory}"
        python3 simulation.py --num-clients $i --num-servers 8 --stage-assignment "${stage_assignment_policy}" --routing "${routing_policy}" --scheduling "${scheduling_policy}" --prefix "${directory}" 
      done
    done
  done
done
