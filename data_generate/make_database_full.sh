#!/bin/bash

# 1. Submit the shard processors
JOB_OUTPUT=$(sbatch make_database.slurm)
ARRAY_JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')

# 2. Schedule the compilation script to automatically fire once all array parts are successful
sbatch --dependency=afterok:$ARRAY_JOB_ID merge_database.slurm