#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "name"
#SBATCH -p normal
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1


conda activate carps

# Parse all command-line arguments as Hydra overrides or check for --multirun flag
HYDRA_OVERRIDES+=("$1")


# Construct the Hydra command
echo $CWD
HYDRA_CMD="python -m carps.run hydra.searchpath=[file:////hpi_parego/configs]" # possibly change to absolute path

# Add all Hydra overrides
for override in "${HYDRA_OVERRIDES[@]}"; do
    HYDRA_CMD+=" $override"
done

# Prepare final command with commit hash and multirun flag if specified
FINAL_CMD=$HYDRA_CMD
FINAL_CMD+=" -m"

export HYDRA_FULL_ERROR=1
echo "Executing: $FINAL_CMD"
eval "$FINAL_CMD"

wait