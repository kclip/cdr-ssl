#!/bin/bash

#SBATCH --job-name=beamforming_dr
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --array=1-50
#SBATCH --gres=gpu:1

# Init env
module load texlive/20220321-gcc-13.2.0-python-3.11.6
module load anaconda3/2021.05-gcc-13.2.0
source $(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh
conda activate cppi

# Get command for this run
RUN_COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /users/k21139912/projects/contextual-ppi/study/protocols/commands/beamforming_dr.sh)

# Run
$RUN_COMMAND
