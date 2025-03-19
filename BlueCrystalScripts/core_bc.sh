#!/bin/bash

#SBATCH --job-name=DenoisingData
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --account=phys033185
#SBATCH --mail-user=to21072@bristol.ac.uk
#SBATCH --mail-type=END

# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job:
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

. ~/initMamba.sh
conda activate denosing-data

ENV_FILE_NAME=".env1"
MODEL_NAME="HybridTransformer"

srun python core.py $ENV_FILE_NAME $MODEL_NAME

# Output the end time
printf "\n\n"
echo "Ended on: $(date)"