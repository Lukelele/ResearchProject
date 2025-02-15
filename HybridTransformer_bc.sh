#!/bin/bash

#SBATCH --job-name=HybridTransformer
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --account=phys033185
##SBATCH --mail-user=to21072@bristol.ac.uk
##SBATCH --mail-type=END

# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

mamba activate denosing-data

# Record some potentially useful details about the job:
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"


NUM_DATA=1024
BATCH_SIZE=256
EPOCH=3000
MODEL_PATH="HybridTransformer_${NUM_DATA}_${BATCH_SIZE}_${EPOCH}"

srun python HybridTransformer_cmd.py $NUM_DATA $BATCH_SIZE $EPOCH $MODEL_PATH


# Output the end time
printf "\n\n"
echo "Ended on: $(date)"