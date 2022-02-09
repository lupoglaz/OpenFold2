#!/bin/bash
#SBATCH --job-name=OpenFold2Train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time 00:10:00
#SBATCH --mem=64G
#SBATCH --output=/home/g.derevyanko/Logs/OpenFold2/Train/OpenFold2Train_%j.log


module load gpu/cuda-11.3
# conda init bash
conda activate torch

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python training.py \
-dataset_dir /gpfs/gpfs0/g.derevyanko/OpenFold2Dataset/Features \
-log_dir TrainLog \
-model_name test \
-num_gpus 2 \
-num_nodes 1 \
-num_accum 2 \
-max_iter 10000
