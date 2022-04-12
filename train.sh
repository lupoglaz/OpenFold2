#!/bin/bash
#SBATCH --job-name=OpenFold2Train
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time 3-00:00:01
#SBATCH --mem=64G
#SBATCH --output=/home/g.derevyanko/Logs/OpenFold2/Train/OpenFold2Train_%j.log


module load gpu/cuda-11.3
module load compilers/gcc-8.3.0
# conda init bash
conda activate torch

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python training.py \
-dataset_dir /gpfs/gpfs0/g.derevyanko/OpenFold2Dataset/Features \
-log_dir TrainLog \
-model_name model_big \
-num_gpus 4 \
-num_nodes 4 \
-num_accum 4 \
-max_iter 1500000 \
-precision 16 \
-progress_bar 0
