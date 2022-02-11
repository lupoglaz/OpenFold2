#!/bin/bash
#SBATCH --job-name=DatasetConvert
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --time=30:00:00
#SBATCH --output=/home/g.derevyanko/Logs/OpenFold2/Dataset/tRosettaConvert_%j.log

module load gpu/cuda-11.3
conda activate torch

python convert_rosetta_data.py \
-rosetta_data_dir /gpfs/gpfs0/g.derevyanko/tRosettaDataset \
-fasta_dir /gpfs/gpfs0/g.derevyanko/OpenFold2Dataset/Sequences \
-pdb_dir /gpfs/gpfs0/g.derevyanko/OpenFold2Dataset/Structures \
-output_msa_dir /gpfs/gpfs0/g.derevyanko/OpenFold2Dataset/Alignments \
-output_feat_dir /gpfs/gpfs0/g.derevyanko/OpenFold2Dataset/Features \
-data_dir /gpfs/gpfs0/datasets/AlphaFold \
\
-hhsearch_binary_path /trinity/home/g.derevyanko/miniconda3/bin/hhsearch \
-hhblits_binary_path /trinity/home/g.derevyanko/miniconda3/bin/hhblits
