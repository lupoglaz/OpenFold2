#!/bin/bash
#SBATCH --job-name=DatasetConvert
#SBATCH --partition=cpu_small
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --time=03:00:00
#SBATCH --output=/home/g.derevyanko/Logs/OpenFold2/Dataset/tRosettaConvert_%j.log

python convert_rosetta_data.py