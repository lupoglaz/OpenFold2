#!/bin/sh

#SBATCH --ntasks=1
#SBATCH -t 04:00:00               # max runtime is 4 hours
#SBATCH -J  tensorboard_server    # name
#SBATCH -o //home/g.derevyanko/Logs/Tensorboard/tb-%J.out #TODO: Where to save your output

source /home/g.derevyanko/.bashrc
MODEL_DIR=/home/g.derevyanko/OpenFold2/TrainLog

#let ipnport=($UID-6025)%65274
let ipnport=6006
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

#module load cuda/8.0 #TODO: Your Cuda Module if required

tensorboard --logdir="${MODEL_DIR}" --port=$ipnport