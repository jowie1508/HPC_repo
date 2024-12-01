#!/bin/sh
### General options
### -- set the job Name --
#BSUB -J physics
### -- ask for number of cores (default: 1) --
#BSUB -q gpua100
#BSUB -n 1
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 150GB of system-memory
#BSUB -R "rusage[mem=300GB]"

cd /zhome/14/b/214266/HPC_repo
source venv39/bin/activate

module load cuda/12.4
echo $CUDA_HOME
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

python3 clustering_physics_chunked.py