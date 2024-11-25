#!/bin/sh
### General options
### -- set the job Name --
#BSUB -J jonas_150GB
### -- ask for number of cores (default: 1) --
#BSUB -q gpua100
#BSUB -n 1
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 150GB of system-memory
#BSUB -R "rusage[mem=150GB]"

source venv/bin/activate
pip install -r requirements.txt
cd examples/00_quick_start

module load cuda/12.2
echo $CUDA_HOME
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

python3 hpc_clustering.py