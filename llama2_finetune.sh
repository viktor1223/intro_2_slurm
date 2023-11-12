#!/bin/zsh

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2   # This needs to match Trainer(num_nodes=...)
#SBATCH --account=<ENTER ACCOUNT NAME>
#SBATCH --partition=<ENTER PARTITION NAME>
#SBATCH --gres=gpu:2   # number of gpus per node | This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=2  
#SBATCH --time=02:00:00
#SBATCH --job-name=llama2_finetune #change name to match job
#SBATCH --output=slurmout/%x-%j.out
#SBATCH --error=slurmout/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<ENTER YOU EMAIL>

# this is for auto resubmit

# activate conda env
module load anaconda/2020.11-py38
module load use.own
conda activate elab_2

export TRANSFORMERS_CACHE=<SET DIRECTORY TO SAVE MDOE AND DATAL>
export HF_HOME=<SET DIRECTORY TO SAVE MODEL AND DATA>
export HUGGING_FACE_HUB_TOKEN=<SET HUGGING FACE TOKEN>

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="ib"

# run script from above
srun python3 llama2_finetune.py