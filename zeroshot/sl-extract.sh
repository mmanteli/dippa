#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --account=project_2009498
#SBATCH --time=06:15:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:a100:1,nvme:6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


export HF_HOME=/scratch/project_2009498/cache
split=$1
lang=$2
model=$3

#module purge
#module load pytorch/2.4
source .venv/bin/activate
python run_score_extraction.py --split=$1 --lang=$2 --model=$3
