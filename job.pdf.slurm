#!/bin/bash
#SBATCH --job-name=thesis-processing    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                 # total memory (RAM) per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=%u@princeton.edu

module purge
module load anaconda3/2025.6
conda activate vlm
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
python main_pdf.py

